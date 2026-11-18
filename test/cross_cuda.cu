/**
 *  @file test/cross_cuda.cu
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batch operation tests - CUDA capabilities.
 *
 *  Compiled with every capability some listed architecture runs: the CUDA baseline, Ampere, Hopper,
 *  Blackwell and Blackwell RTX.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "cross_device.hpp"
#include "geospatial.hpp" // `test_geodesics_pack`, `test_geodesics_packed`, `test_terrain_packed`

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_CUDA_

void run_cross_cuda_cases() {
    auto const cuda = nk_cap_cuda_k;
    run_kernel_case("nk_dots_packed_f64_cuda", test_dots_packed<f64_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_f64_cuda, nk_dots_pack_f64_cuda, nk_dots_packed_f64_cuda);
    run_kernel_case("nk_dots_pack_f64_cuda", test_dots_pack_layout<f64_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_f64_cuda, nk_dots_packed_shape_f64_cuda, nk_dots_pack_f64_cuda);
    run_kernel_case("test_dots_launch_contract<f64_t, cuda_backend_t>(nk_dots_packed_f64_cuda)",
                    test_dots_launch_contract<f64_t, cuda_backend_t>, cuda, nk_dots_pack_size_f64_cuda,
                    nk_dots_packed_f64_cuda, nk_dots_symmetric_f64_cuda);
    run_kernel_case("nk_dots_symmetric_f64_cuda", test_dots_symmetric<f64_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_f64_cuda);
    run_kernel_case("nk_dots_packed_f32_cuda", test_dots_packed<f32_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_f32_cuda, nk_dots_pack_f32_cuda, nk_dots_packed_f32_cuda);
    run_kernel_case("nk_dots_pack_f32_cuda", test_dots_pack_layout<f32_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_f32_cuda, nk_dots_packed_shape_f32_cuda, nk_dots_pack_f32_cuda);
    run_kernel_case("test_dots_launch_contract<f32_t, cuda_backend_t>(nk_dots_packed_f32_cuda)",
                    test_dots_launch_contract<f32_t, cuda_backend_t>, cuda, nk_dots_pack_size_f32_cuda,
                    nk_dots_packed_f32_cuda, nk_dots_symmetric_f32_cuda);
    run_kernel_case("nk_dots_symmetric_f32_cuda", test_dots_symmetric<f32_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_f32_cuda);
    run_kernel_case("nk_dots_packed_bf16_cuda", test_dots_packed<bf16_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_bf16_cuda, nk_dots_pack_bf16_cuda, nk_dots_packed_bf16_cuda);
    run_kernel_case("nk_dots_pack_bf16_cuda", test_dots_pack_layout<bf16_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_bf16_cuda, nk_dots_packed_shape_bf16_cuda, nk_dots_pack_bf16_cuda);
    run_kernel_case("test_dots_launch_contract<bf16_t, cuda_backend_t>(nk_dots_packed_bf16_cuda)",
                    test_dots_launch_contract<bf16_t, cuda_backend_t>, cuda, nk_dots_pack_size_bf16_cuda,
                    nk_dots_packed_bf16_cuda, nk_dots_symmetric_bf16_cuda);
    run_kernel_case("nk_dots_symmetric_bf16_cuda", test_dots_symmetric<bf16_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_bf16_cuda);
    run_kernel_case("nk_dots_packed_f16_cuda", test_dots_packed<f16_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_f16_cuda, nk_dots_pack_f16_cuda, nk_dots_packed_f16_cuda);
    run_kernel_case("nk_dots_pack_f16_cuda", test_dots_pack_layout<f16_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_f16_cuda, nk_dots_packed_shape_f16_cuda, nk_dots_pack_f16_cuda);
    run_kernel_case("test_dots_launch_contract<f16_t, cuda_backend_t>(nk_dots_packed_f16_cuda)",
                    test_dots_launch_contract<f16_t, cuda_backend_t>, cuda, nk_dots_pack_size_f16_cuda,
                    nk_dots_packed_f16_cuda, nk_dots_symmetric_f16_cuda);
    run_kernel_case("nk_dots_symmetric_f16_cuda", test_dots_symmetric<f16_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_f16_cuda);
    run_kernel_case("nk_dots_packed_e5m2_cuda", test_dots_packed<e5m2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e5m2_cuda, nk_dots_pack_e5m2_cuda, nk_dots_packed_e5m2_cuda);
    run_kernel_case("nk_dots_pack_e5m2_cuda", test_dots_pack_layout<e5m2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e5m2_cuda, nk_dots_packed_shape_e5m2_cuda, nk_dots_pack_e5m2_cuda);
    run_kernel_case("test_dots_launch_contract<e5m2_t, cuda_backend_t>(nk_dots_packed_e5m2_cuda)",
                    test_dots_launch_contract<e5m2_t, cuda_backend_t>, cuda, nk_dots_pack_size_e5m2_cuda,
                    nk_dots_packed_e5m2_cuda, nk_dots_symmetric_e5m2_cuda);
    run_kernel_case("nk_dots_symmetric_e5m2_cuda", test_dots_symmetric<e5m2_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_e5m2_cuda);
    run_kernel_case("nk_dots_packed_e4m3_cuda", test_dots_packed<e4m3_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e4m3_cuda, nk_dots_pack_e4m3_cuda, nk_dots_packed_e4m3_cuda);
    run_kernel_case("nk_dots_pack_e4m3_cuda", test_dots_pack_layout<e4m3_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e4m3_cuda, nk_dots_packed_shape_e4m3_cuda, nk_dots_pack_e4m3_cuda);
    run_kernel_case("test_dots_launch_contract<e4m3_t, cuda_backend_t>(nk_dots_packed_e4m3_cuda)",
                    test_dots_launch_contract<e4m3_t, cuda_backend_t>, cuda, nk_dots_pack_size_e4m3_cuda,
                    nk_dots_packed_e4m3_cuda, nk_dots_symmetric_e4m3_cuda);
    run_kernel_case("nk_dots_symmetric_e4m3_cuda", test_dots_symmetric<e4m3_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_e4m3_cuda);
    run_kernel_case("nk_dots_packed_e3m2_cuda", test_dots_packed<e3m2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e3m2_cuda, nk_dots_pack_e3m2_cuda, nk_dots_packed_e3m2_cuda);
    run_kernel_case("nk_dots_pack_e3m2_cuda", test_dots_pack_layout<e3m2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e3m2_cuda, nk_dots_packed_shape_e3m2_cuda, nk_dots_pack_e3m2_cuda);
    run_kernel_case("test_dots_launch_contract<e3m2_t, cuda_backend_t>(nk_dots_packed_e3m2_cuda)",
                    test_dots_launch_contract<e3m2_t, cuda_backend_t>, cuda, nk_dots_pack_size_e3m2_cuda,
                    nk_dots_packed_e3m2_cuda, nk_dots_symmetric_e3m2_cuda);
    run_kernel_case("nk_dots_symmetric_e3m2_cuda", test_dots_symmetric<e3m2_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_e3m2_cuda);
    run_kernel_case("nk_dots_packed_e2m3_cuda", test_dots_packed<e2m3_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e2m3_cuda, nk_dots_pack_e2m3_cuda, nk_dots_packed_e2m3_cuda);
    run_kernel_case("nk_dots_pack_e2m3_cuda", test_dots_pack_layout<e2m3_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e2m3_cuda, nk_dots_packed_shape_e2m3_cuda, nk_dots_pack_e2m3_cuda);
    run_kernel_case("test_dots_launch_contract<e2m3_t, cuda_backend_t>(nk_dots_packed_e2m3_cuda)",
                    test_dots_launch_contract<e2m3_t, cuda_backend_t>, cuda, nk_dots_pack_size_e2m3_cuda,
                    nk_dots_packed_e2m3_cuda, nk_dots_symmetric_e2m3_cuda);
    run_kernel_case("nk_dots_symmetric_e2m3_cuda", test_dots_symmetric<e2m3_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_e2m3_cuda);
    run_kernel_case("nk_dots_packed_e2m1_cuda", test_dots_packed<e2m1x2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e2m1_cuda, nk_dots_pack_e2m1_cuda, nk_dots_packed_e2m1_cuda);
    run_kernel_case("nk_dots_pack_e2m1_cuda", test_dots_pack_layout<e2m1x2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e2m1_cuda, nk_dots_packed_shape_e2m1_cuda, nk_dots_pack_e2m1_cuda);
    run_kernel_case("test_dots_launch_contract<e2m1x2_t, cuda_backend_t>(nk_dots_packed_e2m1_cuda)",
                    test_dots_launch_contract<e2m1x2_t, cuda_backend_t>, cuda, nk_dots_pack_size_e2m1_cuda,
                    nk_dots_packed_e2m1_cuda, nk_dots_symmetric_e2m1_cuda);
    run_kernel_case("nk_dots_symmetric_e2m1_cuda", test_dots_symmetric<e2m1x2_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_e2m1_cuda);
    run_kernel_case("nk_dots_packed_nvfp4_cuda", test_dots_packed<nvfp4_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_nvfp4_cuda, nk_dots_pack_nvfp4_cuda, nk_dots_packed_nvfp4_cuda);
    run_kernel_case("nk_dots_pack_nvfp4_cuda", test_dots_pack_layout<nvfp4_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_nvfp4_cuda, nk_dots_packed_shape_nvfp4_cuda, nk_dots_pack_nvfp4_cuda);
    run_kernel_case("test_dots_launch_contract<nvfp4_t, cuda_backend_t>(nk_dots_packed_nvfp4_cuda)",
                    test_dots_launch_contract<nvfp4_t, cuda_backend_t>, cuda, nk_dots_pack_size_nvfp4_cuda,
                    nk_dots_packed_nvfp4_cuda, nk_dots_symmetric_nvfp4_cuda);
    run_kernel_case("nk_dots_symmetric_nvfp4_cuda", test_dots_symmetric<nvfp4_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_nvfp4_cuda);
    run_kernel_case("nk_dots_packed_mxfp4_cuda", test_dots_packed<mxfp4_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_mxfp4_cuda, nk_dots_pack_mxfp4_cuda, nk_dots_packed_mxfp4_cuda);
    run_kernel_case("nk_dots_pack_mxfp4_cuda", test_dots_pack_layout<mxfp4_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_mxfp4_cuda, nk_dots_packed_shape_mxfp4_cuda, nk_dots_pack_mxfp4_cuda);
    run_kernel_case("test_dots_launch_contract<mxfp4_t, cuda_backend_t>(nk_dots_packed_mxfp4_cuda)",
                    test_dots_launch_contract<mxfp4_t, cuda_backend_t>, cuda, nk_dots_pack_size_mxfp4_cuda,
                    nk_dots_packed_mxfp4_cuda, nk_dots_symmetric_mxfp4_cuda);
    run_kernel_case("nk_dots_symmetric_mxfp4_cuda", test_dots_symmetric<mxfp4_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_mxfp4_cuda);
    run_kernel_case("nk_dots_packed_mxfp8e4m3_cuda", test_dots_packed<mxfp8e4m3_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_mxfp8e4m3_cuda, nk_dots_pack_mxfp8e4m3_cuda, nk_dots_packed_mxfp8e4m3_cuda);
    run_kernel_case("nk_dots_pack_mxfp8e4m3_cuda", test_dots_pack_layout<mxfp8e4m3_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_mxfp8e4m3_cuda, nk_dots_packed_shape_mxfp8e4m3_cuda, nk_dots_pack_mxfp8e4m3_cuda);
    run_kernel_case("test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t>(nk_dots_packed_mxfp8e4m3_cuda)",
                    test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t>, cuda, nk_dots_pack_size_mxfp8e4m3_cuda,
                    nk_dots_packed_mxfp8e4m3_cuda, nk_dots_symmetric_mxfp8e4m3_cuda);
    run_kernel_case("nk_dots_symmetric_mxfp8e4m3_cuda", test_dots_symmetric<mxfp8e4m3_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_mxfp8e4m3_cuda);
    run_kernel_case("nk_dots_packed_mxfp8e5m2_cuda", test_dots_packed<mxfp8e5m2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_mxfp8e5m2_cuda, nk_dots_pack_mxfp8e5m2_cuda, nk_dots_packed_mxfp8e5m2_cuda);
    run_kernel_case("nk_dots_pack_mxfp8e5m2_cuda", test_dots_pack_layout<mxfp8e5m2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_mxfp8e5m2_cuda, nk_dots_packed_shape_mxfp8e5m2_cuda, nk_dots_pack_mxfp8e5m2_cuda);
    run_kernel_case("test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t>(nk_dots_packed_mxfp8e5m2_cuda)",
                    test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t>, cuda, nk_dots_pack_size_mxfp8e5m2_cuda,
                    nk_dots_packed_mxfp8e5m2_cuda, nk_dots_symmetric_mxfp8e5m2_cuda);
    run_kernel_case("nk_dots_symmetric_mxfp8e5m2_cuda", test_dots_symmetric<mxfp8e5m2_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_mxfp8e5m2_cuda);
    run_kernel_case("nk_dots_packed_i8_cuda", test_dots_packed<i8_t, cuda_backend_t>, cuda, nk_dots_pack_size_i8_cuda,
                    nk_dots_pack_i8_cuda, nk_dots_packed_i8_cuda);
    run_kernel_case("nk_dots_pack_i8_cuda", test_dots_pack_layout<i8_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_i8_cuda, nk_dots_packed_shape_i8_cuda, nk_dots_pack_i8_cuda);
    run_kernel_case("test_dots_launch_contract<i8_t, cuda_backend_t>(nk_dots_packed_i8_cuda)",
                    test_dots_launch_contract<i8_t, cuda_backend_t>, cuda, nk_dots_pack_size_i8_cuda,
                    nk_dots_packed_i8_cuda, nk_dots_symmetric_i8_cuda);
    run_kernel_case("nk_dots_symmetric_i8_cuda", test_dots_symmetric<i8_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_i8_cuda);
    run_kernel_case("nk_dots_packed_u8_cuda", test_dots_packed<u8_t, cuda_backend_t>, cuda, nk_dots_pack_size_u8_cuda,
                    nk_dots_pack_u8_cuda, nk_dots_packed_u8_cuda);
    run_kernel_case("nk_dots_pack_u8_cuda", test_dots_pack_layout<u8_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_u8_cuda, nk_dots_packed_shape_u8_cuda, nk_dots_pack_u8_cuda);
    run_kernel_case("test_dots_launch_contract<u8_t, cuda_backend_t>(nk_dots_packed_u8_cuda)",
                    test_dots_launch_contract<u8_t, cuda_backend_t>, cuda, nk_dots_pack_size_u8_cuda,
                    nk_dots_packed_u8_cuda, nk_dots_symmetric_u8_cuda);
    run_kernel_case("nk_dots_symmetric_u8_cuda", test_dots_symmetric<u8_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_u8_cuda);
    run_kernel_case("nk_dots_packed_i4_cuda", test_dots_packed<i4x2_t, cuda_backend_t>, cuda, nk_dots_pack_size_i4_cuda,
                    nk_dots_pack_i4_cuda, nk_dots_packed_i4_cuda);
    run_kernel_case("nk_dots_pack_i4_cuda", test_dots_pack_layout<i4x2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_i4_cuda, nk_dots_packed_shape_i4_cuda, nk_dots_pack_i4_cuda);
    run_kernel_case("test_dots_launch_contract<i4x2_t, cuda_backend_t>(nk_dots_packed_i4_cuda)",
                    test_dots_launch_contract<i4x2_t, cuda_backend_t>, cuda, nk_dots_pack_size_i4_cuda,
                    nk_dots_packed_i4_cuda, nk_dots_symmetric_i4_cuda);
    run_kernel_case("nk_dots_symmetric_i4_cuda", test_dots_symmetric<i4x2_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_i4_cuda);
    run_kernel_case("nk_dots_packed_u4_cuda", test_dots_packed<u4x2_t, cuda_backend_t>, cuda, nk_dots_pack_size_u4_cuda,
                    nk_dots_pack_u4_cuda, nk_dots_packed_u4_cuda);
    run_kernel_case("nk_dots_pack_u4_cuda", test_dots_pack_layout<u4x2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_u4_cuda, nk_dots_packed_shape_u4_cuda, nk_dots_pack_u4_cuda);
    run_kernel_case("test_dots_launch_contract<u4x2_t, cuda_backend_t>(nk_dots_packed_u4_cuda)",
                    test_dots_launch_contract<u4x2_t, cuda_backend_t>, cuda, nk_dots_pack_size_u4_cuda,
                    nk_dots_packed_u4_cuda, nk_dots_symmetric_u4_cuda);
    run_kernel_case("nk_dots_symmetric_u4_cuda", test_dots_symmetric<u4x2_t, cuda_backend_t>, cuda,
                    nk_dots_symmetric_u4_cuda);
    run_kernel_case("nk_angulars_packed_f64_cuda", test_angulars_packed<f64_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_f64_cuda, nk_dots_pack_f64_cuda, nk_angulars_packed_f64_cuda);
    run_kernel_case("nk_angulars_symmetric_f64_cuda", test_angulars_symmetric<f64_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_f64_cuda);
    run_kernel_case("nk_euclideans_packed_f64_cuda", test_euclideans_packed<f64_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_f64_cuda, nk_dots_pack_f64_cuda, nk_euclideans_packed_f64_cuda);
    run_kernel_case("nk_euclideans_symmetric_f64_cuda", test_euclideans_symmetric<f64_t, cuda_backend_t>, cuda,
                    nk_euclideans_symmetric_f64_cuda);
    run_kernel_case("nk_angulars_packed_f32_cuda", test_angulars_packed<f32_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_f32_cuda, nk_dots_pack_f32_cuda, nk_angulars_packed_f32_cuda);
    run_kernel_case("nk_angulars_symmetric_f32_cuda", test_angulars_symmetric<f32_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_f32_cuda);
    run_kernel_case("nk_euclideans_packed_f32_cuda", test_euclideans_packed<f32_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_f32_cuda, nk_dots_pack_f32_cuda, nk_euclideans_packed_f32_cuda);
    run_kernel_case("nk_euclideans_symmetric_f32_cuda", test_euclideans_symmetric<f32_t, cuda_backend_t>, cuda,
                    nk_euclideans_symmetric_f32_cuda);
    run_kernel_case("nk_angulars_packed_bf16_cuda", test_angulars_packed<bf16_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_bf16_cuda, nk_dots_pack_bf16_cuda, nk_angulars_packed_bf16_cuda);
    run_kernel_case("nk_angulars_symmetric_bf16_cuda", test_angulars_symmetric<bf16_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_bf16_cuda);
    run_kernel_case("nk_euclideans_packed_bf16_cuda", test_euclideans_packed<bf16_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_bf16_cuda, nk_dots_pack_bf16_cuda, nk_euclideans_packed_bf16_cuda);
    run_kernel_case("nk_euclideans_symmetric_bf16_cuda", test_euclideans_symmetric<bf16_t, cuda_backend_t>, cuda,
                    nk_euclideans_symmetric_bf16_cuda);
    run_kernel_case("nk_angulars_packed_f16_cuda", test_angulars_packed<f16_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_f16_cuda, nk_dots_pack_f16_cuda, nk_angulars_packed_f16_cuda);
    run_kernel_case("nk_angulars_symmetric_f16_cuda", test_angulars_symmetric<f16_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_f16_cuda);
    run_kernel_case("nk_euclideans_packed_f16_cuda", test_euclideans_packed<f16_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_f16_cuda, nk_dots_pack_f16_cuda, nk_euclideans_packed_f16_cuda);
    run_kernel_case("nk_euclideans_symmetric_f16_cuda", test_euclideans_symmetric<f16_t, cuda_backend_t>, cuda,
                    nk_euclideans_symmetric_f16_cuda);
    run_kernel_case("nk_angulars_packed_e5m2_cuda", test_angulars_packed<e5m2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e5m2_cuda, nk_dots_pack_e5m2_cuda, nk_angulars_packed_e5m2_cuda);
    run_kernel_case("nk_angulars_symmetric_e5m2_cuda", test_angulars_symmetric<e5m2_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_e5m2_cuda);
    run_kernel_case("nk_euclideans_packed_e5m2_cuda", test_euclideans_packed<e5m2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e5m2_cuda, nk_dots_pack_e5m2_cuda, nk_euclideans_packed_e5m2_cuda);
    run_kernel_case("nk_euclideans_symmetric_e5m2_cuda", test_euclideans_symmetric<e5m2_t, cuda_backend_t>, cuda,
                    nk_euclideans_symmetric_e5m2_cuda);
    run_kernel_case("nk_angulars_packed_e4m3_cuda", test_angulars_packed<e4m3_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e4m3_cuda, nk_dots_pack_e4m3_cuda, nk_angulars_packed_e4m3_cuda);
    run_kernel_case("nk_angulars_symmetric_e4m3_cuda", test_angulars_symmetric<e4m3_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_e4m3_cuda);
    run_kernel_case("nk_euclideans_packed_e4m3_cuda", test_euclideans_packed<e4m3_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e4m3_cuda, nk_dots_pack_e4m3_cuda, nk_euclideans_packed_e4m3_cuda);
    run_kernel_case("nk_euclideans_symmetric_e4m3_cuda", test_euclideans_symmetric<e4m3_t, cuda_backend_t>, cuda,
                    nk_euclideans_symmetric_e4m3_cuda);
    run_kernel_case("nk_angulars_packed_e3m2_cuda", test_angulars_packed<e3m2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e3m2_cuda, nk_dots_pack_e3m2_cuda, nk_angulars_packed_e3m2_cuda);
    run_kernel_case("nk_angulars_symmetric_e3m2_cuda", test_angulars_symmetric<e3m2_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_e3m2_cuda);
    run_kernel_case("nk_euclideans_packed_e3m2_cuda", test_euclideans_packed<e3m2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e3m2_cuda, nk_dots_pack_e3m2_cuda, nk_euclideans_packed_e3m2_cuda);
    run_kernel_case("nk_euclideans_symmetric_e3m2_cuda", test_euclideans_symmetric<e3m2_t, cuda_backend_t>, cuda,
                    nk_euclideans_symmetric_e3m2_cuda);
    run_kernel_case("nk_angulars_packed_e2m3_cuda", test_angulars_packed<e2m3_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e2m3_cuda, nk_dots_pack_e2m3_cuda, nk_angulars_packed_e2m3_cuda);
    run_kernel_case("nk_angulars_symmetric_e2m3_cuda", test_angulars_symmetric<e2m3_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_e2m3_cuda);
    run_kernel_case("nk_euclideans_packed_e2m3_cuda", test_euclideans_packed<e2m3_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e2m3_cuda, nk_dots_pack_e2m3_cuda, nk_euclideans_packed_e2m3_cuda);
    run_kernel_case("nk_euclideans_symmetric_e2m3_cuda", test_euclideans_symmetric<e2m3_t, cuda_backend_t>, cuda,
                    nk_euclideans_symmetric_e2m3_cuda);
    run_kernel_case("nk_angulars_packed_e2m1_cuda", test_angulars_packed<e2m1x2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e2m1_cuda, nk_dots_pack_e2m1_cuda, nk_angulars_packed_e2m1_cuda);
    run_kernel_case("nk_angulars_symmetric_e2m1_cuda", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_e2m1_cuda);
    run_kernel_case("nk_euclideans_packed_e2m1_cuda", test_euclideans_packed<e2m1x2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_e2m1_cuda, nk_dots_pack_e2m1_cuda, nk_euclideans_packed_e2m1_cuda);
    run_kernel_case("nk_euclideans_symmetric_e2m1_cuda", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>, cuda,
                    nk_euclideans_symmetric_e2m1_cuda);
    run_kernel_case("nk_angulars_packed_nvfp4_cuda", test_angulars_packed<nvfp4_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_nvfp4_cuda, nk_dots_pack_nvfp4_cuda, nk_angulars_packed_nvfp4_cuda);
    run_kernel_case("nk_angulars_symmetric_nvfp4_cuda", test_angulars_symmetric<nvfp4_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_nvfp4_cuda);
    run_kernel_case("nk_euclideans_packed_nvfp4_cuda", test_euclideans_packed<nvfp4_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_nvfp4_cuda, nk_dots_pack_nvfp4_cuda, nk_euclideans_packed_nvfp4_cuda);
    run_kernel_case("nk_euclideans_symmetric_nvfp4_cuda", test_euclideans_symmetric<nvfp4_t, cuda_backend_t>, cuda,
                    nk_euclideans_symmetric_nvfp4_cuda);
    run_kernel_case("nk_angulars_packed_mxfp4_cuda", test_angulars_packed<mxfp4_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_mxfp4_cuda, nk_dots_pack_mxfp4_cuda, nk_angulars_packed_mxfp4_cuda);
    run_kernel_case("nk_angulars_symmetric_mxfp4_cuda", test_angulars_symmetric<mxfp4_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_mxfp4_cuda);
    run_kernel_case("nk_euclideans_packed_mxfp4_cuda", test_euclideans_packed<mxfp4_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_mxfp4_cuda, nk_dots_pack_mxfp4_cuda, nk_euclideans_packed_mxfp4_cuda);
    run_kernel_case("nk_euclideans_symmetric_mxfp4_cuda", test_euclideans_symmetric<mxfp4_t, cuda_backend_t>, cuda,
                    nk_euclideans_symmetric_mxfp4_cuda);
    run_kernel_case("nk_angulars_packed_mxfp8e4m3_cuda", test_angulars_packed<mxfp8e4m3_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_mxfp8e4m3_cuda, nk_dots_pack_mxfp8e4m3_cuda, nk_angulars_packed_mxfp8e4m3_cuda);
    run_kernel_case("nk_angulars_symmetric_mxfp8e4m3_cuda", test_angulars_symmetric<mxfp8e4m3_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_mxfp8e4m3_cuda);
    run_kernel_case("nk_euclideans_packed_mxfp8e4m3_cuda", test_euclideans_packed<mxfp8e4m3_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_mxfp8e4m3_cuda, nk_dots_pack_mxfp8e4m3_cuda, nk_euclideans_packed_mxfp8e4m3_cuda);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e4m3_cuda", test_euclideans_symmetric<mxfp8e4m3_t, cuda_backend_t>,
                    cuda, nk_euclideans_symmetric_mxfp8e4m3_cuda);
    run_kernel_case("nk_angulars_packed_mxfp8e5m2_cuda", test_angulars_packed<mxfp8e5m2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_mxfp8e5m2_cuda, nk_dots_pack_mxfp8e5m2_cuda, nk_angulars_packed_mxfp8e5m2_cuda);
    run_kernel_case("nk_angulars_symmetric_mxfp8e5m2_cuda", test_angulars_symmetric<mxfp8e5m2_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_mxfp8e5m2_cuda);
    run_kernel_case("nk_euclideans_packed_mxfp8e5m2_cuda", test_euclideans_packed<mxfp8e5m2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_mxfp8e5m2_cuda, nk_dots_pack_mxfp8e5m2_cuda, nk_euclideans_packed_mxfp8e5m2_cuda);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e5m2_cuda", test_euclideans_symmetric<mxfp8e5m2_t, cuda_backend_t>,
                    cuda, nk_euclideans_symmetric_mxfp8e5m2_cuda);
    run_kernel_case("nk_angulars_packed_i8_cuda", test_angulars_packed<i8_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_i8_cuda, nk_dots_pack_i8_cuda, nk_angulars_packed_i8_cuda);
    run_kernel_case("nk_angulars_symmetric_i8_cuda", test_angulars_symmetric<i8_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_i8_cuda);
    run_kernel_case("nk_euclideans_packed_i8_cuda", test_euclideans_packed<i8_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_i8_cuda, nk_dots_pack_i8_cuda, nk_euclideans_packed_i8_cuda);
    run_kernel_case("nk_euclideans_symmetric_i8_cuda", test_euclideans_symmetric<i8_t, cuda_backend_t>, cuda,
                    nk_euclideans_symmetric_i8_cuda);
    run_kernel_case("nk_angulars_packed_u8_cuda", test_angulars_packed<u8_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_u8_cuda, nk_dots_pack_u8_cuda, nk_angulars_packed_u8_cuda);
    run_kernel_case("nk_angulars_symmetric_u8_cuda", test_angulars_symmetric<u8_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_u8_cuda);
    run_kernel_case("nk_euclideans_packed_u8_cuda", test_euclideans_packed<u8_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_u8_cuda, nk_dots_pack_u8_cuda, nk_euclideans_packed_u8_cuda);
    run_kernel_case("nk_euclideans_symmetric_u8_cuda", test_euclideans_symmetric<u8_t, cuda_backend_t>, cuda,
                    nk_euclideans_symmetric_u8_cuda);
    run_kernel_case("nk_angulars_packed_i4_cuda", test_angulars_packed<i4x2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_i4_cuda, nk_dots_pack_i4_cuda, nk_angulars_packed_i4_cuda);
    run_kernel_case("nk_angulars_symmetric_i4_cuda", test_angulars_symmetric<i4x2_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_i4_cuda);
    run_kernel_case("nk_euclideans_packed_i4_cuda", test_euclideans_packed<i4x2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_i4_cuda, nk_dots_pack_i4_cuda, nk_euclideans_packed_i4_cuda);
    run_kernel_case("nk_euclideans_symmetric_i4_cuda", test_euclideans_symmetric<i4x2_t, cuda_backend_t>, cuda,
                    nk_euclideans_symmetric_i4_cuda);
    run_kernel_case("nk_angulars_packed_u4_cuda", test_angulars_packed<u4x2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_u4_cuda, nk_dots_pack_u4_cuda, nk_angulars_packed_u4_cuda);
    run_kernel_case("nk_angulars_symmetric_u4_cuda", test_angulars_symmetric<u4x2_t, cuda_backend_t>, cuda,
                    nk_angulars_symmetric_u4_cuda);
    run_kernel_case("nk_euclideans_packed_u4_cuda", test_euclideans_packed<u4x2_t, cuda_backend_t>, cuda,
                    nk_dots_pack_size_u4_cuda, nk_dots_pack_u4_cuda, nk_euclideans_packed_u4_cuda);
    run_kernel_case("nk_euclideans_symmetric_u4_cuda", test_euclideans_symmetric<u4x2_t, cuda_backend_t>, cuda,
                    nk_euclideans_symmetric_u4_cuda);
    run_kernel_case("nk_attention_packed_bf16_cuda",
                    test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>, cuda,
                    nk_attention_pack_size_bf16_cuda, nk_attention_pack_bf16_cuda, nk_attention_packed_bf16_cuda);
    run_kernel_case("nk_attention_packed_f16_cuda",
                    test_attention_packed<f16_t, cuda_backend_t, attention_weights_t::bits_11_k>, cuda,
                    nk_attention_pack_size_f16_cuda, nk_attention_pack_f16_cuda, nk_attention_packed_f16_cuda);
    run_kernel_case("nk_attention_packed_e4m3_cuda",
                    test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_11_k>, cuda,
                    nk_attention_pack_size_e4m3_cuda, nk_attention_pack_e4m3_cuda, nk_attention_packed_e4m3_cuda);
    run_kernel_case("nk_attention_packed_i8_cuda",
                    test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>, cuda,
                    nk_attention_pack_size_i8_cuda, nk_attention_pack_i8_cuda, nk_attention_packed_i8_cuda);
    run_kernel_case("nk_attention_packed_gradients_bf16_cuda", test_attention_packed_gradients<bf16_t, cuda_backend_t>,
                    cuda, nk_attention_pack_size_bf16_cuda, nk_attention_pack_bf16_cuda,
                    nk_attention_packed_gradients_bf16_cuda);
    run_kernel_case("nk_attention_rope_f32_cuda", test_attention_rope<f32_t, cuda_backend_t>, cuda,
                    nk_attention_rope_f32_cuda);
    run_kernel_case("nk_attention_rope_bf16_cuda", test_attention_rope<bf16_t, cuda_backend_t>, cuda,
                    nk_attention_rope_bf16_cuda);
    run_kernel_case("nk_attention_rope_e4m3_cuda", test_attention_rope<e4m3_t, cuda_backend_t>, cuda,
                    nk_attention_rope_e4m3_cuda);
    run_kernel_case("nk_wassersteinprocrustes_packed_f32_cuda",
                    test_transport_packed<f32_t, nk_kernel_wassersteinprocrustes_packed_k, cuda_backend_t>, cuda,
                    nk_wassersteinprocrustes_workspace_size_f32_cuda, nk_wassersteinprocrustes_packed_f32_cuda);
    run_kernel_case("nk_wassersteinprocrustes_packed_f16_cuda",
                    test_transport_packed<f16_t, nk_kernel_wassersteinprocrustes_packed_k, cuda_backend_t>, cuda,
                    nk_wassersteinprocrustes_workspace_size_f16_cuda, nk_wassersteinprocrustes_packed_f16_cuda);
    run_kernel_case("nk_gromovwassersteins_packed_f32_cuda",
                    test_transport_packed<f32_t, nk_kernel_gromovwassersteins_packed_k, cuda_backend_t>, cuda,
                    nk_gromovwassersteins_workspace_size_f32_cuda, nk_gromovwassersteins_packed_f32_cuda);
    run_kernel_case("nk_gromovwassersteins_packed_f16_cuda",
                    test_transport_packed<f16_t, nk_kernel_gromovwassersteins_packed_k, cuda_backend_t>, cuda,
                    nk_gromovwassersteins_workspace_size_f16_cuda, nk_gromovwassersteins_packed_f16_cuda);
    run_kernel_case("nk_haversines_pack_f64_cuda",
                    test_geodesics_pack<f64_t, nk_kernel_haversines_pack_k, cuda_backend_t>, cuda,
                    nk_haversines_pack_size_f64_cuda, nk_haversines_pack_f64_cuda);
    run_kernel_case("nk_vincentys_pack_f64_cuda",
                    test_geodesics_pack<f64_t, nk_kernel_vincentys_pack_k, cuda_backend_t>, cuda,
                    nk_vincentys_pack_size_f64_cuda, nk_vincentys_pack_f64_cuda);
    run_kernel_case("nk_haversines_packed_f64_cuda",
                    test_geodesics_packed<f64_t, nk_kernel_haversines_packed_k, cuda_backend_t>, cuda,
                    nk_haversines_packed_f64_cuda);
    run_kernel_case("nk_vincentys_packed_f64_cuda",
                    test_geodesics_packed<f64_t, nk_kernel_vincentys_packed_k, cuda_backend_t>, cuda,
                    nk_vincentys_packed_f64_cuda);
    run_kernel_case("nk_clearances_packed_f32_cuda", test_terrain_packed<f32_t, cuda_backend_t>, cuda,
                    nk_terrain_pack_size_f32_cuda, nk_terrain_pack_f32_cuda, nk_terrain_heights_packed_f32_cuda,
                    nk_clearances_packed_f32_cuda);

#if NUMKONG_TARGET_AMPERE
    auto const ampere = nk_cap_ampere_k;
    run_kernel_case("nk_dots_packed_f64_ampere", test_dots_packed<f64_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_f64_ampere, nk_dots_pack_f64_ampere, nk_dots_packed_f64_ampere);
    run_kernel_case("nk_dots_pack_f64_ampere", test_dots_pack_layout<f64_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_f64_ampere, nk_dots_packed_shape_f64_ampere, nk_dots_pack_f64_ampere);
    run_kernel_case("test_dots_launch_contract<f64_t, cuda_backend_t>(nk_dots_packed_f64_ampere)",
                    test_dots_launch_contract<f64_t, cuda_backend_t>, ampere, nk_dots_pack_size_f64_ampere,
                    nk_dots_packed_f64_ampere, nk_dots_symmetric_f64_ampere);
    run_kernel_case("nk_dots_symmetric_f64_ampere", test_dots_symmetric<f64_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_f64_ampere);
    run_kernel_case("test_dots_packed<f64_t, cuda_backend_t>(with_workspace(nk_dots_packed_f64_ampere))",
                    test_dots_packed<f64_t, cuda_backend_t>, ampere, nk_dots_pack_size_f64_ampere,
                    nk_dots_pack_f64_ampere,
                    with_workspace(nk_dots_packed_f64_ampere, nk_dots_packed_workspace_size_f64_ampere));
    run_kernel_case("test_dots_symmetric<f64_t, cuda_backend_t>(with_workspace(nk_dots_symmetric_f64_ampere))",
                    test_dots_symmetric<f64_t, cuda_backend_t>, ampere,
                    with_workspace(nk_dots_symmetric_f64_ampere, nk_dots_symmetric_workspace_size_f64_ampere));
    run_kernel_case("nk_dots_packed_f32_ampere", test_dots_packed<f32_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_f32_ampere, nk_dots_pack_f32_ampere, nk_dots_packed_f32_ampere);
    run_kernel_case("nk_dots_pack_f32_ampere", test_dots_pack_layout<f32_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_f32_ampere, nk_dots_packed_shape_f32_ampere, nk_dots_pack_f32_ampere);
    run_kernel_case("test_dots_launch_contract<f32_t, cuda_backend_t>(nk_dots_packed_f32_ampere)",
                    test_dots_launch_contract<f32_t, cuda_backend_t>, ampere, nk_dots_pack_size_f32_ampere,
                    nk_dots_packed_f32_ampere, nk_dots_symmetric_f32_ampere);
    run_kernel_case("nk_dots_symmetric_f32_ampere", test_dots_symmetric<f32_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_f32_ampere);
    run_kernel_case("nk_dots_packed_bf16_ampere", test_dots_packed<bf16_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_bf16_ampere, nk_dots_pack_bf16_ampere, nk_dots_packed_bf16_ampere);
    run_kernel_case("nk_dots_pack_bf16_ampere", test_dots_pack_layout<bf16_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_bf16_ampere, nk_dots_packed_shape_bf16_ampere, nk_dots_pack_bf16_ampere);
    run_kernel_case("test_dots_launch_contract<bf16_t, cuda_backend_t>(nk_dots_packed_bf16_ampere)",
                    test_dots_launch_contract<bf16_t, cuda_backend_t>, ampere, nk_dots_pack_size_bf16_ampere,
                    nk_dots_packed_bf16_ampere, nk_dots_symmetric_bf16_ampere);
    run_kernel_case("nk_dots_symmetric_bf16_ampere", test_dots_symmetric<bf16_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_bf16_ampere);
    run_kernel_case("nk_dots_packed_f16_ampere", test_dots_packed<f16_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_f16_ampere, nk_dots_pack_f16_ampere, nk_dots_packed_f16_ampere);
    run_kernel_case("nk_dots_pack_f16_ampere", test_dots_pack_layout<f16_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_f16_ampere, nk_dots_packed_shape_f16_ampere, nk_dots_pack_f16_ampere);
    run_kernel_case("test_dots_launch_contract<f16_t, cuda_backend_t>(nk_dots_packed_f16_ampere)",
                    test_dots_launch_contract<f16_t, cuda_backend_t>, ampere, nk_dots_pack_size_f16_ampere,
                    nk_dots_packed_f16_ampere, nk_dots_symmetric_f16_ampere);
    run_kernel_case("nk_dots_symmetric_f16_ampere", test_dots_symmetric<f16_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_f16_ampere);
    run_kernel_case("nk_dots_packed_e5m2_ampere", test_dots_packed<e5m2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e5m2_ampere, nk_dots_pack_e5m2_ampere, nk_dots_packed_e5m2_ampere);
    run_kernel_case("nk_dots_pack_e5m2_ampere", test_dots_pack_layout<e5m2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e5m2_ampere, nk_dots_packed_shape_e5m2_ampere, nk_dots_pack_e5m2_ampere);
    run_kernel_case("test_dots_launch_contract<e5m2_t, cuda_backend_t>(nk_dots_packed_e5m2_ampere)",
                    test_dots_launch_contract<e5m2_t, cuda_backend_t>, ampere, nk_dots_pack_size_e5m2_ampere,
                    nk_dots_packed_e5m2_ampere, nk_dots_symmetric_e5m2_ampere);
    run_kernel_case("nk_dots_symmetric_e5m2_ampere", test_dots_symmetric<e5m2_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_e5m2_ampere);
    run_kernel_case("nk_dots_packed_e4m3_ampere", test_dots_packed<e4m3_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e4m3_ampere, nk_dots_pack_e4m3_ampere, nk_dots_packed_e4m3_ampere);
    run_kernel_case("nk_dots_pack_e4m3_ampere", test_dots_pack_layout<e4m3_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e4m3_ampere, nk_dots_packed_shape_e4m3_ampere, nk_dots_pack_e4m3_ampere);
    run_kernel_case("test_dots_launch_contract<e4m3_t, cuda_backend_t>(nk_dots_packed_e4m3_ampere)",
                    test_dots_launch_contract<e4m3_t, cuda_backend_t>, ampere, nk_dots_pack_size_e4m3_ampere,
                    nk_dots_packed_e4m3_ampere, nk_dots_symmetric_e4m3_ampere);
    run_kernel_case("nk_dots_symmetric_e4m3_ampere", test_dots_symmetric<e4m3_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_e4m3_ampere);
    run_kernel_case("nk_dots_packed_e3m2_ampere", test_dots_packed<e3m2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e3m2_ampere, nk_dots_pack_e3m2_ampere, nk_dots_packed_e3m2_ampere);
    run_kernel_case("nk_dots_pack_e3m2_ampere", test_dots_pack_layout<e3m2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e3m2_ampere, nk_dots_packed_shape_e3m2_ampere, nk_dots_pack_e3m2_ampere);
    run_kernel_case("test_dots_launch_contract<e3m2_t, cuda_backend_t>(nk_dots_packed_e3m2_ampere)",
                    test_dots_launch_contract<e3m2_t, cuda_backend_t>, ampere, nk_dots_pack_size_e3m2_ampere,
                    nk_dots_packed_e3m2_ampere, nk_dots_symmetric_e3m2_ampere);
    run_kernel_case("nk_dots_symmetric_e3m2_ampere", test_dots_symmetric<e3m2_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_e3m2_ampere);
    run_kernel_case("nk_dots_packed_e2m3_ampere", test_dots_packed<e2m3_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e2m3_ampere, nk_dots_pack_e2m3_ampere, nk_dots_packed_e2m3_ampere);
    run_kernel_case("nk_dots_pack_e2m3_ampere", test_dots_pack_layout<e2m3_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e2m3_ampere, nk_dots_packed_shape_e2m3_ampere, nk_dots_pack_e2m3_ampere);
    run_kernel_case("test_dots_launch_contract<e2m3_t, cuda_backend_t>(nk_dots_packed_e2m3_ampere)",
                    test_dots_launch_contract<e2m3_t, cuda_backend_t>, ampere, nk_dots_pack_size_e2m3_ampere,
                    nk_dots_packed_e2m3_ampere, nk_dots_symmetric_e2m3_ampere);
    run_kernel_case("nk_dots_symmetric_e2m3_ampere", test_dots_symmetric<e2m3_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_e2m3_ampere);
    run_kernel_case("nk_dots_packed_e2m1_ampere", test_dots_packed<e2m1x2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e2m1_ampere, nk_dots_pack_e2m1_ampere, nk_dots_packed_e2m1_ampere);
    run_kernel_case("nk_dots_pack_e2m1_ampere", test_dots_pack_layout<e2m1x2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e2m1_ampere, nk_dots_packed_shape_e2m1_ampere, nk_dots_pack_e2m1_ampere);
    run_kernel_case("test_dots_launch_contract<e2m1x2_t, cuda_backend_t>(nk_dots_packed_e2m1_ampere)",
                    test_dots_launch_contract<e2m1x2_t, cuda_backend_t>, ampere, nk_dots_pack_size_e2m1_ampere,
                    nk_dots_packed_e2m1_ampere, nk_dots_symmetric_e2m1_ampere);
    run_kernel_case("nk_dots_symmetric_e2m1_ampere", test_dots_symmetric<e2m1x2_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_e2m1_ampere);
    run_kernel_case("nk_dots_packed_nvfp4_ampere", test_dots_packed<nvfp4_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_nvfp4_ampere, nk_dots_pack_nvfp4_ampere, nk_dots_packed_nvfp4_ampere);
    run_kernel_case("nk_dots_pack_nvfp4_ampere", test_dots_pack_layout<nvfp4_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_nvfp4_ampere, nk_dots_packed_shape_nvfp4_ampere, nk_dots_pack_nvfp4_ampere);
    run_kernel_case("test_dots_launch_contract<nvfp4_t, cuda_backend_t>(nk_dots_packed_nvfp4_ampere)",
                    test_dots_launch_contract<nvfp4_t, cuda_backend_t>, ampere, nk_dots_pack_size_nvfp4_ampere,
                    nk_dots_packed_nvfp4_ampere, nk_dots_symmetric_nvfp4_ampere);
    run_kernel_case("nk_dots_symmetric_nvfp4_ampere", test_dots_symmetric<nvfp4_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_nvfp4_ampere);
    run_kernel_case("nk_dots_packed_mxfp4_ampere", test_dots_packed<mxfp4_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_mxfp4_ampere, nk_dots_pack_mxfp4_ampere, nk_dots_packed_mxfp4_ampere);
    run_kernel_case("nk_dots_pack_mxfp4_ampere", test_dots_pack_layout<mxfp4_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_mxfp4_ampere, nk_dots_packed_shape_mxfp4_ampere, nk_dots_pack_mxfp4_ampere);
    run_kernel_case("test_dots_launch_contract<mxfp4_t, cuda_backend_t>(nk_dots_packed_mxfp4_ampere)",
                    test_dots_launch_contract<mxfp4_t, cuda_backend_t>, ampere, nk_dots_pack_size_mxfp4_ampere,
                    nk_dots_packed_mxfp4_ampere, nk_dots_symmetric_mxfp4_ampere);
    run_kernel_case("nk_dots_symmetric_mxfp4_ampere", test_dots_symmetric<mxfp4_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_mxfp4_ampere);
    run_kernel_case("nk_dots_packed_mxfp8e4m3_ampere", test_dots_packed<mxfp8e4m3_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_mxfp8e4m3_ampere, nk_dots_pack_mxfp8e4m3_ampere, nk_dots_packed_mxfp8e4m3_ampere);
    run_kernel_case("nk_dots_pack_mxfp8e4m3_ampere", test_dots_pack_layout<mxfp8e4m3_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_mxfp8e4m3_ampere, nk_dots_packed_shape_mxfp8e4m3_ampere,
                    nk_dots_pack_mxfp8e4m3_ampere);
    run_kernel_case("test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t>(nk_dots_packed_mxfp8e4m3_ampere)",
                    test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t>, ampere, nk_dots_pack_size_mxfp8e4m3_ampere,
                    nk_dots_packed_mxfp8e4m3_ampere, nk_dots_symmetric_mxfp8e4m3_ampere);
    run_kernel_case("nk_dots_symmetric_mxfp8e4m3_ampere", test_dots_symmetric<mxfp8e4m3_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_mxfp8e4m3_ampere);
    run_kernel_case("nk_dots_packed_mxfp8e5m2_ampere", test_dots_packed<mxfp8e5m2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_mxfp8e5m2_ampere, nk_dots_pack_mxfp8e5m2_ampere, nk_dots_packed_mxfp8e5m2_ampere);
    run_kernel_case("nk_dots_pack_mxfp8e5m2_ampere", test_dots_pack_layout<mxfp8e5m2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_mxfp8e5m2_ampere, nk_dots_packed_shape_mxfp8e5m2_ampere,
                    nk_dots_pack_mxfp8e5m2_ampere);
    run_kernel_case("test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t>(nk_dots_packed_mxfp8e5m2_ampere)",
                    test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t>, ampere, nk_dots_pack_size_mxfp8e5m2_ampere,
                    nk_dots_packed_mxfp8e5m2_ampere, nk_dots_symmetric_mxfp8e5m2_ampere);
    run_kernel_case("nk_dots_symmetric_mxfp8e5m2_ampere", test_dots_symmetric<mxfp8e5m2_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_mxfp8e5m2_ampere);
    run_kernel_case("nk_dots_packed_i8_ampere", test_dots_packed<i8_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_i8_ampere, nk_dots_pack_i8_ampere, nk_dots_packed_i8_ampere);
    run_kernel_case("nk_dots_pack_i8_ampere", test_dots_pack_layout<i8_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_i8_ampere, nk_dots_packed_shape_i8_ampere, nk_dots_pack_i8_ampere);
    run_kernel_case("test_dots_launch_contract<i8_t, cuda_backend_t>(nk_dots_packed_i8_ampere)",
                    test_dots_launch_contract<i8_t, cuda_backend_t>, ampere, nk_dots_pack_size_i8_ampere,
                    nk_dots_packed_i8_ampere, nk_dots_symmetric_i8_ampere);
    run_kernel_case("nk_dots_symmetric_i8_ampere", test_dots_symmetric<i8_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_i8_ampere);
    run_kernel_case("nk_dots_packed_u8_ampere", test_dots_packed<u8_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_u8_ampere, nk_dots_pack_u8_ampere, nk_dots_packed_u8_ampere);
    run_kernel_case("nk_dots_pack_u8_ampere", test_dots_pack_layout<u8_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_u8_ampere, nk_dots_packed_shape_u8_ampere, nk_dots_pack_u8_ampere);
    run_kernel_case("test_dots_launch_contract<u8_t, cuda_backend_t>(nk_dots_packed_u8_ampere)",
                    test_dots_launch_contract<u8_t, cuda_backend_t>, ampere, nk_dots_pack_size_u8_ampere,
                    nk_dots_packed_u8_ampere, nk_dots_symmetric_u8_ampere);
    run_kernel_case("nk_dots_symmetric_u8_ampere", test_dots_symmetric<u8_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_u8_ampere);
    run_kernel_case("nk_dots_packed_i4_ampere", test_dots_packed<i4x2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_i4_ampere, nk_dots_pack_i4_ampere, nk_dots_packed_i4_ampere);
    run_kernel_case("nk_dots_pack_i4_ampere", test_dots_pack_layout<i4x2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_i4_ampere, nk_dots_packed_shape_i4_ampere, nk_dots_pack_i4_ampere);
    run_kernel_case("test_dots_launch_contract<i4x2_t, cuda_backend_t>(nk_dots_packed_i4_ampere)",
                    test_dots_launch_contract<i4x2_t, cuda_backend_t>, ampere, nk_dots_pack_size_i4_ampere,
                    nk_dots_packed_i4_ampere, nk_dots_symmetric_i4_ampere);
    run_kernel_case("nk_dots_symmetric_i4_ampere", test_dots_symmetric<i4x2_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_i4_ampere);
    run_kernel_case("nk_dots_packed_u4_ampere", test_dots_packed<u4x2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_u4_ampere, nk_dots_pack_u4_ampere, nk_dots_packed_u4_ampere);
    run_kernel_case("nk_dots_pack_u4_ampere", test_dots_pack_layout<u4x2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_u4_ampere, nk_dots_packed_shape_u4_ampere, nk_dots_pack_u4_ampere);
    run_kernel_case("test_dots_launch_contract<u4x2_t, cuda_backend_t>(nk_dots_packed_u4_ampere)",
                    test_dots_launch_contract<u4x2_t, cuda_backend_t>, ampere, nk_dots_pack_size_u4_ampere,
                    nk_dots_packed_u4_ampere, nk_dots_symmetric_u4_ampere);
    run_kernel_case("nk_dots_symmetric_u4_ampere", test_dots_symmetric<u4x2_t, cuda_backend_t>, ampere,
                    nk_dots_symmetric_u4_ampere);
    run_kernel_case("nk_angulars_packed_f64_ampere", test_angulars_packed<f64_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_f64_ampere, nk_dots_pack_f64_ampere, nk_angulars_packed_f64_ampere);
    run_kernel_case("nk_angulars_symmetric_f64_ampere", test_angulars_symmetric<f64_t, cuda_backend_t>, ampere,
                    nk_angulars_symmetric_f64_ampere);
    run_kernel_case("nk_euclideans_packed_f64_ampere", test_euclideans_packed<f64_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_f64_ampere, nk_dots_pack_f64_ampere, nk_euclideans_packed_f64_ampere);
    run_kernel_case("nk_euclideans_symmetric_f64_ampere", test_euclideans_symmetric<f64_t, cuda_backend_t>, ampere,
                    nk_euclideans_symmetric_f64_ampere);
    run_kernel_case("test_angulars_packed<f64_t, cuda_backend_t>(with_workspace(nk_angulars_packed_f64_ampere))",
                    test_angulars_packed<f64_t, cuda_backend_t>, ampere, nk_dots_pack_size_f64_ampere,
                    nk_dots_pack_f64_ampere,
                    with_workspace(nk_angulars_packed_f64_ampere, nk_angulars_packed_workspace_size_f64_ampere));
    run_kernel_case("test_angulars_symmetric<f64_t, cuda_backend_t>(with_workspace(nk_angulars_symmetric_f64_ampere))",
                    test_angulars_symmetric<f64_t, cuda_backend_t>, ampere,
                    with_workspace(nk_angulars_symmetric_f64_ampere, nk_angulars_symmetric_workspace_size_f64_ampere));
    run_kernel_case("test_euclideans_packed<f64_t, cuda_backend_t>(with_workspace(nk_euclideans_packed_f64_ampere))",
                    test_euclideans_packed<f64_t, cuda_backend_t>, ampere, nk_dots_pack_size_f64_ampere,
                    nk_dots_pack_f64_ampere,
                    with_workspace(nk_euclideans_packed_f64_ampere, nk_euclideans_packed_workspace_size_f64_ampere));
    run_kernel_case(
        "test_euclideans_symmetric<f64_t, cuda_backend_t>(with_workspace(nk_euclideans_symmetric_f64_ampere))",
        test_euclideans_symmetric<f64_t, cuda_backend_t>, ampere,
        with_workspace(nk_euclideans_symmetric_f64_ampere, nk_euclideans_symmetric_workspace_size_f64_ampere));
    run_kernel_case("nk_angulars_packed_f32_ampere", test_angulars_packed<f32_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_f32_ampere, nk_dots_pack_f32_ampere, nk_angulars_packed_f32_ampere);
    run_kernel_case("nk_angulars_symmetric_f32_ampere", test_angulars_symmetric<f32_t, cuda_backend_t>, ampere,
                    nk_angulars_symmetric_f32_ampere);
    run_kernel_case("nk_euclideans_packed_f32_ampere", test_euclideans_packed<f32_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_f32_ampere, nk_dots_pack_f32_ampere, nk_euclideans_packed_f32_ampere);
    run_kernel_case("nk_euclideans_symmetric_f32_ampere", test_euclideans_symmetric<f32_t, cuda_backend_t>, ampere,
                    nk_euclideans_symmetric_f32_ampere);
    run_kernel_case("nk_angulars_packed_bf16_ampere", test_angulars_packed<bf16_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_bf16_ampere, nk_dots_pack_bf16_ampere, nk_angulars_packed_bf16_ampere);
    run_kernel_case("nk_angulars_symmetric_bf16_ampere", test_angulars_symmetric<bf16_t, cuda_backend_t>, ampere,
                    nk_angulars_symmetric_bf16_ampere);
    run_kernel_case("nk_euclideans_packed_bf16_ampere", test_euclideans_packed<bf16_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_bf16_ampere, nk_dots_pack_bf16_ampere, nk_euclideans_packed_bf16_ampere);
    run_kernel_case("nk_euclideans_symmetric_bf16_ampere", test_euclideans_symmetric<bf16_t, cuda_backend_t>, ampere,
                    nk_euclideans_symmetric_bf16_ampere);
    run_kernel_case("nk_angulars_packed_f16_ampere", test_angulars_packed<f16_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_f16_ampere, nk_dots_pack_f16_ampere, nk_angulars_packed_f16_ampere);
    run_kernel_case("nk_angulars_symmetric_f16_ampere", test_angulars_symmetric<f16_t, cuda_backend_t>, ampere,
                    nk_angulars_symmetric_f16_ampere);
    run_kernel_case("nk_euclideans_packed_f16_ampere", test_euclideans_packed<f16_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_f16_ampere, nk_dots_pack_f16_ampere, nk_euclideans_packed_f16_ampere);
    run_kernel_case("nk_euclideans_symmetric_f16_ampere", test_euclideans_symmetric<f16_t, cuda_backend_t>, ampere,
                    nk_euclideans_symmetric_f16_ampere);
    run_kernel_case("nk_angulars_packed_e5m2_ampere", test_angulars_packed<e5m2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e5m2_ampere, nk_dots_pack_e5m2_ampere, nk_angulars_packed_e5m2_ampere);
    run_kernel_case("nk_angulars_symmetric_e5m2_ampere", test_angulars_symmetric<e5m2_t, cuda_backend_t>, ampere,
                    nk_angulars_symmetric_e5m2_ampere);
    run_kernel_case("nk_euclideans_packed_e5m2_ampere", test_euclideans_packed<e5m2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e5m2_ampere, nk_dots_pack_e5m2_ampere, nk_euclideans_packed_e5m2_ampere);
    run_kernel_case("nk_euclideans_symmetric_e5m2_ampere", test_euclideans_symmetric<e5m2_t, cuda_backend_t>, ampere,
                    nk_euclideans_symmetric_e5m2_ampere);
    run_kernel_case("nk_angulars_packed_e4m3_ampere", test_angulars_packed<e4m3_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e4m3_ampere, nk_dots_pack_e4m3_ampere, nk_angulars_packed_e4m3_ampere);
    run_kernel_case("nk_angulars_symmetric_e4m3_ampere", test_angulars_symmetric<e4m3_t, cuda_backend_t>, ampere,
                    nk_angulars_symmetric_e4m3_ampere);
    run_kernel_case("nk_euclideans_packed_e4m3_ampere", test_euclideans_packed<e4m3_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e4m3_ampere, nk_dots_pack_e4m3_ampere, nk_euclideans_packed_e4m3_ampere);
    run_kernel_case("nk_euclideans_symmetric_e4m3_ampere", test_euclideans_symmetric<e4m3_t, cuda_backend_t>, ampere,
                    nk_euclideans_symmetric_e4m3_ampere);
    run_kernel_case("nk_angulars_packed_e3m2_ampere", test_angulars_packed<e3m2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e3m2_ampere, nk_dots_pack_e3m2_ampere, nk_angulars_packed_e3m2_ampere);
    run_kernel_case("nk_angulars_symmetric_e3m2_ampere", test_angulars_symmetric<e3m2_t, cuda_backend_t>, ampere,
                    nk_angulars_symmetric_e3m2_ampere);
    run_kernel_case("nk_euclideans_packed_e3m2_ampere", test_euclideans_packed<e3m2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e3m2_ampere, nk_dots_pack_e3m2_ampere, nk_euclideans_packed_e3m2_ampere);
    run_kernel_case("nk_euclideans_symmetric_e3m2_ampere", test_euclideans_symmetric<e3m2_t, cuda_backend_t>, ampere,
                    nk_euclideans_symmetric_e3m2_ampere);
    run_kernel_case("nk_angulars_packed_e2m3_ampere", test_angulars_packed<e2m3_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e2m3_ampere, nk_dots_pack_e2m3_ampere, nk_angulars_packed_e2m3_ampere);
    run_kernel_case("nk_angulars_symmetric_e2m3_ampere", test_angulars_symmetric<e2m3_t, cuda_backend_t>, ampere,
                    nk_angulars_symmetric_e2m3_ampere);
    run_kernel_case("nk_euclideans_packed_e2m3_ampere", test_euclideans_packed<e2m3_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e2m3_ampere, nk_dots_pack_e2m3_ampere, nk_euclideans_packed_e2m3_ampere);
    run_kernel_case("nk_euclideans_symmetric_e2m3_ampere", test_euclideans_symmetric<e2m3_t, cuda_backend_t>, ampere,
                    nk_euclideans_symmetric_e2m3_ampere);
    run_kernel_case("nk_angulars_packed_e2m1_ampere", test_angulars_packed<e2m1x2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e2m1_ampere, nk_dots_pack_e2m1_ampere, nk_angulars_packed_e2m1_ampere);
    run_kernel_case("nk_angulars_symmetric_e2m1_ampere", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>, ampere,
                    nk_angulars_symmetric_e2m1_ampere);
    run_kernel_case("nk_euclideans_packed_e2m1_ampere", test_euclideans_packed<e2m1x2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_e2m1_ampere, nk_dots_pack_e2m1_ampere, nk_euclideans_packed_e2m1_ampere);
    run_kernel_case("nk_euclideans_symmetric_e2m1_ampere", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>, ampere,
                    nk_euclideans_symmetric_e2m1_ampere);
    run_kernel_case("nk_angulars_packed_i8_ampere", test_angulars_packed<i8_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_i8_ampere, nk_dots_pack_i8_ampere, nk_angulars_packed_i8_ampere);
    run_kernel_case("nk_angulars_symmetric_i8_ampere", test_angulars_symmetric<i8_t, cuda_backend_t>, ampere,
                    nk_angulars_symmetric_i8_ampere);
    run_kernel_case("nk_euclideans_packed_i8_ampere", test_euclideans_packed<i8_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_i8_ampere, nk_dots_pack_i8_ampere, nk_euclideans_packed_i8_ampere);
    run_kernel_case("nk_euclideans_symmetric_i8_ampere", test_euclideans_symmetric<i8_t, cuda_backend_t>, ampere,
                    nk_euclideans_symmetric_i8_ampere);
    run_kernel_case("nk_angulars_packed_u8_ampere", test_angulars_packed<u8_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_u8_ampere, nk_dots_pack_u8_ampere, nk_angulars_packed_u8_ampere);
    run_kernel_case("nk_angulars_symmetric_u8_ampere", test_angulars_symmetric<u8_t, cuda_backend_t>, ampere,
                    nk_angulars_symmetric_u8_ampere);
    run_kernel_case("nk_euclideans_packed_u8_ampere", test_euclideans_packed<u8_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_u8_ampere, nk_dots_pack_u8_ampere, nk_euclideans_packed_u8_ampere);
    run_kernel_case("nk_euclideans_symmetric_u8_ampere", test_euclideans_symmetric<u8_t, cuda_backend_t>, ampere,
                    nk_euclideans_symmetric_u8_ampere);
    run_kernel_case("nk_angulars_packed_i4_ampere", test_angulars_packed<i4x2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_i4_ampere, nk_dots_pack_i4_ampere, nk_angulars_packed_i4_ampere);
    run_kernel_case("nk_angulars_symmetric_i4_ampere", test_angulars_symmetric<i4x2_t, cuda_backend_t>, ampere,
                    nk_angulars_symmetric_i4_ampere);
    run_kernel_case("nk_euclideans_packed_i4_ampere", test_euclideans_packed<i4x2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_i4_ampere, nk_dots_pack_i4_ampere, nk_euclideans_packed_i4_ampere);
    run_kernel_case("nk_euclideans_symmetric_i4_ampere", test_euclideans_symmetric<i4x2_t, cuda_backend_t>, ampere,
                    nk_euclideans_symmetric_i4_ampere);
    run_kernel_case("nk_angulars_packed_u4_ampere", test_angulars_packed<u4x2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_u4_ampere, nk_dots_pack_u4_ampere, nk_angulars_packed_u4_ampere);
    run_kernel_case("nk_angulars_symmetric_u4_ampere", test_angulars_symmetric<u4x2_t, cuda_backend_t>, ampere,
                    nk_angulars_symmetric_u4_ampere);
    run_kernel_case("nk_euclideans_packed_u4_ampere", test_euclideans_packed<u4x2_t, cuda_backend_t>, ampere,
                    nk_dots_pack_size_u4_ampere, nk_dots_pack_u4_ampere, nk_euclideans_packed_u4_ampere);
    run_kernel_case("nk_euclideans_symmetric_u4_ampere", test_euclideans_symmetric<u4x2_t, cuda_backend_t>, ampere,
                    nk_euclideans_symmetric_u4_ampere);
    run_kernel_case("nk_attention_packed_bf16_ampere",
                    test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>, ampere,
                    nk_attention_pack_size_bf16_ampere, nk_attention_pack_bf16_ampere, nk_attention_packed_bf16_ampere);
    run_kernel_case("nk_attention_packed_f16_ampere",
                    test_attention_packed<f16_t, cuda_backend_t, attention_weights_t::bits_11_k>, ampere,
                    nk_attention_pack_size_f16_ampere, nk_attention_pack_f16_ampere, nk_attention_packed_f16_ampere);
    run_kernel_case("nk_attention_packed_e4m3_ampere",
                    test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_11_k>, ampere,
                    nk_attention_pack_size_e4m3_ampere, nk_attention_pack_e4m3_ampere, nk_attention_packed_e4m3_ampere);
    run_kernel_case("nk_attention_packed_i8_ampere",
                    test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>, ampere,
                    nk_attention_pack_size_i8_ampere, nk_attention_pack_i8_ampere, nk_attention_packed_i8_ampere);
    run_kernel_case("nk_attention_packed_gradients_bf16_ampere",
                    test_attention_packed_gradients<bf16_t, cuda_backend_t>, ampere, nk_attention_pack_size_bf16_ampere,
                    nk_attention_pack_bf16_ampere, nk_attention_packed_gradients_bf16_ampere);
#endif // NUMKONG_TARGET_AMPERE

#if NUMKONG_TARGET_HOPPER
    auto const hopper = nk_cap_hopper_k;
    run_kernel_case("nk_dots_packed_bf16_hopper", test_dots_packed<bf16_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_bf16_hopper, nk_dots_pack_bf16_hopper, nk_dots_packed_bf16_hopper);
    run_kernel_case("nk_dots_pack_bf16_hopper", test_dots_pack_layout<bf16_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_bf16_hopper, nk_dots_packed_shape_bf16_hopper, nk_dots_pack_bf16_hopper);
    run_kernel_case("test_dots_launch_contract<bf16_t, cuda_backend_t>(nk_dots_packed_bf16_hopper)",
                    test_dots_launch_contract<bf16_t, cuda_backend_t>, hopper, nk_dots_pack_size_bf16_hopper,
                    nk_dots_packed_bf16_hopper, nk_dots_symmetric_bf16_hopper);
    run_kernel_case("nk_dots_symmetric_bf16_hopper", test_dots_symmetric<bf16_t, cuda_backend_t>, hopper,
                    nk_dots_symmetric_bf16_hopper);
    run_kernel_case("nk_dots_packed_f16_hopper", test_dots_packed<f16_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_f16_hopper, nk_dots_pack_f16_hopper, nk_dots_packed_f16_hopper);
    run_kernel_case("nk_dots_pack_f16_hopper", test_dots_pack_layout<f16_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_f16_hopper, nk_dots_packed_shape_f16_hopper, nk_dots_pack_f16_hopper);
    run_kernel_case("test_dots_launch_contract<f16_t, cuda_backend_t>(nk_dots_packed_f16_hopper)",
                    test_dots_launch_contract<f16_t, cuda_backend_t>, hopper, nk_dots_pack_size_f16_hopper,
                    nk_dots_packed_f16_hopper, nk_dots_symmetric_f16_hopper);
    run_kernel_case("nk_dots_symmetric_f16_hopper", test_dots_symmetric<f16_t, cuda_backend_t>, hopper,
                    nk_dots_symmetric_f16_hopper);
    run_kernel_case("nk_dots_packed_e2m3_hopper", test_dots_packed<e2m3_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_e2m3_hopper, nk_dots_pack_e2m3_hopper, nk_dots_packed_e2m3_hopper);
    run_kernel_case("nk_dots_pack_e2m3_hopper", test_dots_pack_layout<e2m3_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_e2m3_hopper, nk_dots_packed_shape_e2m3_hopper, nk_dots_pack_e2m3_hopper);
    run_kernel_case("test_dots_launch_contract<e2m3_t, cuda_backend_t>(nk_dots_packed_e2m3_hopper)",
                    test_dots_launch_contract<e2m3_t, cuda_backend_t>, hopper, nk_dots_pack_size_e2m3_hopper,
                    nk_dots_packed_e2m3_hopper, nk_dots_symmetric_e2m3_hopper);
    run_kernel_case("nk_dots_symmetric_e2m3_hopper", test_dots_symmetric<e2m3_t, cuda_backend_t>, hopper,
                    nk_dots_symmetric_e2m3_hopper);
    run_kernel_case("nk_dots_packed_e2m1_hopper", test_dots_packed<e2m1x2_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_e2m1_hopper, nk_dots_pack_e2m1_hopper, nk_dots_packed_e2m1_hopper);
    run_kernel_case("nk_dots_pack_e2m1_hopper", test_dots_pack_layout<e2m1x2_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_e2m1_hopper, nk_dots_packed_shape_e2m1_hopper, nk_dots_pack_e2m1_hopper);
    run_kernel_case("test_dots_launch_contract<e2m1x2_t, cuda_backend_t>(nk_dots_packed_e2m1_hopper)",
                    test_dots_launch_contract<e2m1x2_t, cuda_backend_t>, hopper, nk_dots_pack_size_e2m1_hopper,
                    nk_dots_packed_e2m1_hopper, nk_dots_symmetric_e2m1_hopper);
    run_kernel_case("nk_dots_symmetric_e2m1_hopper", test_dots_symmetric<e2m1x2_t, cuda_backend_t>, hopper,
                    nk_dots_symmetric_e2m1_hopper);
    run_kernel_case("nk_dots_packed_nvfp4_hopper", test_dots_packed<nvfp4_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_nvfp4_hopper, nk_dots_pack_nvfp4_hopper, nk_dots_packed_nvfp4_hopper);
    run_kernel_case("nk_dots_pack_nvfp4_hopper", test_dots_pack_layout<nvfp4_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_nvfp4_hopper, nk_dots_packed_shape_nvfp4_hopper, nk_dots_pack_nvfp4_hopper);
    run_kernel_case("test_dots_launch_contract<nvfp4_t, cuda_backend_t>(nk_dots_packed_nvfp4_hopper)",
                    test_dots_launch_contract<nvfp4_t, cuda_backend_t>, hopper, nk_dots_pack_size_nvfp4_hopper,
                    nk_dots_packed_nvfp4_hopper, nk_dots_symmetric_nvfp4_hopper);
    run_kernel_case("nk_dots_symmetric_nvfp4_hopper", test_dots_symmetric<nvfp4_t, cuda_backend_t>, hopper,
                    nk_dots_symmetric_nvfp4_hopper);
    run_kernel_case("nk_dots_packed_mxfp4_hopper", test_dots_packed<mxfp4_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_mxfp4_hopper, nk_dots_pack_mxfp4_hopper, nk_dots_packed_mxfp4_hopper);
    run_kernel_case("nk_dots_pack_mxfp4_hopper", test_dots_pack_layout<mxfp4_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_mxfp4_hopper, nk_dots_packed_shape_mxfp4_hopper, nk_dots_pack_mxfp4_hopper);
    run_kernel_case("test_dots_launch_contract<mxfp4_t, cuda_backend_t>(nk_dots_packed_mxfp4_hopper)",
                    test_dots_launch_contract<mxfp4_t, cuda_backend_t>, hopper, nk_dots_pack_size_mxfp4_hopper,
                    nk_dots_packed_mxfp4_hopper, nk_dots_symmetric_mxfp4_hopper);
    run_kernel_case("nk_dots_symmetric_mxfp4_hopper", test_dots_symmetric<mxfp4_t, cuda_backend_t>, hopper,
                    nk_dots_symmetric_mxfp4_hopper);
    run_kernel_case("nk_dots_packed_mxfp8e4m3_hopper", test_dots_packed<mxfp8e4m3_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_mxfp8e4m3_hopper, nk_dots_pack_mxfp8e4m3_hopper, nk_dots_packed_mxfp8e4m3_hopper);
    run_kernel_case("nk_dots_pack_mxfp8e4m3_hopper", test_dots_pack_layout<mxfp8e4m3_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_mxfp8e4m3_hopper, nk_dots_packed_shape_mxfp8e4m3_hopper,
                    nk_dots_pack_mxfp8e4m3_hopper);
    run_kernel_case("test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t>(nk_dots_packed_mxfp8e4m3_hopper)",
                    test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t>, hopper, nk_dots_pack_size_mxfp8e4m3_hopper,
                    nk_dots_packed_mxfp8e4m3_hopper, nk_dots_symmetric_mxfp8e4m3_hopper);
    run_kernel_case("nk_dots_symmetric_mxfp8e4m3_hopper", test_dots_symmetric<mxfp8e4m3_t, cuda_backend_t>, hopper,
                    nk_dots_symmetric_mxfp8e4m3_hopper);
    run_kernel_case("nk_dots_packed_mxfp8e5m2_hopper", test_dots_packed<mxfp8e5m2_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_mxfp8e5m2_hopper, nk_dots_pack_mxfp8e5m2_hopper, nk_dots_packed_mxfp8e5m2_hopper);
    run_kernel_case("nk_dots_pack_mxfp8e5m2_hopper", test_dots_pack_layout<mxfp8e5m2_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_mxfp8e5m2_hopper, nk_dots_packed_shape_mxfp8e5m2_hopper,
                    nk_dots_pack_mxfp8e5m2_hopper);
    run_kernel_case("test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t>(nk_dots_packed_mxfp8e5m2_hopper)",
                    test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t>, hopper, nk_dots_pack_size_mxfp8e5m2_hopper,
                    nk_dots_packed_mxfp8e5m2_hopper, nk_dots_symmetric_mxfp8e5m2_hopper);
    run_kernel_case("nk_dots_symmetric_mxfp8e5m2_hopper", test_dots_symmetric<mxfp8e5m2_t, cuda_backend_t>, hopper,
                    nk_dots_symmetric_mxfp8e5m2_hopper);
    run_kernel_case("nk_dots_packed_i8_hopper", test_dots_packed<i8_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_i8_hopper, nk_dots_pack_i8_hopper, nk_dots_packed_i8_hopper);
    run_kernel_case("nk_dots_pack_i8_hopper", test_dots_pack_layout<i8_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_i8_hopper, nk_dots_packed_shape_i8_hopper, nk_dots_pack_i8_hopper);
    run_kernel_case("test_dots_launch_contract<i8_t, cuda_backend_t>(nk_dots_packed_i8_hopper)",
                    test_dots_launch_contract<i8_t, cuda_backend_t>, hopper, nk_dots_pack_size_i8_hopper,
                    nk_dots_packed_i8_hopper, nk_dots_symmetric_i8_hopper);
    run_kernel_case("nk_dots_symmetric_i8_hopper", test_dots_symmetric<i8_t, cuda_backend_t>, hopper,
                    nk_dots_symmetric_i8_hopper);
    run_kernel_case("nk_dots_packed_u8_hopper", test_dots_packed<u8_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_u8_hopper, nk_dots_pack_u8_hopper, nk_dots_packed_u8_hopper);
    run_kernel_case("nk_dots_pack_u8_hopper", test_dots_pack_layout<u8_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_u8_hopper, nk_dots_packed_shape_u8_hopper, nk_dots_pack_u8_hopper);
    run_kernel_case("test_dots_launch_contract<u8_t, cuda_backend_t>(nk_dots_packed_u8_hopper)",
                    test_dots_launch_contract<u8_t, cuda_backend_t>, hopper, nk_dots_pack_size_u8_hopper,
                    nk_dots_packed_u8_hopper, nk_dots_symmetric_u8_hopper);
    run_kernel_case("nk_dots_symmetric_u8_hopper", test_dots_symmetric<u8_t, cuda_backend_t>, hopper,
                    nk_dots_symmetric_u8_hopper);
    run_kernel_case("nk_dots_packed_i4_hopper", test_dots_packed<i4x2_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_i4_hopper, nk_dots_pack_i4_hopper, nk_dots_packed_i4_hopper);
    run_kernel_case("nk_dots_pack_i4_hopper", test_dots_pack_layout<i4x2_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_i4_hopper, nk_dots_packed_shape_i4_hopper, nk_dots_pack_i4_hopper);
    run_kernel_case("test_dots_launch_contract<i4x2_t, cuda_backend_t>(nk_dots_packed_i4_hopper)",
                    test_dots_launch_contract<i4x2_t, cuda_backend_t>, hopper, nk_dots_pack_size_i4_hopper,
                    nk_dots_packed_i4_hopper, nk_dots_symmetric_i4_hopper);
    run_kernel_case("nk_dots_symmetric_i4_hopper", test_dots_symmetric<i4x2_t, cuda_backend_t>, hopper,
                    nk_dots_symmetric_i4_hopper);
    run_kernel_case("nk_dots_packed_u4_hopper", test_dots_packed<u4x2_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_u4_hopper, nk_dots_pack_u4_hopper, nk_dots_packed_u4_hopper);
    run_kernel_case("nk_dots_pack_u4_hopper", test_dots_pack_layout<u4x2_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_u4_hopper, nk_dots_packed_shape_u4_hopper, nk_dots_pack_u4_hopper);
    run_kernel_case("test_dots_launch_contract<u4x2_t, cuda_backend_t>(nk_dots_packed_u4_hopper)",
                    test_dots_launch_contract<u4x2_t, cuda_backend_t>, hopper, nk_dots_pack_size_u4_hopper,
                    nk_dots_packed_u4_hopper, nk_dots_symmetric_u4_hopper);
    run_kernel_case("nk_dots_symmetric_u4_hopper", test_dots_symmetric<u4x2_t, cuda_backend_t>, hopper,
                    nk_dots_symmetric_u4_hopper);
    run_kernel_case("nk_angulars_packed_bf16_hopper", test_angulars_packed<bf16_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_bf16_hopper, nk_dots_pack_bf16_hopper, nk_angulars_packed_bf16_hopper);
    run_kernel_case("nk_angulars_symmetric_bf16_hopper", test_angulars_symmetric<bf16_t, cuda_backend_t>, hopper,
                    nk_angulars_symmetric_bf16_hopper);
    run_kernel_case("nk_euclideans_packed_bf16_hopper", test_euclideans_packed<bf16_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_bf16_hopper, nk_dots_pack_bf16_hopper, nk_euclideans_packed_bf16_hopper);
    run_kernel_case("nk_euclideans_symmetric_bf16_hopper", test_euclideans_symmetric<bf16_t, cuda_backend_t>, hopper,
                    nk_euclideans_symmetric_bf16_hopper);
    run_kernel_case("nk_angulars_packed_f16_hopper", test_angulars_packed<f16_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_f16_hopper, nk_dots_pack_f16_hopper, nk_angulars_packed_f16_hopper);
    run_kernel_case("nk_angulars_symmetric_f16_hopper", test_angulars_symmetric<f16_t, cuda_backend_t>, hopper,
                    nk_angulars_symmetric_f16_hopper);
    run_kernel_case("nk_euclideans_packed_f16_hopper", test_euclideans_packed<f16_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_f16_hopper, nk_dots_pack_f16_hopper, nk_euclideans_packed_f16_hopper);
    run_kernel_case("nk_euclideans_symmetric_f16_hopper", test_euclideans_symmetric<f16_t, cuda_backend_t>, hopper,
                    nk_euclideans_symmetric_f16_hopper);
    run_kernel_case("nk_angulars_packed_e2m3_hopper", test_angulars_packed<e2m3_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_e2m3_hopper, nk_dots_pack_e2m3_hopper, nk_angulars_packed_e2m3_hopper);
    run_kernel_case("nk_angulars_symmetric_e2m3_hopper", test_angulars_symmetric<e2m3_t, cuda_backend_t>, hopper,
                    nk_angulars_symmetric_e2m3_hopper);
    run_kernel_case("nk_euclideans_packed_e2m3_hopper", test_euclideans_packed<e2m3_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_e2m3_hopper, nk_dots_pack_e2m3_hopper, nk_euclideans_packed_e2m3_hopper);
    run_kernel_case("nk_euclideans_symmetric_e2m3_hopper", test_euclideans_symmetric<e2m3_t, cuda_backend_t>, hopper,
                    nk_euclideans_symmetric_e2m3_hopper);
    run_kernel_case("nk_angulars_packed_e2m1_hopper", test_angulars_packed<e2m1x2_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_e2m1_hopper, nk_dots_pack_e2m1_hopper, nk_angulars_packed_e2m1_hopper);
    run_kernel_case("nk_angulars_symmetric_e2m1_hopper", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>, hopper,
                    nk_angulars_symmetric_e2m1_hopper);
    run_kernel_case("nk_euclideans_packed_e2m1_hopper", test_euclideans_packed<e2m1x2_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_e2m1_hopper, nk_dots_pack_e2m1_hopper, nk_euclideans_packed_e2m1_hopper);
    run_kernel_case("nk_euclideans_symmetric_e2m1_hopper", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>, hopper,
                    nk_euclideans_symmetric_e2m1_hopper);
    run_kernel_case("nk_angulars_packed_i8_hopper", test_angulars_packed<i8_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_i8_hopper, nk_dots_pack_i8_hopper, nk_angulars_packed_i8_hopper);
    run_kernel_case("nk_angulars_symmetric_i8_hopper", test_angulars_symmetric<i8_t, cuda_backend_t>, hopper,
                    nk_angulars_symmetric_i8_hopper);
    run_kernel_case("nk_euclideans_packed_i8_hopper", test_euclideans_packed<i8_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_i8_hopper, nk_dots_pack_i8_hopper, nk_euclideans_packed_i8_hopper);
    run_kernel_case("nk_euclideans_symmetric_i8_hopper", test_euclideans_symmetric<i8_t, cuda_backend_t>, hopper,
                    nk_euclideans_symmetric_i8_hopper);
    run_kernel_case("nk_angulars_packed_u8_hopper", test_angulars_packed<u8_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_u8_hopper, nk_dots_pack_u8_hopper, nk_angulars_packed_u8_hopper);
    run_kernel_case("nk_angulars_symmetric_u8_hopper", test_angulars_symmetric<u8_t, cuda_backend_t>, hopper,
                    nk_angulars_symmetric_u8_hopper);
    run_kernel_case("nk_euclideans_packed_u8_hopper", test_euclideans_packed<u8_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_u8_hopper, nk_dots_pack_u8_hopper, nk_euclideans_packed_u8_hopper);
    run_kernel_case("nk_euclideans_symmetric_u8_hopper", test_euclideans_symmetric<u8_t, cuda_backend_t>, hopper,
                    nk_euclideans_symmetric_u8_hopper);
    run_kernel_case("nk_angulars_packed_i4_hopper", test_angulars_packed<i4x2_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_i4_hopper, nk_dots_pack_i4_hopper, nk_angulars_packed_i4_hopper);
    run_kernel_case("nk_angulars_symmetric_i4_hopper", test_angulars_symmetric<i4x2_t, cuda_backend_t>, hopper,
                    nk_angulars_symmetric_i4_hopper);
    run_kernel_case("nk_euclideans_packed_i4_hopper", test_euclideans_packed<i4x2_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_i4_hopper, nk_dots_pack_i4_hopper, nk_euclideans_packed_i4_hopper);
    run_kernel_case("nk_euclideans_symmetric_i4_hopper", test_euclideans_symmetric<i4x2_t, cuda_backend_t>, hopper,
                    nk_euclideans_symmetric_i4_hopper);
    run_kernel_case("nk_angulars_packed_u4_hopper", test_angulars_packed<u4x2_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_u4_hopper, nk_dots_pack_u4_hopper, nk_angulars_packed_u4_hopper);
    run_kernel_case("nk_angulars_symmetric_u4_hopper", test_angulars_symmetric<u4x2_t, cuda_backend_t>, hopper,
                    nk_angulars_symmetric_u4_hopper);
    run_kernel_case("nk_euclideans_packed_u4_hopper", test_euclideans_packed<u4x2_t, cuda_backend_t>, hopper,
                    nk_dots_pack_size_u4_hopper, nk_dots_pack_u4_hopper, nk_euclideans_packed_u4_hopper);
    run_kernel_case("nk_euclideans_symmetric_u4_hopper", test_euclideans_symmetric<u4x2_t, cuda_backend_t>, hopper,
                    nk_euclideans_symmetric_u4_hopper);
    run_kernel_case("nk_attention_packed_bf16_hopper",
                    test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>, hopper,
                    nk_attention_pack_size_bf16_hopper, nk_attention_pack_bf16_hopper, nk_attention_packed_bf16_hopper);
    run_kernel_case("nk_attention_packed_e4m3_hopper",
                    test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>, hopper,
                    nk_attention_pack_size_e4m3_hopper, nk_attention_pack_e4m3_hopper, nk_attention_packed_e4m3_hopper);
    run_kernel_case("nk_attention_packed_i8_hopper",
                    test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>, hopper,
                    nk_attention_pack_size_i8_hopper, nk_attention_pack_i8_hopper, nk_attention_packed_i8_hopper);
    run_kernel_case("nk_attention_packed_gradients_bf16_hopper",
                    test_attention_packed_gradients<bf16_t, cuda_backend_t>, hopper, nk_attention_pack_size_bf16_hopper,
                    nk_attention_pack_bf16_hopper, nk_attention_packed_gradients_bf16_hopper);
#endif // NUMKONG_TARGET_HOPPER

#if NUMKONG_TARGET_BLACKWELL
    auto const blackwell = nk_cap_blackwell_k;
    run_kernel_case("nk_dots_packed_bf16_blackwell", test_dots_packed<bf16_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_bf16_blackwell, nk_dots_pack_bf16_blackwell,
                    with_workspace(nk_dots_packed_bf16_blackwell, nk_dots_packed_workspace_size_bf16_blackwell));
    run_kernel_case("nk_dots_pack_bf16_blackwell", test_dots_pack_layout<bf16_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_bf16_blackwell, nk_dots_packed_shape_bf16_blackwell, nk_dots_pack_bf16_blackwell);
    run_kernel_case("test_dots_launch_contract<bf16_t, cuda_backend_t>(nk_dots_packed_bf16_blackwell)",
                    test_dots_launch_contract<bf16_t, cuda_backend_t>, blackwell, nk_dots_pack_size_bf16_blackwell,
                    nk_dots_packed_bf16_blackwell, nk_dots_symmetric_bf16_blackwell);
    run_kernel_case("nk_dots_symmetric_bf16_blackwell", test_dots_symmetric<bf16_t, cuda_backend_t>, blackwell,
                    nk_dots_symmetric_bf16_blackwell);
    run_kernel_case("nk_dots_packed_f16_blackwell", test_dots_packed<f16_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_f16_blackwell, nk_dots_pack_f16_blackwell,
                    with_workspace(nk_dots_packed_f16_blackwell, nk_dots_packed_workspace_size_f16_blackwell));
    run_kernel_case("nk_dots_pack_f16_blackwell", test_dots_pack_layout<f16_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_f16_blackwell, nk_dots_packed_shape_f16_blackwell, nk_dots_pack_f16_blackwell);
    run_kernel_case("test_dots_launch_contract<f16_t, cuda_backend_t>(nk_dots_packed_f16_blackwell)",
                    test_dots_launch_contract<f16_t, cuda_backend_t>, blackwell, nk_dots_pack_size_f16_blackwell,
                    nk_dots_packed_f16_blackwell, nk_dots_symmetric_f16_blackwell);
    run_kernel_case("nk_dots_symmetric_f16_blackwell", test_dots_symmetric<f16_t, cuda_backend_t>, blackwell,
                    nk_dots_symmetric_f16_blackwell);
    run_kernel_case("nk_dots_packed_e5m2_blackwell", test_dots_packed<e5m2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e5m2_blackwell, nk_dots_pack_e5m2_blackwell,
                    with_workspace(nk_dots_packed_e5m2_blackwell, nk_dots_packed_workspace_size_e5m2_blackwell));
    run_kernel_case("nk_dots_pack_e5m2_blackwell", test_dots_pack_layout<e5m2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e5m2_blackwell, nk_dots_packed_shape_e5m2_blackwell, nk_dots_pack_e5m2_blackwell);
    run_kernel_case("test_dots_launch_contract<e5m2_t, cuda_backend_t>(nk_dots_packed_e5m2_blackwell)",
                    test_dots_launch_contract<e5m2_t, cuda_backend_t>, blackwell, nk_dots_pack_size_e5m2_blackwell,
                    nk_dots_packed_e5m2_blackwell, nk_dots_symmetric_e5m2_blackwell);
    run_kernel_case("nk_dots_symmetric_e5m2_blackwell", test_dots_symmetric<e5m2_t, cuda_backend_t>, blackwell,
                    nk_dots_symmetric_e5m2_blackwell);
    run_kernel_case("nk_dots_packed_e4m3_blackwell", test_dots_packed<e4m3_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e4m3_blackwell, nk_dots_pack_e4m3_blackwell,
                    with_workspace(nk_dots_packed_e4m3_blackwell, nk_dots_packed_workspace_size_e4m3_blackwell));
    run_kernel_case("nk_dots_pack_e4m3_blackwell", test_dots_pack_layout<e4m3_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e4m3_blackwell, nk_dots_packed_shape_e4m3_blackwell, nk_dots_pack_e4m3_blackwell);
    run_kernel_case("test_dots_launch_contract<e4m3_t, cuda_backend_t>(nk_dots_packed_e4m3_blackwell)",
                    test_dots_launch_contract<e4m3_t, cuda_backend_t>, blackwell, nk_dots_pack_size_e4m3_blackwell,
                    nk_dots_packed_e4m3_blackwell, nk_dots_symmetric_e4m3_blackwell);
    run_kernel_case("nk_dots_symmetric_e4m3_blackwell", test_dots_symmetric<e4m3_t, cuda_backend_t>, blackwell,
                    nk_dots_symmetric_e4m3_blackwell);
    run_kernel_case("nk_dots_packed_e3m2_blackwell", test_dots_packed<e3m2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e3m2_blackwell, nk_dots_pack_e3m2_blackwell,
                    with_workspace(nk_dots_packed_e3m2_blackwell, nk_dots_packed_workspace_size_e3m2_blackwell));
    run_kernel_case("nk_dots_pack_e3m2_blackwell", test_dots_pack_layout<e3m2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e3m2_blackwell, nk_dots_packed_shape_e3m2_blackwell, nk_dots_pack_e3m2_blackwell);
    run_kernel_case("test_dots_launch_contract<e3m2_t, cuda_backend_t>(nk_dots_packed_e3m2_blackwell)",
                    test_dots_launch_contract<e3m2_t, cuda_backend_t>, blackwell, nk_dots_pack_size_e3m2_blackwell,
                    nk_dots_packed_e3m2_blackwell, nk_dots_symmetric_e3m2_blackwell);
    run_kernel_case("nk_dots_symmetric_e3m2_blackwell", test_dots_symmetric<e3m2_t, cuda_backend_t>, blackwell,
                    nk_dots_symmetric_e3m2_blackwell);
    run_kernel_case("nk_dots_packed_e2m3_blackwell", test_dots_packed<e2m3_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e2m3_blackwell, nk_dots_pack_e2m3_blackwell,
                    with_workspace(nk_dots_packed_e2m3_blackwell, nk_dots_packed_workspace_size_e2m3_blackwell));
    run_kernel_case("nk_dots_pack_e2m3_blackwell", test_dots_pack_layout<e2m3_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e2m3_blackwell, nk_dots_packed_shape_e2m3_blackwell, nk_dots_pack_e2m3_blackwell);
    run_kernel_case("test_dots_launch_contract<e2m3_t, cuda_backend_t>(nk_dots_packed_e2m3_blackwell)",
                    test_dots_launch_contract<e2m3_t, cuda_backend_t>, blackwell, nk_dots_pack_size_e2m3_blackwell,
                    nk_dots_packed_e2m3_blackwell, nk_dots_symmetric_e2m3_blackwell);
    run_kernel_case("nk_dots_symmetric_e2m3_blackwell", test_dots_symmetric<e2m3_t, cuda_backend_t>, blackwell,
                    nk_dots_symmetric_e2m3_blackwell);
    run_kernel_case("nk_dots_packed_e2m1_blackwell", test_dots_packed<e2m1x2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e2m1_blackwell, nk_dots_pack_e2m1_blackwell,
                    with_workspace(nk_dots_packed_e2m1_blackwell, nk_dots_packed_workspace_size_e2m1_blackwell));
    run_kernel_case("nk_dots_pack_e2m1_blackwell", test_dots_pack_layout<e2m1x2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e2m1_blackwell, nk_dots_packed_shape_e2m1_blackwell, nk_dots_pack_e2m1_blackwell);
    run_kernel_case("test_dots_launch_contract<e2m1x2_t, cuda_backend_t>(nk_dots_packed_e2m1_blackwell)",
                    test_dots_launch_contract<e2m1x2_t, cuda_backend_t>, blackwell, nk_dots_pack_size_e2m1_blackwell,
                    nk_dots_packed_e2m1_blackwell, nk_dots_symmetric_e2m1_blackwell);
    run_kernel_case("nk_dots_symmetric_e2m1_blackwell", test_dots_symmetric<e2m1x2_t, cuda_backend_t>, blackwell,
                    nk_dots_symmetric_e2m1_blackwell);
    run_kernel_case("nk_dots_packed_nvfp4_blackwell", test_dots_packed<nvfp4_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_nvfp4_blackwell, nk_dots_pack_nvfp4_blackwell, nk_dots_packed_nvfp4_blackwell);
    run_kernel_case("nk_dots_pack_nvfp4_blackwell", test_dots_pack_layout<nvfp4_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_nvfp4_blackwell, nk_dots_packed_shape_nvfp4_blackwell,
                    nk_dots_pack_nvfp4_blackwell);
    run_kernel_case("test_dots_launch_contract<nvfp4_t, cuda_backend_t>(nk_dots_packed_nvfp4_blackwell)",
                    test_dots_launch_contract<nvfp4_t, cuda_backend_t>, blackwell, nk_dots_pack_size_nvfp4_blackwell,
                    nk_dots_packed_nvfp4_blackwell, nk_dots_symmetric_nvfp4_blackwell);
    run_kernel_case("nk_dots_symmetric_nvfp4_blackwell", test_dots_symmetric<nvfp4_t, cuda_backend_t>, blackwell,
                    nk_dots_symmetric_nvfp4_blackwell);
    run_kernel_case("nk_dots_packed_mxfp4_blackwell", test_dots_packed<mxfp4_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_mxfp4_blackwell, nk_dots_pack_mxfp4_blackwell, nk_dots_packed_mxfp4_blackwell);
    run_kernel_case("nk_dots_pack_mxfp4_blackwell", test_dots_pack_layout<mxfp4_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_mxfp4_blackwell, nk_dots_packed_shape_mxfp4_blackwell,
                    nk_dots_pack_mxfp4_blackwell);
    run_kernel_case("test_dots_launch_contract<mxfp4_t, cuda_backend_t>(nk_dots_packed_mxfp4_blackwell)",
                    test_dots_launch_contract<mxfp4_t, cuda_backend_t>, blackwell, nk_dots_pack_size_mxfp4_blackwell,
                    nk_dots_packed_mxfp4_blackwell, nk_dots_symmetric_mxfp4_blackwell);
    run_kernel_case("nk_dots_symmetric_mxfp4_blackwell", test_dots_symmetric<mxfp4_t, cuda_backend_t>, blackwell,
                    nk_dots_symmetric_mxfp4_blackwell);
    run_kernel_case("nk_dots_packed_mxfp8e4m3_blackwell", test_dots_packed<mxfp8e4m3_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_mxfp8e4m3_blackwell, nk_dots_pack_mxfp8e4m3_blackwell,
                    nk_dots_packed_mxfp8e4m3_blackwell);
    run_kernel_case("nk_dots_pack_mxfp8e4m3_blackwell", test_dots_pack_layout<mxfp8e4m3_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_mxfp8e4m3_blackwell, nk_dots_packed_shape_mxfp8e4m3_blackwell,
                    nk_dots_pack_mxfp8e4m3_blackwell);
    run_kernel_case("test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t>(nk_dots_packed_mxfp8e4m3_blackwell)",
                    test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_mxfp8e4m3_blackwell, nk_dots_packed_mxfp8e4m3_blackwell,
                    nk_dots_symmetric_mxfp8e4m3_blackwell);
    run_kernel_case("nk_dots_symmetric_mxfp8e4m3_blackwell", test_dots_symmetric<mxfp8e4m3_t, cuda_backend_t>,
                    blackwell, nk_dots_symmetric_mxfp8e4m3_blackwell);
    run_kernel_case("nk_dots_packed_mxfp8e5m2_blackwell", test_dots_packed<mxfp8e5m2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_mxfp8e5m2_blackwell, nk_dots_pack_mxfp8e5m2_blackwell,
                    nk_dots_packed_mxfp8e5m2_blackwell);
    run_kernel_case("nk_dots_pack_mxfp8e5m2_blackwell", test_dots_pack_layout<mxfp8e5m2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_mxfp8e5m2_blackwell, nk_dots_packed_shape_mxfp8e5m2_blackwell,
                    nk_dots_pack_mxfp8e5m2_blackwell);
    run_kernel_case("test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t>(nk_dots_packed_mxfp8e5m2_blackwell)",
                    test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_mxfp8e5m2_blackwell, nk_dots_packed_mxfp8e5m2_blackwell,
                    nk_dots_symmetric_mxfp8e5m2_blackwell);
    run_kernel_case("nk_dots_symmetric_mxfp8e5m2_blackwell", test_dots_symmetric<mxfp8e5m2_t, cuda_backend_t>,
                    blackwell, nk_dots_symmetric_mxfp8e5m2_blackwell);
    run_kernel_case("nk_angulars_packed_bf16_blackwell", test_angulars_packed<bf16_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_bf16_blackwell, nk_dots_pack_bf16_blackwell, nk_angulars_packed_bf16_blackwell);
    run_kernel_case("nk_angulars_symmetric_bf16_blackwell", test_angulars_symmetric<bf16_t, cuda_backend_t>, blackwell,
                    nk_angulars_symmetric_bf16_blackwell);
    run_kernel_case("nk_euclideans_packed_bf16_blackwell", test_euclideans_packed<bf16_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_bf16_blackwell, nk_dots_pack_bf16_blackwell, nk_euclideans_packed_bf16_blackwell);
    run_kernel_case("nk_euclideans_symmetric_bf16_blackwell", test_euclideans_symmetric<bf16_t, cuda_backend_t>,
                    blackwell, nk_euclideans_symmetric_bf16_blackwell);
    run_kernel_case("nk_angulars_packed_f16_blackwell", test_angulars_packed<f16_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_f16_blackwell, nk_dots_pack_f16_blackwell, nk_angulars_packed_f16_blackwell);
    run_kernel_case("nk_angulars_symmetric_f16_blackwell", test_angulars_symmetric<f16_t, cuda_backend_t>, blackwell,
                    nk_angulars_symmetric_f16_blackwell);
    run_kernel_case("nk_euclideans_packed_f16_blackwell", test_euclideans_packed<f16_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_f16_blackwell, nk_dots_pack_f16_blackwell, nk_euclideans_packed_f16_blackwell);
    run_kernel_case("nk_euclideans_symmetric_f16_blackwell", test_euclideans_symmetric<f16_t, cuda_backend_t>,
                    blackwell, nk_euclideans_symmetric_f16_blackwell);
    run_kernel_case("nk_angulars_packed_e5m2_blackwell", test_angulars_packed<e5m2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e5m2_blackwell, nk_dots_pack_e5m2_blackwell, nk_angulars_packed_e5m2_blackwell);
    run_kernel_case("nk_angulars_symmetric_e5m2_blackwell", test_angulars_symmetric<e5m2_t, cuda_backend_t>, blackwell,
                    nk_angulars_symmetric_e5m2_blackwell);
    run_kernel_case("nk_euclideans_packed_e5m2_blackwell", test_euclideans_packed<e5m2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e5m2_blackwell, nk_dots_pack_e5m2_blackwell, nk_euclideans_packed_e5m2_blackwell);
    run_kernel_case("nk_euclideans_symmetric_e5m2_blackwell", test_euclideans_symmetric<e5m2_t, cuda_backend_t>,
                    blackwell, nk_euclideans_symmetric_e5m2_blackwell);
    run_kernel_case("nk_angulars_packed_e4m3_blackwell", test_angulars_packed<e4m3_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e4m3_blackwell, nk_dots_pack_e4m3_blackwell, nk_angulars_packed_e4m3_blackwell);
    run_kernel_case("nk_angulars_symmetric_e4m3_blackwell", test_angulars_symmetric<e4m3_t, cuda_backend_t>, blackwell,
                    nk_angulars_symmetric_e4m3_blackwell);
    run_kernel_case("nk_euclideans_packed_e4m3_blackwell", test_euclideans_packed<e4m3_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e4m3_blackwell, nk_dots_pack_e4m3_blackwell, nk_euclideans_packed_e4m3_blackwell);
    run_kernel_case("nk_euclideans_symmetric_e4m3_blackwell", test_euclideans_symmetric<e4m3_t, cuda_backend_t>,
                    blackwell, nk_euclideans_symmetric_e4m3_blackwell);
    run_kernel_case("nk_angulars_packed_e3m2_blackwell", test_angulars_packed<e3m2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e3m2_blackwell, nk_dots_pack_e3m2_blackwell, nk_angulars_packed_e3m2_blackwell);
    run_kernel_case("nk_angulars_symmetric_e3m2_blackwell", test_angulars_symmetric<e3m2_t, cuda_backend_t>, blackwell,
                    nk_angulars_symmetric_e3m2_blackwell);
    run_kernel_case("nk_euclideans_packed_e3m2_blackwell", test_euclideans_packed<e3m2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e3m2_blackwell, nk_dots_pack_e3m2_blackwell, nk_euclideans_packed_e3m2_blackwell);
    run_kernel_case("nk_euclideans_symmetric_e3m2_blackwell", test_euclideans_symmetric<e3m2_t, cuda_backend_t>,
                    blackwell, nk_euclideans_symmetric_e3m2_blackwell);
    run_kernel_case("nk_angulars_packed_e2m3_blackwell", test_angulars_packed<e2m3_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e2m3_blackwell, nk_dots_pack_e2m3_blackwell, nk_angulars_packed_e2m3_blackwell);
    run_kernel_case("nk_angulars_symmetric_e2m3_blackwell", test_angulars_symmetric<e2m3_t, cuda_backend_t>, blackwell,
                    nk_angulars_symmetric_e2m3_blackwell);
    run_kernel_case("nk_euclideans_packed_e2m3_blackwell", test_euclideans_packed<e2m3_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e2m3_blackwell, nk_dots_pack_e2m3_blackwell, nk_euclideans_packed_e2m3_blackwell);
    run_kernel_case("nk_euclideans_symmetric_e2m3_blackwell", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
                    blackwell, nk_euclideans_symmetric_e2m3_blackwell);
    run_kernel_case("nk_angulars_packed_e2m1_blackwell", test_angulars_packed<e2m1x2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e2m1_blackwell, nk_dots_pack_e2m1_blackwell, nk_angulars_packed_e2m1_blackwell);
    run_kernel_case("nk_angulars_symmetric_e2m1_blackwell", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
                    blackwell, nk_angulars_symmetric_e2m1_blackwell);
    run_kernel_case("nk_euclideans_packed_e2m1_blackwell", test_euclideans_packed<e2m1x2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_e2m1_blackwell, nk_dots_pack_e2m1_blackwell, nk_euclideans_packed_e2m1_blackwell);
    run_kernel_case("nk_euclideans_symmetric_e2m1_blackwell", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
                    blackwell, nk_euclideans_symmetric_e2m1_blackwell);
    run_kernel_case("nk_angulars_packed_nvfp4_blackwell", test_angulars_packed<nvfp4_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_nvfp4_blackwell, nk_dots_pack_nvfp4_blackwell,
                    nk_angulars_packed_nvfp4_blackwell);
    run_kernel_case("nk_angulars_symmetric_nvfp4_blackwell", test_angulars_symmetric<nvfp4_t, cuda_backend_t>,
                    blackwell, nk_angulars_symmetric_nvfp4_blackwell);
    run_kernel_case("nk_euclideans_packed_nvfp4_blackwell", test_euclideans_packed<nvfp4_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_nvfp4_blackwell, nk_dots_pack_nvfp4_blackwell,
                    nk_euclideans_packed_nvfp4_blackwell);
    run_kernel_case("nk_euclideans_symmetric_nvfp4_blackwell", test_euclideans_symmetric<nvfp4_t, cuda_backend_t>,
                    blackwell, nk_euclideans_symmetric_nvfp4_blackwell);
    run_kernel_case("nk_angulars_packed_mxfp4_blackwell", test_angulars_packed<mxfp4_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_mxfp4_blackwell, nk_dots_pack_mxfp4_blackwell,
                    nk_angulars_packed_mxfp4_blackwell);
    run_kernel_case("nk_angulars_symmetric_mxfp4_blackwell", test_angulars_symmetric<mxfp4_t, cuda_backend_t>,
                    blackwell, nk_angulars_symmetric_mxfp4_blackwell);
    run_kernel_case("nk_euclideans_packed_mxfp4_blackwell", test_euclideans_packed<mxfp4_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_mxfp4_blackwell, nk_dots_pack_mxfp4_blackwell,
                    nk_euclideans_packed_mxfp4_blackwell);
    run_kernel_case("nk_euclideans_symmetric_mxfp4_blackwell", test_euclideans_symmetric<mxfp4_t, cuda_backend_t>,
                    blackwell, nk_euclideans_symmetric_mxfp4_blackwell);
    run_kernel_case("nk_angulars_packed_mxfp8e4m3_blackwell", test_angulars_packed<mxfp8e4m3_t, cuda_backend_t>,
                    blackwell, nk_dots_pack_size_mxfp8e4m3_blackwell, nk_dots_pack_mxfp8e4m3_blackwell,
                    nk_angulars_packed_mxfp8e4m3_blackwell);
    run_kernel_case("nk_angulars_symmetric_mxfp8e4m3_blackwell", test_angulars_symmetric<mxfp8e4m3_t, cuda_backend_t>,
                    blackwell, nk_angulars_symmetric_mxfp8e4m3_blackwell);
    run_kernel_case("nk_euclideans_packed_mxfp8e4m3_blackwell", test_euclideans_packed<mxfp8e4m3_t, cuda_backend_t>,
                    blackwell, nk_dots_pack_size_mxfp8e4m3_blackwell, nk_dots_pack_mxfp8e4m3_blackwell,
                    nk_euclideans_packed_mxfp8e4m3_blackwell);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e4m3_blackwell",
                    test_euclideans_symmetric<mxfp8e4m3_t, cuda_backend_t>, blackwell,
                    nk_euclideans_symmetric_mxfp8e4m3_blackwell);
    run_kernel_case("nk_angulars_packed_mxfp8e5m2_blackwell", test_angulars_packed<mxfp8e5m2_t, cuda_backend_t>,
                    blackwell, nk_dots_pack_size_mxfp8e5m2_blackwell, nk_dots_pack_mxfp8e5m2_blackwell,
                    nk_angulars_packed_mxfp8e5m2_blackwell);
    run_kernel_case("nk_angulars_symmetric_mxfp8e5m2_blackwell", test_angulars_symmetric<mxfp8e5m2_t, cuda_backend_t>,
                    blackwell, nk_angulars_symmetric_mxfp8e5m2_blackwell);
    run_kernel_case("nk_euclideans_packed_mxfp8e5m2_blackwell", test_euclideans_packed<mxfp8e5m2_t, cuda_backend_t>,
                    blackwell, nk_dots_pack_size_mxfp8e5m2_blackwell, nk_dots_pack_mxfp8e5m2_blackwell,
                    nk_euclideans_packed_mxfp8e5m2_blackwell);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e5m2_blackwell",
                    test_euclideans_symmetric<mxfp8e5m2_t, cuda_backend_t>, blackwell,
                    nk_euclideans_symmetric_mxfp8e5m2_blackwell);
    run_kernel_case("nk_dots_packed_i8_blackwell", test_dots_packed<i8_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_i8_blackwell, nk_dots_pack_i8_blackwell, nk_dots_packed_i8_blackwell);
    run_kernel_case("nk_dots_pack_i8_blackwell", test_dots_pack_layout<i8_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_i8_blackwell, nk_dots_packed_shape_i8_blackwell, nk_dots_pack_i8_blackwell);
    run_kernel_case("test_dots_launch_contract<i8_t, cuda_backend_t>(nk_dots_packed_i8_blackwell)",
                    test_dots_launch_contract<i8_t, cuda_backend_t>, blackwell, nk_dots_pack_size_i8_blackwell,
                    nk_dots_packed_i8_blackwell, nk_dots_symmetric_i8_blackwell);
    run_kernel_case("nk_dots_symmetric_i8_blackwell", test_dots_symmetric<i8_t, cuda_backend_t>, blackwell,
                    nk_dots_symmetric_i8_blackwell);
    run_kernel_case("nk_dots_packed_u8_blackwell", test_dots_packed<u8_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_u8_blackwell, nk_dots_pack_u8_blackwell, nk_dots_packed_u8_blackwell);
    run_kernel_case("nk_dots_pack_u8_blackwell", test_dots_pack_layout<u8_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_u8_blackwell, nk_dots_packed_shape_u8_blackwell, nk_dots_pack_u8_blackwell);
    run_kernel_case("test_dots_launch_contract<u8_t, cuda_backend_t>(nk_dots_packed_u8_blackwell)",
                    test_dots_launch_contract<u8_t, cuda_backend_t>, blackwell, nk_dots_pack_size_u8_blackwell,
                    nk_dots_packed_u8_blackwell, nk_dots_symmetric_u8_blackwell);
    run_kernel_case("nk_dots_symmetric_u8_blackwell", test_dots_symmetric<u8_t, cuda_backend_t>, blackwell,
                    nk_dots_symmetric_u8_blackwell);
    run_kernel_case("nk_dots_packed_i4_blackwell", test_dots_packed<i4x2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_i4_blackwell, nk_dots_pack_i4_blackwell, nk_dots_packed_i4_blackwell);
    run_kernel_case("nk_dots_pack_i4_blackwell", test_dots_pack_layout<i4x2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_i4_blackwell, nk_dots_packed_shape_i4_blackwell, nk_dots_pack_i4_blackwell);
    run_kernel_case("test_dots_launch_contract<i4x2_t, cuda_backend_t>(nk_dots_packed_i4_blackwell)",
                    test_dots_launch_contract<i4x2_t, cuda_backend_t>, blackwell, nk_dots_pack_size_i4_blackwell,
                    nk_dots_packed_i4_blackwell, nk_dots_symmetric_i4_blackwell);
    run_kernel_case("nk_dots_symmetric_i4_blackwell", test_dots_symmetric<i4x2_t, cuda_backend_t>, blackwell,
                    nk_dots_symmetric_i4_blackwell);
    run_kernel_case("nk_dots_packed_u4_blackwell", test_dots_packed<u4x2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_u4_blackwell, nk_dots_pack_u4_blackwell, nk_dots_packed_u4_blackwell);
    run_kernel_case("nk_dots_pack_u4_blackwell", test_dots_pack_layout<u4x2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_u4_blackwell, nk_dots_packed_shape_u4_blackwell, nk_dots_pack_u4_blackwell);
    run_kernel_case("test_dots_launch_contract<u4x2_t, cuda_backend_t>(nk_dots_packed_u4_blackwell)",
                    test_dots_launch_contract<u4x2_t, cuda_backend_t>, blackwell, nk_dots_pack_size_u4_blackwell,
                    nk_dots_packed_u4_blackwell, nk_dots_symmetric_u4_blackwell);
    run_kernel_case("nk_dots_symmetric_u4_blackwell", test_dots_symmetric<u4x2_t, cuda_backend_t>, blackwell,
                    nk_dots_symmetric_u4_blackwell);
    run_kernel_case("nk_angulars_packed_i8_blackwell", test_angulars_packed<i8_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_i8_blackwell, nk_dots_pack_i8_blackwell, nk_angulars_packed_i8_blackwell);
    run_kernel_case("nk_angulars_symmetric_i8_blackwell", test_angulars_symmetric<i8_t, cuda_backend_t>, blackwell,
                    nk_angulars_symmetric_i8_blackwell);
    run_kernel_case("nk_euclideans_packed_i8_blackwell", test_euclideans_packed<i8_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_i8_blackwell, nk_dots_pack_i8_blackwell, nk_euclideans_packed_i8_blackwell);
    run_kernel_case("nk_euclideans_symmetric_i8_blackwell", test_euclideans_symmetric<i8_t, cuda_backend_t>, blackwell,
                    nk_euclideans_symmetric_i8_blackwell);
    run_kernel_case("nk_angulars_packed_u8_blackwell", test_angulars_packed<u8_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_u8_blackwell, nk_dots_pack_u8_blackwell, nk_angulars_packed_u8_blackwell);
    run_kernel_case("nk_angulars_symmetric_u8_blackwell", test_angulars_symmetric<u8_t, cuda_backend_t>, blackwell,
                    nk_angulars_symmetric_u8_blackwell);
    run_kernel_case("nk_euclideans_packed_u8_blackwell", test_euclideans_packed<u8_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_u8_blackwell, nk_dots_pack_u8_blackwell, nk_euclideans_packed_u8_blackwell);
    run_kernel_case("nk_euclideans_symmetric_u8_blackwell", test_euclideans_symmetric<u8_t, cuda_backend_t>, blackwell,
                    nk_euclideans_symmetric_u8_blackwell);
    run_kernel_case("nk_angulars_packed_i4_blackwell", test_angulars_packed<i4x2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_i4_blackwell, nk_dots_pack_i4_blackwell, nk_angulars_packed_i4_blackwell);
    run_kernel_case("nk_angulars_symmetric_i4_blackwell", test_angulars_symmetric<i4x2_t, cuda_backend_t>, blackwell,
                    nk_angulars_symmetric_i4_blackwell);
    run_kernel_case("nk_euclideans_packed_i4_blackwell", test_euclideans_packed<i4x2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_i4_blackwell, nk_dots_pack_i4_blackwell, nk_euclideans_packed_i4_blackwell);
    run_kernel_case("nk_euclideans_symmetric_i4_blackwell", test_euclideans_symmetric<i4x2_t, cuda_backend_t>,
                    blackwell, nk_euclideans_symmetric_i4_blackwell);
    run_kernel_case("nk_angulars_packed_u4_blackwell", test_angulars_packed<u4x2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_u4_blackwell, nk_dots_pack_u4_blackwell, nk_angulars_packed_u4_blackwell);
    run_kernel_case("nk_angulars_symmetric_u4_blackwell", test_angulars_symmetric<u4x2_t, cuda_backend_t>, blackwell,
                    nk_angulars_symmetric_u4_blackwell);
    run_kernel_case("nk_euclideans_packed_u4_blackwell", test_euclideans_packed<u4x2_t, cuda_backend_t>, blackwell,
                    nk_dots_pack_size_u4_blackwell, nk_dots_pack_u4_blackwell, nk_euclideans_packed_u4_blackwell);
    run_kernel_case("nk_euclideans_symmetric_u4_blackwell", test_euclideans_symmetric<u4x2_t, cuda_backend_t>,
                    blackwell, nk_euclideans_symmetric_u4_blackwell);
    run_kernel_case("nk_attention_packed_bf16_blackwell",
                    test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>, blackwell,
                    nk_attention_pack_size_bf16_blackwell, nk_attention_pack_bf16_blackwell,
                    nk_attention_packed_bf16_blackwell);
    run_kernel_case("nk_attention_packed_f16_blackwell",
                    test_attention_packed<f16_t, cuda_backend_t, attention_weights_t::bits_11_k>, blackwell,
                    nk_attention_pack_size_f16_blackwell, nk_attention_pack_f16_blackwell,
                    nk_attention_packed_f16_blackwell);
    run_kernel_case("nk_attention_packed_e4m3_blackwell",
                    test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>, blackwell,
                    nk_attention_pack_size_e4m3_blackwell, nk_attention_pack_e4m3_blackwell,
                    nk_attention_packed_e4m3_blackwell);
    run_kernel_case("nk_attention_packed_i8_blackwell",
                    test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>, blackwell,
                    nk_attention_pack_size_i8_blackwell, nk_attention_pack_i8_blackwell,
                    nk_attention_packed_i8_blackwell);
    run_kernel_case("nk_attention_packed_gradients_bf16_blackwell",
                    test_attention_packed_gradients<bf16_t, cuda_backend_t>, blackwell,
                    nk_attention_pack_size_bf16_blackwell, nk_attention_pack_bf16_blackwell,
                    nk_attention_packed_gradients_bf16_blackwell);
    run_kernel_case("nk_rmsds_packed_f16_blackwell", test_rmsds_packed<f16_t, cuda_backend_t>, blackwell,
                    nk_rmsds_packed_f16_blackwell);
    run_kernel_case("nk_chamfers_packed_f16_blackwell",
                    test_shapes_flips<f16_t, nk_kernel_chamfers_packed_k, cuda_backend_t>, blackwell,
                    nk_chamfers_packed_f16_blackwell);
    run_kernel_case("nk_hausdorffs_packed_f16_blackwell",
                    test_shapes_flips<f16_t, nk_kernel_hausdorffs_packed_k, cuda_backend_t>, blackwell,
                    nk_hausdorffs_packed_f16_blackwell);
#endif // NUMKONG_TARGET_BLACKWELL

#if NUMKONG_TARGET_BLACKWELLRTX
    auto const blackwellrtx = nk_cap_blackwellrtx_k;
    run_kernel_case("nk_dots_packed_e5m2_blackwellrtx", test_dots_packed<e5m2_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx,
                    nk_dots_packed_e5m2_blackwellrtx);
    run_kernel_case("nk_dots_pack_e5m2_blackwellrtx", test_dots_pack_layout<e5m2_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_packed_shape_e5m2_blackwellrtx,
                    nk_dots_pack_e5m2_blackwellrtx);
    run_kernel_case("test_dots_launch_contract<e5m2_t, cuda_backend_t>(nk_dots_packed_e5m2_blackwellrtx)",
                    test_dots_launch_contract<e5m2_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_packed_e5m2_blackwellrtx,
                    nk_dots_symmetric_e5m2_blackwellrtx);
    run_kernel_case("nk_dots_symmetric_e5m2_blackwellrtx", test_dots_symmetric<e5m2_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_symmetric_e5m2_blackwellrtx);
    run_kernel_case("nk_dots_packed_e4m3_blackwellrtx", test_dots_packed<e4m3_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx,
                    nk_dots_packed_e4m3_blackwellrtx);
    run_kernel_case("nk_dots_pack_e4m3_blackwellrtx", test_dots_pack_layout<e4m3_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_packed_shape_e4m3_blackwellrtx,
                    nk_dots_pack_e4m3_blackwellrtx);
    run_kernel_case("test_dots_launch_contract<e4m3_t, cuda_backend_t>(nk_dots_packed_e4m3_blackwellrtx)",
                    test_dots_launch_contract<e4m3_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_packed_e4m3_blackwellrtx,
                    nk_dots_symmetric_e4m3_blackwellrtx);
    run_kernel_case("nk_dots_symmetric_e4m3_blackwellrtx", test_dots_symmetric<e4m3_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_symmetric_e4m3_blackwellrtx);
    run_kernel_case("nk_dots_packed_e3m2_blackwellrtx", test_dots_packed<e3m2_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx,
                    nk_dots_packed_e3m2_blackwellrtx);
    run_kernel_case("nk_dots_pack_e3m2_blackwellrtx", test_dots_pack_layout<e3m2_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_packed_shape_e3m2_blackwellrtx,
                    nk_dots_pack_e3m2_blackwellrtx);
    run_kernel_case("test_dots_launch_contract<e3m2_t, cuda_backend_t>(nk_dots_packed_e3m2_blackwellrtx)",
                    test_dots_launch_contract<e3m2_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_packed_e3m2_blackwellrtx,
                    nk_dots_symmetric_e3m2_blackwellrtx);
    run_kernel_case("nk_dots_symmetric_e3m2_blackwellrtx", test_dots_symmetric<e3m2_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_symmetric_e3m2_blackwellrtx);
    run_kernel_case("nk_dots_packed_e2m3_blackwellrtx", test_dots_packed<e2m3_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx,
                    nk_dots_packed_e2m3_blackwellrtx);
    run_kernel_case("nk_dots_pack_e2m3_blackwellrtx", test_dots_pack_layout<e2m3_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_packed_shape_e2m3_blackwellrtx,
                    nk_dots_pack_e2m3_blackwellrtx);
    run_kernel_case("test_dots_launch_contract<e2m3_t, cuda_backend_t>(nk_dots_packed_e2m3_blackwellrtx)",
                    test_dots_launch_contract<e2m3_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_packed_e2m3_blackwellrtx,
                    nk_dots_symmetric_e2m3_blackwellrtx);
    run_kernel_case("nk_dots_symmetric_e2m3_blackwellrtx", test_dots_symmetric<e2m3_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_symmetric_e2m3_blackwellrtx);
    run_kernel_case("nk_dots_packed_e2m1_blackwellrtx", test_dots_packed<e2m1x2_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx,
                    nk_dots_packed_e2m1_blackwellrtx);
    run_kernel_case("nk_dots_pack_e2m1_blackwellrtx", test_dots_pack_layout<e2m1x2_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_packed_shape_e2m1_blackwellrtx,
                    nk_dots_pack_e2m1_blackwellrtx);
    run_kernel_case("test_dots_launch_contract<e2m1x2_t, cuda_backend_t>(nk_dots_packed_e2m1_blackwellrtx)",
                    test_dots_launch_contract<e2m1x2_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_packed_e2m1_blackwellrtx,
                    nk_dots_symmetric_e2m1_blackwellrtx);
    run_kernel_case("nk_dots_symmetric_e2m1_blackwellrtx", test_dots_symmetric<e2m1x2_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_symmetric_e2m1_blackwellrtx);
    run_kernel_case("nk_angulars_packed_e5m2_blackwellrtx", test_angulars_packed<e5m2_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx,
                    nk_angulars_packed_e5m2_blackwellrtx);
    run_kernel_case("nk_angulars_symmetric_e5m2_blackwellrtx", test_angulars_symmetric<e5m2_t, cuda_backend_t>,
                    blackwellrtx, nk_angulars_symmetric_e5m2_blackwellrtx);
    run_kernel_case("nk_euclideans_packed_e5m2_blackwellrtx", test_euclideans_packed<e5m2_t, cuda_backend_t>,
                    blackwellrtx, nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx,
                    nk_euclideans_packed_e5m2_blackwellrtx);
    run_kernel_case("nk_euclideans_symmetric_e5m2_blackwellrtx", test_euclideans_symmetric<e5m2_t, cuda_backend_t>,
                    blackwellrtx, nk_euclideans_symmetric_e5m2_blackwellrtx);
    run_kernel_case("nk_angulars_packed_e4m3_blackwellrtx", test_angulars_packed<e4m3_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx,
                    nk_angulars_packed_e4m3_blackwellrtx);
    run_kernel_case("nk_angulars_symmetric_e4m3_blackwellrtx", test_angulars_symmetric<e4m3_t, cuda_backend_t>,
                    blackwellrtx, nk_angulars_symmetric_e4m3_blackwellrtx);
    run_kernel_case("nk_euclideans_packed_e4m3_blackwellrtx", test_euclideans_packed<e4m3_t, cuda_backend_t>,
                    blackwellrtx, nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx,
                    nk_euclideans_packed_e4m3_blackwellrtx);
    run_kernel_case("nk_euclideans_symmetric_e4m3_blackwellrtx", test_euclideans_symmetric<e4m3_t, cuda_backend_t>,
                    blackwellrtx, nk_euclideans_symmetric_e4m3_blackwellrtx);
    run_kernel_case("nk_angulars_packed_e3m2_blackwellrtx", test_angulars_packed<e3m2_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx,
                    nk_angulars_packed_e3m2_blackwellrtx);
    run_kernel_case("nk_angulars_symmetric_e3m2_blackwellrtx", test_angulars_symmetric<e3m2_t, cuda_backend_t>,
                    blackwellrtx, nk_angulars_symmetric_e3m2_blackwellrtx);
    run_kernel_case("nk_euclideans_packed_e3m2_blackwellrtx", test_euclideans_packed<e3m2_t, cuda_backend_t>,
                    blackwellrtx, nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx,
                    nk_euclideans_packed_e3m2_blackwellrtx);
    run_kernel_case("nk_euclideans_symmetric_e3m2_blackwellrtx", test_euclideans_symmetric<e3m2_t, cuda_backend_t>,
                    blackwellrtx, nk_euclideans_symmetric_e3m2_blackwellrtx);
    run_kernel_case("nk_angulars_packed_e2m3_blackwellrtx", test_angulars_packed<e2m3_t, cuda_backend_t>, blackwellrtx,
                    nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx,
                    nk_angulars_packed_e2m3_blackwellrtx);
    run_kernel_case("nk_angulars_symmetric_e2m3_blackwellrtx", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
                    blackwellrtx, nk_angulars_symmetric_e2m3_blackwellrtx);
    run_kernel_case("nk_euclideans_packed_e2m3_blackwellrtx", test_euclideans_packed<e2m3_t, cuda_backend_t>,
                    blackwellrtx, nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx,
                    nk_euclideans_packed_e2m3_blackwellrtx);
    run_kernel_case("nk_euclideans_symmetric_e2m3_blackwellrtx", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
                    blackwellrtx, nk_euclideans_symmetric_e2m3_blackwellrtx);
    run_kernel_case("nk_angulars_packed_e2m1_blackwellrtx", test_angulars_packed<e2m1x2_t, cuda_backend_t>,
                    blackwellrtx, nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx,
                    nk_angulars_packed_e2m1_blackwellrtx);
    run_kernel_case("nk_angulars_symmetric_e2m1_blackwellrtx", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
                    blackwellrtx, nk_angulars_symmetric_e2m1_blackwellrtx);
    run_kernel_case("nk_euclideans_packed_e2m1_blackwellrtx", test_euclideans_packed<e2m1x2_t, cuda_backend_t>,
                    blackwellrtx, nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx,
                    nk_euclideans_packed_e2m1_blackwellrtx);
    run_kernel_case("nk_euclideans_symmetric_e2m1_blackwellrtx", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
                    blackwellrtx, nk_euclideans_symmetric_e2m1_blackwellrtx);
    run_kernel_case("nk_attention_packed_e4m3_blackwellrtx",
                    test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>, blackwellrtx,
                    nk_attention_pack_size_e4m3_blackwellrtx, nk_attention_pack_e4m3_blackwellrtx,
                    nk_attention_packed_e4m3_blackwellrtx);
#endif // NUMKONG_TARGET_BLACKWELLRTX

#if NUMKONG_TARGET_BLACKWELLULTRA
    auto const blackwellultra = nk_cap_blackwellultra_k;
    run_kernel_case("nk_attention_packed_bf16_blackwellultra",
                    test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>, blackwellultra,
                    nk_attention_pack_size_bf16_blackwellultra, nk_attention_pack_bf16_blackwellultra,
                    nk_attention_packed_bf16_blackwellultra);
    run_kernel_case("nk_attention_packed_f16_blackwellultra",
                    test_attention_packed<f16_t, cuda_backend_t, attention_weights_t::bits_11_k>, blackwellultra,
                    nk_attention_pack_size_f16_blackwellultra, nk_attention_pack_f16_blackwellultra,
                    nk_attention_packed_f16_blackwellultra);
    run_kernel_case("nk_attention_packed_e4m3_blackwellultra",
                    test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>, blackwellultra,
                    nk_attention_pack_size_e4m3_blackwellultra, nk_attention_pack_e4m3_blackwellultra,
                    nk_attention_packed_e4m3_blackwellultra);
    run_kernel_case("nk_attention_packed_i8_blackwellultra",
                    test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>, blackwellultra,
                    nk_attention_pack_size_i8_blackwellultra, nk_attention_pack_i8_blackwellultra,
                    nk_attention_packed_i8_blackwellultra);
    run_kernel_case("nk_attention_packed_gradients_bf16_blackwellultra",
                    test_attention_packed_gradients<bf16_t, cuda_backend_t>, blackwellultra,
                    nk_attention_pack_size_bf16_blackwellultra, nk_attention_pack_bf16_blackwellultra,
                    nk_attention_packed_gradients_bf16_blackwellultra);
#endif // NUMKONG_TARGET_BLACKWELLULTRA

    run_cross_dispatch_cases<cuda_backend_t>();
}

#else  // !NUMKONG_ARCH_CUDA_
void run_cross_cuda_cases() {}
#endif // NUMKONG_ARCH_CUDA_

} // namespace ashvardanian::numkong::test
