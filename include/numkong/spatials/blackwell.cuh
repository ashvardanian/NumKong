/**
 *  @file include/numkong/spatials/blackwell.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief SIMD-accelerated batched spatial distances for the NVIDIA compute capability 10.x family.
 *
 *  @sa include/numkong/spatials.h
 *  @sa include/numkong/dots/blackwell.cuh
 *
 *  The @c tcgen05 dots tile with a metric in its epilogue. While the tensor cores multiply a stage,
 *  the epilogue warps square the A rows they will drain, and for @c symmetric the B rows too, with
 *  the Ampere norm updates over the codes as loaded, before any Float6 widening; @c packed reads
 *  the column norms its pack stored. Output precision, zero-norm handling and the triangle
 *  @c symmetric writes follow the Ampere capability.
 */
#ifndef NUMKONG_SPATIALS_BLACKWELL_CUH
#define NUMKONG_SPATIALS_BLACKWELL_CUH

#if NUMKONG_TARGET_BLACKWELL

#include "numkong/dots/blackwell.cuh"
#include "numkong/spatials/ampere.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region BF16

nk_define_device_cross_tma_(angular, bf16, blackwell, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                            /*dimensions_per_value=*/1, nk_dots_bf16_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_bf16_norm_update_, /*norm_scale=*/1.0f)
nk_define_device_cross_tma_(euclidean, bf16, blackwell, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                            /*dimensions_per_value=*/1, nk_dots_bf16_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_bf16_norm_update_, /*norm_scale=*/1.0f)

#pragma endregion BF16

#pragma region F16

nk_define_device_cross_tma_(angular, f16, blackwell, f16, f16, f32, /*depth_simd_dimensions=*/8,
                            /*dimensions_per_value=*/1, nk_dots_f16_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_f16_norm_update_, /*norm_scale=*/1.0f)
nk_define_device_cross_tma_(euclidean, f16, blackwell, f16, f16, f32, /*depth_simd_dimensions=*/8,
                            /*dimensions_per_value=*/1, nk_dots_f16_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_f16_norm_update_, /*norm_scale=*/1.0f)

#pragma endregion F16

#pragma region E5M2

nk_define_device_cross_tma_(angular, e5m2, blackwell, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                            /*dimensions_per_value=*/1, nk_dots_e5m2_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e5m2_norm_update_ampere_,
                            /*norm_scale=*/1.0f)
nk_define_device_cross_tma_(euclidean, e5m2, blackwell, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                            /*dimensions_per_value=*/1, nk_dots_e5m2_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e5m2_norm_update_ampere_,
                            /*norm_scale=*/1.0f)

#pragma endregion E5M2

#pragma region E4M3

nk_define_device_cross_tma_(angular, e4m3, blackwell, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                            /*dimensions_per_value=*/1, nk_dots_e4m3_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e4m3_norm_update_ampere_,
                            /*norm_scale=*/65536.0f)
nk_define_device_cross_tma_(euclidean, e4m3, blackwell, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                            /*dimensions_per_value=*/1, nk_dots_e4m3_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e4m3_norm_update_ampere_,
                            /*norm_scale=*/65536.0f)

#pragma endregion E4M3

#pragma region E3M2

nk_define_device_cross_tma_(angular, e3m2, blackwell, e3m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                            /*dimensions_per_value=*/1, nk_dots_e5m2_mma_blackwell_, nk_f6x4_to_f8x4_ada_,
                            /*output_scale=*/16777216.0f, nk_cross_norm_f32_k, nk_e3m2_norm_update_ampere_,
                            /*norm_scale=*/16777216.0f)
nk_define_device_cross_tma_(euclidean, e3m2, blackwell, e3m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                            /*dimensions_per_value=*/1, nk_dots_e5m2_mma_blackwell_, nk_f6x4_to_f8x4_ada_,
                            /*output_scale=*/16777216.0f, nk_cross_norm_f32_k, nk_e3m2_norm_update_ampere_,
                            /*norm_scale=*/16777216.0f)

#pragma endregion E3M2

#pragma region E2M3

nk_define_device_cross_tma_(angular, e2m3, blackwell, e2m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                            /*dimensions_per_value=*/1, nk_dots_e4m3_mma_blackwell_, nk_f6x4_to_f8x4_ada_,
                            /*output_scale=*/4096.0f, nk_cross_norm_f32_k, nk_e2m3_norm_update_ampere_,
                            /*norm_scale=*/0.015625f)
nk_define_device_cross_tma_(euclidean, e2m3, blackwell, e2m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                            /*dimensions_per_value=*/1, nk_dots_e4m3_mma_blackwell_, nk_f6x4_to_f8x4_ada_,
                            /*output_scale=*/4096.0f, nk_cross_norm_f32_k, nk_e2m3_norm_update_ampere_,
                            /*norm_scale=*/0.015625f)

#pragma endregion E2M3

#pragma region E2M1

nk_define_device_cross_tma_(angular, e2m1, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                            /*dimensions_per_value=*/2, nk_dots_e2m1_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e2m1_norm_update_ampere_,
                            /*norm_scale=*/0.25f)
nk_define_device_cross_tma_(euclidean, e2m1, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                            /*dimensions_per_value=*/2, nk_dots_e2m1_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e2m1_norm_update_ampere_,
                            /*norm_scale=*/0.25f)

#pragma endregion E2M1

#pragma region Block Scaled Floats

nk_define_device_cross_tma_(angular, nvfp4, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                            /*dimensions_per_value=*/2, nk_dots_nvfp4_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e2m1_norm_update_ampere_,
                            /*norm_scale=*/1.0f)
nk_define_device_cross_tma_(euclidean, nvfp4, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                            /*dimensions_per_value=*/2, nk_dots_nvfp4_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e2m1_norm_update_ampere_,
                            /*norm_scale=*/1.0f)
nk_define_device_cross_tma_(angular, mxfp4, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                            /*dimensions_per_value=*/2, nk_dots_mxfp4_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e2m1_norm_update_ampere_,
                            /*norm_scale=*/1.0f)
nk_define_device_cross_tma_(euclidean, mxfp4, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                            /*dimensions_per_value=*/2, nk_dots_mxfp4_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e2m1_norm_update_ampere_,
                            /*norm_scale=*/1.0f)
nk_define_device_cross_tma_(angular, mxfp8e4m3, blackwell, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                            /*dimensions_per_value=*/1, nk_dots_mxfp8e4m3_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e4m3_norm_update_ampere_,
                            /*norm_scale=*/1.0f)
nk_define_device_cross_tma_(euclidean, mxfp8e4m3, blackwell, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                            /*dimensions_per_value=*/1, nk_dots_mxfp8e4m3_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e4m3_norm_update_ampere_,
                            /*norm_scale=*/1.0f)
nk_define_device_cross_tma_(angular, mxfp8e5m2, blackwell, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                            /*dimensions_per_value=*/1, nk_dots_mxfp8e5m2_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e5m2_norm_update_ampere_,
                            /*norm_scale=*/1.0f)
nk_define_device_cross_tma_(euclidean, mxfp8e5m2, blackwell, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                            /*dimensions_per_value=*/1, nk_dots_mxfp8e5m2_mma_blackwell_, /*widen_fn=*/NUMKONG_NULL,
                            /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e5m2_norm_update_ampere_,
                            /*norm_scale=*/1.0f)

#pragma endregion Block Scaled Floats

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_BLACKWELL
#endif // NUMKONG_SPATIALS_BLACKWELL_CUH
