/**
 *  @file include/numkong/spatials/cdna5.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief SIMD-accelerated batched spatial distances for AMD Instinct MI400, gfx1250 and gfx1251.
 *
 *  @sa include/numkong/spatials.h
 *  @sa include/numkong/dots/cdna5.cuh
 *
 *  The CDNA5 dots tile with the CDNA4 norm updates and metric epilogue: squared norms accumulate
 *  from the staged chunks while the products do, and @c packed reads the column norms its pack
 *  stored. F32 and F64 stay on the @c rocm capability.
 */
#ifndef NUMKONG_SPATIALS_CDNA5_CUH
#define NUMKONG_SPATIALS_CDNA5_CUH

#if NUMKONG_TARGET_CDNA5

#include "numkong/dots/cdna5.cuh"
#include "numkong/spatials/simt.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region BF16

nk_define_device_cross_(angular, bf16, cdna5, cdna5, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                        /*dimensions_per_value=*/1, nk_dots_bf16_multiply_cdna5_, nk_cross_epilogue_f32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_bf16_norm_update_, /*norm_scale=*/1.0f)
nk_define_device_cross_(euclidean, bf16, cdna5, cdna5, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                        /*dimensions_per_value=*/1, nk_dots_bf16_multiply_cdna5_, nk_cross_epilogue_f32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_bf16_norm_update_, /*norm_scale=*/1.0f)

#pragma endregion BF16

#pragma region F16

nk_define_device_cross_(angular, f16, cdna5, cdna5, f16, f16, f32, /*depth_simd_dimensions=*/8,
                        /*dimensions_per_value=*/1, nk_dots_f16_multiply_cdna5_, nk_cross_epilogue_f32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_f16_norm_update_, /*norm_scale=*/1.0f)
nk_define_device_cross_(euclidean, f16, cdna5, cdna5, f16, f16, f32, /*depth_simd_dimensions=*/8,
                        /*dimensions_per_value=*/1, nk_dots_f16_multiply_cdna5_, nk_cross_epilogue_f32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_f16_norm_update_, /*norm_scale=*/1.0f)

#pragma endregion F16

#pragma region E5M2

nk_define_device_cross_(angular, e5m2, cdna5, cdna5, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                        /*dimensions_per_value=*/1, nk_dots_e5m2_multiply_cdna5_, nk_cross_epilogue_f32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e5m2_norm_update_cdna4_, /*norm_scale=*/1.0f)
nk_define_device_cross_(euclidean, e5m2, cdna5, cdna5, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                        /*dimensions_per_value=*/1, nk_dots_e5m2_multiply_cdna5_, nk_cross_epilogue_f32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e5m2_norm_update_cdna4_, /*norm_scale=*/1.0f)

#pragma endregion E5M2

#pragma region E4M3

nk_define_device_cross_(angular, e4m3, cdna5, cdna5, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                        /*dimensions_per_value=*/1, nk_dots_e4m3_multiply_cdna5_, nk_cross_epilogue_f32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e4m3_norm_update_cdna4_,
                        /*norm_scale=*/65536.0f)
nk_define_device_cross_(euclidean, e4m3, cdna5, cdna5, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                        /*dimensions_per_value=*/1, nk_dots_e4m3_multiply_cdna5_, nk_cross_epilogue_f32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e4m3_norm_update_cdna4_,
                        /*norm_scale=*/65536.0f)

#pragma endregion E4M3

#pragma region E3M2

nk_define_device_cross_(angular, e3m2, cdna5, cdna5, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                        /*dimensions_per_value=*/1, nk_dots_e3m2_multiply_cdna5_, nk_cross_epilogue_f32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e3m2_norm_update_cdna4_,
                        /*norm_scale=*/16777216.0f)
nk_define_device_cross_(euclidean, e3m2, cdna5, cdna5, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                        /*dimensions_per_value=*/1, nk_dots_e3m2_multiply_cdna5_, nk_cross_epilogue_f32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e3m2_norm_update_cdna4_,
                        /*norm_scale=*/16777216.0f)

#pragma endregion E3M2

#pragma region E2M3

nk_define_device_cross_(angular, e2m3, cdna5, cdna5, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                        /*dimensions_per_value=*/1, nk_dots_e2m3_multiply_cdna5_, nk_cross_epilogue_f32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e2m3_norm_update_cdna4_,
                        /*norm_scale=*/0.015625f)
nk_define_device_cross_(euclidean, e2m3, cdna5, cdna5, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                        /*dimensions_per_value=*/1, nk_dots_e2m3_multiply_cdna5_, nk_cross_epilogue_f32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e2m3_norm_update_cdna4_,
                        /*norm_scale=*/0.015625f)

#pragma endregion E2M3

#pragma region E2M1

nk_define_device_cross_(angular, e2m1, cdna5, cdna5, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                        /*dimensions_per_value=*/2, nk_dots_e2m1_multiply_cdna5_, nk_cross_epilogue_f32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e2m1_norm_update_cdna4_, /*norm_scale=*/0.25f)
nk_define_device_cross_(euclidean, e2m1, cdna5, cdna5, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                        /*dimensions_per_value=*/2, nk_dots_e2m1_multiply_cdna5_, nk_cross_epilogue_f32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e2m1_norm_update_cdna4_, /*norm_scale=*/0.25f)

#pragma endregion E2M1

#pragma region I8

nk_define_device_cross_(angular, i8, cdna5, cdna5, i8, i8, f32, /*depth_simd_dimensions=*/16,
                        /*dimensions_per_value=*/1, nk_dots_i8_multiply_cdna5_, nk_cross_epilogue_i32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_i32_k, nk_i8_norm_update_cdna4_, /*norm_scale=*/1.0f)
nk_define_device_cross_(euclidean, i8, cdna5, cdna5, i8, i8, f32, /*depth_simd_dimensions=*/16,
                        /*dimensions_per_value=*/1, nk_dots_i8_multiply_cdna5_, nk_cross_epilogue_i32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_i32_k, nk_i8_norm_update_cdna4_, /*norm_scale=*/1.0f)

#pragma endregion I8

#pragma region I4

nk_define_device_cross_(angular, i4, cdna5, cdna5, i4x2, i4x2, f32, /*depth_simd_dimensions=*/32,
                        /*dimensions_per_value=*/2, nk_dots_i4_multiply_cdna5_, nk_cross_epilogue_i32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_i32_k, nk_i4_norm_update_cdna4_, /*norm_scale=*/1.0f)
nk_define_device_cross_(euclidean, i4, cdna5, cdna5, i4x2, i4x2, f32, /*depth_simd_dimensions=*/32,
                        /*dimensions_per_value=*/2, nk_dots_i4_multiply_cdna5_, nk_cross_epilogue_i32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_i32_k, nk_i4_norm_update_cdna4_, /*norm_scale=*/1.0f)

#pragma endregion I4

#pragma region U8

nk_define_device_cross_(angular, u8, cdna5, cdna5, u8, u8, f32, /*depth_simd_dimensions=*/16,
                        /*dimensions_per_value=*/1, nk_dots_u8_multiply_cdna5_, nk_cross_epilogue_i32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_u32_k, nk_u8_norm_update_cdna4_, /*norm_scale=*/1.0f)
nk_define_device_cross_(euclidean, u8, cdna5, cdna5, u8, u8, f32, /*depth_simd_dimensions=*/16,
                        /*dimensions_per_value=*/1, nk_dots_u8_multiply_cdna5_, nk_cross_epilogue_i32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_u32_k, nk_u8_norm_update_cdna4_, /*norm_scale=*/1.0f)

#pragma endregion U8

#pragma region U4

nk_define_device_cross_(angular, u4, cdna5, cdna5, u4x2, u4x2, f32, /*depth_simd_dimensions=*/32,
                        /*dimensions_per_value=*/2, nk_dots_u4_multiply_cdna5_, nk_cross_epilogue_i32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_u32_k, nk_u4_norm_update_cdna4_, /*norm_scale=*/1.0f)
nk_define_device_cross_(euclidean, u4, cdna5, cdna5, u4x2, u4x2, f32, /*depth_simd_dimensions=*/32,
                        /*dimensions_per_value=*/2, nk_dots_u4_multiply_cdna5_, nk_cross_epilogue_i32_k,
                        /*output_scale=*/1.0f, nk_cross_norm_u32_k, nk_u4_norm_update_cdna4_, /*norm_scale=*/1.0f)

#pragma endregion U4

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_CDNA5
#endif // NUMKONG_SPATIALS_CDNA5_CUH
