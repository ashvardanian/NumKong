/**
 *  @file include/numkong/spatials/cdna3.cuh
 *  @author Ash Vardanian
 *  @date October 7, 2026
 *  @brief SIMD-accelerated batched spatial distances for AMD Instinct MI300, gfx942.
 *
 *  @sa include/numkong/spatials.h
 *  @sa include/numkong/dots/cdna3.cuh
 *
 *  The CDNA3 dots tiles with a metric in their epilogue: squared norms of A, and for @c symmetric
 *  of the column vectors too, accumulate while the products do, and @c packed reads the column
 *  norms its pack stored. Integer codes square exactly through the byte dots, BF16 and F16 sum in
 *  F32, and the Float8, Float6 and Float4 codes fold as F16 pairs on the B32 tile. F32 and F64 use
 *  @c rocm. Metric precision, output precision, zero-norm handling and the triangle @c symmetric
 *  writes follow the serial backends: F32 metrics and outputs, with I8 and I4 norms read as I32.
 */
#ifndef NUMKONG_SPATIALS_CDNA3_CUH
#define NUMKONG_SPATIALS_CDNA3_CUH

#if NUMKONG_ARCH_ROCM_
#if NUMKONG_ARCH_ROCM_CDNA3_

#include "numkong/dots/cdna3.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#if NUMKONG_TARGET_CDNA3
#pragma region BF16

nk_define_cross_rocm_(angular, bf16, cdna3, cdna3, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1, nk_dots_bf16_multiply_cdna3_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_bf16_norm_update_, /*norm_scale=*/1.0f)
nk_define_cross_rocm_(euclidean, bf16, cdna3, cdna3, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1, nk_dots_bf16_multiply_cdna3_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_bf16_norm_update_, /*norm_scale=*/1.0f)

#pragma endregion BF16

#pragma region F16

nk_define_cross_rocm_(angular, f16, cdna3, cdna3, f16, f16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1, nk_dots_f16_multiply_cdna3_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_f16_norm_update_, /*norm_scale=*/1.0f)
nk_define_cross_rocm_(euclidean, f16, cdna3, cdna3, f16, f16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1, nk_dots_f16_multiply_cdna3_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_f16_norm_update_, /*norm_scale=*/1.0f)

#pragma endregion F16

#pragma region Float8, Float6 and Float4

nk_define_cross_rocm_(angular, e5m2, cdna3, b32_cdna3, e5m2, e5m2, f32, 16, 1, nk_e5m2_k)
nk_define_cross_rocm_(euclidean, e5m2, cdna3, b32_cdna3, e5m2, e5m2, f32, 16, 1, nk_e5m2_k)
nk_define_cross_rocm_(angular, e4m3, cdna3, b32_cdna3, e4m3, e4m3, f32, 16, 1, nk_e4m3_k)
nk_define_cross_rocm_(euclidean, e4m3, cdna3, b32_cdna3, e4m3, e4m3, f32, 16, 1, nk_e4m3_k)
nk_define_cross_rocm_(angular, e3m2, cdna3, b32_cdna3, e3m2, e3m2, f32, 16, 1, nk_e3m2_k)
nk_define_cross_rocm_(euclidean, e3m2, cdna3, b32_cdna3, e3m2, e3m2, f32, 16, 1, nk_e3m2_k)
nk_define_cross_rocm_(angular, e2m3, cdna3, b32_cdna3, e2m3, e2m3, f32, 16, 1, nk_e2m3_k)
nk_define_cross_rocm_(euclidean, e2m3, cdna3, b32_cdna3, e2m3, e2m3, f32, 16, 1, nk_e2m3_k)
nk_define_cross_rocm_(angular, e2m1, cdna3, b32_cdna3, e2m1x2, e2m1x2, f32, 32, 2, nk_e2m1_k)
nk_define_cross_rocm_(euclidean, e2m1, cdna3, b32_cdna3, e2m1x2, e2m1x2, f32, 32, 2, nk_e2m1_k)

#pragma endregion Float8, Float6 and Float4

#pragma region I8

nk_define_cross_rocm_(angular, i8, cdna3, cdna3, i8, i8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_i8_multiply_cdna3_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_i32_k, nk_i8_norm_update_cdna3_, /*norm_scale=*/1.0f)
nk_define_cross_rocm_(euclidean, i8, cdna3, cdna3, i8, i8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_i8_multiply_cdna3_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_i32_k, nk_i8_norm_update_cdna3_, /*norm_scale=*/1.0f)

#pragma endregion I8

#pragma region I4

nk_define_cross_rocm_(angular, i4, cdna3, cdna3, i4x2, i4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_i4_multiply_cdna3_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_i32_k, nk_i4_norm_update_cdna3_, /*norm_scale=*/1.0f)
nk_define_cross_rocm_(euclidean, i4, cdna3, cdna3, i4x2, i4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_i4_multiply_cdna3_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_i32_k, nk_i4_norm_update_cdna3_, /*norm_scale=*/1.0f)

#pragma endregion I4

#pragma region U8

nk_define_cross_rocm_(angular, u8, cdna3, cdna3, u8, u8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_u8_multiply_cdna3_, nk_cross_epilogue_offset_u32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_u32_k, nk_u8_norm_update_cdna3_, /*norm_scale=*/1.0f)
nk_define_cross_rocm_(euclidean, u8, cdna3, cdna3, u8, u8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_u8_multiply_cdna3_, nk_cross_epilogue_offset_u32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_u32_k, nk_u8_norm_update_cdna3_, /*norm_scale=*/1.0f)

#pragma endregion U8

#pragma region U4

nk_define_cross_rocm_(angular, u4, cdna3, cdna3, u4x2, u4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_u4_multiply_cdna3_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_u32_k, nk_u4_norm_update_cdna3_, /*norm_scale=*/1.0f)
nk_define_cross_rocm_(euclidean, u4, cdna3, cdna3, u4x2, u4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_u4_multiply_cdna3_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_u32_k, nk_u4_norm_update_cdna3_, /*norm_scale=*/1.0f)

#pragma endregion U4
#endif // NUMKONG_TARGET_CDNA3

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_CDNA3_
#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_SPATIALS_CDNA3_CUH
