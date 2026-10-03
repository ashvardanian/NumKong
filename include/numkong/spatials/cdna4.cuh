/**
 *  @file include/numkong/spatials/cdna4.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief SIMD-accelerated batched spatial distances for AMD Instinct MI350, gfx950.
 *
 *  @sa include/numkong/spatials.h
 *  @sa include/numkong/dots/cdna4.cuh
 *
 *  The CDNA4 dots tile with a metric in its epilogue: squared norms of A, and for @c symmetric of
 *  the column vectors too, accumulate from the staged chunks while the products do, and @c packed
 *  reads the column norms its pack stored. Integer codes square exactly through @c v_dot4, E2M3 and
 *  E2M1 included; the other narrow floats sum in F32. F32 and F64 use @c rocm. Metric precision,
 *  output precision, zero-norm handling and the triangle @c symmetric writes follow the serial
 *  backends: F32 metrics and outputs, with I8 and I4 norms read as I32.
 */
#ifndef NUMKONG_SPATIALS_CDNA4_CUH
#define NUMKONG_SPATIALS_CDNA4_CUH

#if NUMKONG_TARGET_CDNA4

#include "numkong/dots/cdna4.cuh"
#include "numkong/spatials/simt.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region BF16

nk_define_cross_simt_(angular, bf16, cdna4, cdna4, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1, nk_dots_bf16_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_bf16_norm_update_, /*norm_scale=*/1.0f)
nk_define_cross_simt_(euclidean, bf16, cdna4, cdna4, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1, nk_dots_bf16_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_bf16_norm_update_, /*norm_scale=*/1.0f)

#pragma endregion BF16

#pragma region F16

nk_define_cross_simt_(angular, f16, cdna4, cdna4, f16, f16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1, nk_dots_f16_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_f16_norm_update_, /*norm_scale=*/1.0f)
nk_define_cross_simt_(euclidean, f16, cdna4, cdna4, f16, f16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1, nk_dots_f16_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_f16_norm_update_, /*norm_scale=*/1.0f)

#pragma endregion F16

#pragma region E5M2

nk_define_cross_simt_(angular, e5m2, cdna4, cdna4, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e5m2_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e5m2_norm_update_cdna4_, /*norm_scale=*/1.0f)
nk_define_cross_simt_(euclidean, e5m2, cdna4, cdna4, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e5m2_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e5m2_norm_update_cdna4_, /*norm_scale=*/1.0f)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_simt_(angular, e4m3, cdna4, cdna4, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e4m3_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e4m3_norm_update_cdna4_,
                      /*norm_scale=*/65536.0f)
nk_define_cross_simt_(euclidean, e4m3, cdna4, cdna4, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e4m3_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e4m3_norm_update_cdna4_,
                      /*norm_scale=*/65536.0f)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_simt_(angular, e3m2, cdna4, cdna4, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e3m2_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e3m2_norm_update_cdna4_,
                      /*norm_scale=*/16777216.0f)
nk_define_cross_simt_(euclidean, e3m2, cdna4, cdna4, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e3m2_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e3m2_norm_update_cdna4_,
                      /*norm_scale=*/16777216.0f)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_simt_(angular, e2m3, cdna4, cdna4, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e2m3_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e2m3_norm_update_cdna4_,
                      /*norm_scale=*/0.015625f)
nk_define_cross_simt_(euclidean, e2m3, cdna4, cdna4, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e2m3_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e2m3_norm_update_cdna4_,
                      /*norm_scale=*/0.015625f)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_simt_(angular, e2m1, cdna4, cdna4, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_e2m1_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e2m1_norm_update_cdna4_, /*norm_scale=*/0.25f)
nk_define_cross_simt_(euclidean, e2m1, cdna4, cdna4, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_e2m1_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e2m1_norm_update_cdna4_, /*norm_scale=*/0.25f)

#pragma endregion E2M1

#pragma region I8

nk_define_cross_simt_(angular, i8, cdna4, cdna4, i8, i8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_i8_multiply_cdna4_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_i32_k, nk_i8_norm_update_cdna4_, /*norm_scale=*/1.0f)
nk_define_cross_simt_(euclidean, i8, cdna4, cdna4, i8, i8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_i8_multiply_cdna4_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_i32_k, nk_i8_norm_update_cdna4_, /*norm_scale=*/1.0f)

#pragma endregion I8

#pragma region I4

nk_define_cross_simt_(angular, i4, cdna4, cdna4, i4x2, i4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_i4_multiply_cdna4_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_i32_k, nk_i4_norm_update_cdna4_, /*norm_scale=*/1.0f)
nk_define_cross_simt_(euclidean, i4, cdna4, cdna4, i4x2, i4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_i4_multiply_cdna4_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_i32_k, nk_i4_norm_update_cdna4_, /*norm_scale=*/1.0f)

#pragma endregion I4

#pragma region U8

nk_define_cross_simt_(angular, u8, cdna4, cdna4, u8, u8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_u8_multiply_cdna4_, nk_cross_epilogue_offset_u32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_u32_k, nk_u8_norm_update_cdna4_, /*norm_scale=*/1.0f)
nk_define_cross_simt_(euclidean, u8, cdna4, cdna4, u8, u8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_u8_multiply_cdna4_, nk_cross_epilogue_offset_u32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_u32_k, nk_u8_norm_update_cdna4_, /*norm_scale=*/1.0f)

#pragma endregion U8

#pragma region U4

nk_define_cross_simt_(angular, u4, cdna4, cdna4, u4x2, u4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_u4_multiply_cdna4_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_u32_k, nk_u4_norm_update_cdna4_, /*norm_scale=*/1.0f)
nk_define_cross_simt_(euclidean, u4, cdna4, cdna4, u4x2, u4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_u4_multiply_cdna4_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_u32_k, nk_u4_norm_update_cdna4_, /*norm_scale=*/1.0f)

#pragma endregion U4

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_CDNA4
#endif // NUMKONG_SPATIALS_CDNA4_CUH
