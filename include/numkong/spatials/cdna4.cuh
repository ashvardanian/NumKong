/**
 *  @file include/numkong/spatials/cdna4.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief SIMD-accelerated batched spatial distances for AMD Instinct MI350, gfx950.
 *
 *  @sa include/numkong/spatials.h
 *  @sa include/numkong/dots/cdna4.cuh
 *
 *  The CDNA3 dots tile with a metric in its epilogue: squared norms of A, and for @c symmetric of
 *  the column vectors too, accumulate from the staged chunks while the products do, and @c packed
 *  reads the column norms its pack stored. Integer codes square exactly through @c v_dot4, E2M3 and
 *  E2M1 included; the other narrow floats sum in F32. F32 and F64 use @c rocm. Metric precision,
 *  output precision, zero-norm handling and the triangle @c symmetric writes follow the serial
 *  backends: F32 metrics and outputs, with I8 and I4 norms read as I32.
 */
#ifndef NUMKONG_SPATIALS_CDNA4_CUH
#define NUMKONG_SPATIALS_CDNA4_CUH

#if NUMKONG_ARCH_ROCM_
#if NUMKONG_ARCH_ROCM_CDNA4_

#include "numkong/dots/cdna4.cuh"
#include "numkong/spatials/rocm.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#if NUMKONG_TARGET_CDNA4
#pragma region BF16

nk_define_cross_rocm_(angular, bf16, cdna4, cdna3, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)
nk_define_cross_rocm_(euclidean, bf16, cdna4, cdna3, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)

#pragma endregion BF16

#pragma region F16

nk_define_cross_rocm_(angular, f16, cdna4, cdna3, f16, f16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)
nk_define_cross_rocm_(euclidean, f16, cdna4, cdna3, f16, f16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)

#pragma endregion F16

#pragma region E5M2

nk_define_cross_rocm_(angular, e5m2, cdna4, cdna3, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_rocm_(euclidean, e5m2, cdna4, cdna3, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_rocm_(angular, e4m3, cdna4, cdna3, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_rocm_(euclidean, e4m3, cdna4, cdna3, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_rocm_(angular, e3m2, cdna4, cdna3, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_rocm_(euclidean, e3m2, cdna4, cdna3, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_rocm_(angular, e2m3, cdna4, cdna3, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_rocm_(euclidean, e2m3, cdna4, cdna3, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_rocm_(angular, e2m1, cdna4, cdna3, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_rocm_(euclidean, e2m1, cdna4, cdna3, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion E2M1

#pragma region I8

nk_define_cross_rocm_(angular, i8, cdna4, cdna3, i8, i8, f32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(euclidean, i8, cdna4, cdna3, i8, i8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion I8

#pragma region I4

nk_define_cross_rocm_(angular, i4, cdna4, cdna3, i4x2, i4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_rocm_(euclidean, i4, cdna4, cdna3, i4x2, i4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion I4

#pragma region U8

nk_define_cross_rocm_(angular, u8, cdna4, cdna3, u8, u8, f32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(euclidean, u8, cdna4, cdna3, u8, u8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion U8

#pragma region U4

nk_define_cross_rocm_(angular, u4, cdna4, cdna3, u4x2, u4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_rocm_(euclidean, u4, cdna4, cdna3, u4x2, u4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion U4
#endif // NUMKONG_TARGET_CDNA4

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_CDNA4_
#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_SPATIALS_CDNA4_CUH
