/**
 *  @file include/numkong/spatials/cdna5.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief SIMD-accelerated batched spatial distances for AMD Instinct MI400, gfx1250 and gfx1251.
 *
 *  @sa include/numkong/spatials.h
 *  @sa include/numkong/dots/cdna5.cuh
 *
 *  The CDNA5 dots tile with the CDNA4 norm updates and metric epilogue, its own for I8 and I4:
 *  squared norms accumulate from the staged chunks while the products do, and @c packed reads the
 *  column norms its pack stored. F32 and F64 stay on the @c rocm capability.
 */
#ifndef NUMKONG_SPATIALS_CDNA5_CUH
#define NUMKONG_SPATIALS_CDNA5_CUH

#if NUMKONG_ARCH_ROCM_
#if NUMKONG_TARGET_CDNA5

#include "numkong/dots/cdna5.cuh"
#include "numkong/spatials/rocm.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region BF16

nk_define_cross_rocm_(angular, bf16, cdna5, cdna5, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)
nk_define_cross_rocm_(euclidean, bf16, cdna5, cdna5, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)

#pragma endregion BF16

#pragma region F16

nk_define_cross_rocm_(angular, f16, cdna5, cdna5, f16, f16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)
nk_define_cross_rocm_(euclidean, f16, cdna5, cdna5, f16, f16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)

#pragma endregion F16

#pragma region E5M2

nk_define_cross_rocm_(angular, e5m2, cdna5, cdna5, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_rocm_(euclidean, e5m2, cdna5, cdna5, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_rocm_(angular, e4m3, cdna5, cdna5, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_rocm_(euclidean, e4m3, cdna5, cdna5, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_rocm_(angular, e3m2, cdna5, cdna5, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_rocm_(euclidean, e3m2, cdna5, cdna5, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_rocm_(angular, e2m3, cdna5, cdna5, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_rocm_(euclidean, e2m3, cdna5, cdna5, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_rocm_(angular, e2m1, cdna5, cdna5, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_rocm_(euclidean, e2m1, cdna5, cdna5, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion E2M1

#pragma region I8

nk_define_cross_rocm_(angular, i8, cdna5, cdna5, i8, i8, f32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(euclidean, i8, cdna5, cdna5, i8, i8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion I8

#pragma region I4

nk_define_cross_rocm_(angular, i4, cdna5, cdna5, i4x2, i4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_rocm_(euclidean, i4, cdna5, cdna5, i4x2, i4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion I4

#pragma region U8

nk_define_cross_rocm_(angular, u8, cdna5, cdna5, u8, u8, f32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(euclidean, u8, cdna5, cdna5, u8, u8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion U8

#pragma region U4

nk_define_cross_rocm_(angular, u4, cdna5, cdna5, u4x2, u4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_rocm_(euclidean, u4, cdna5, cdna5, u4x2, u4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion U4

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_CDNA5
#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_SPATIALS_CDNA5_CUH
