/**
 *  @file include/numkong/spatials/ampere.cuh
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief SIMD-accelerated batched spatial distances for NVIDIA Ampere and newer.
 *
 *  @sa include/numkong/spatials.h
 *  @sa include/numkong/dots/ampere.cuh
 *
 *  The dots tile with a metric in its epilogue: squared norms of A, and for @c symmetric of the
 *  column vectors too, accumulate from the staged slabs while the products do, and @c packed reads
 *  the column norms its pack stored. Integer codes square exactly through @c dp4a, E2M3 and E2M1
 *  included; the other narrow floats sum each slab apart in F32 before adding it to the running F32
 *  sum. F32 and F64 use @c cuda. Output precision, zero-norm handling and the triangle @c symmetric
 *  writes follow the serial backends: F32 outputs, with I8 and I4 norms read as I32.
 */
#ifndef NUMKONG_SPATIALS_AMPERE_CUH
#define NUMKONG_SPATIALS_AMPERE_CUH

#if NUMKONG_ARCH_CUDA_
#if NUMKONG_ARCH_CUDA_AMPERE_

#include "numkong/dots/ampere.cuh"
#include "numkong/spatials/cuda.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

/*  Later generations include this header and emit only their own kernels. */
#if NUMKONG_TARGET_AMPERE

#pragma region BF16

nk_define_cross_cuda_(angular, bf16, ampere, bf16_ampere, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, bf16, ampere, bf16_ampere, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)

#pragma endregion BF16

#pragma region F16

nk_define_cross_cuda_(angular, f16, ampere, f16_ampere, f16, f16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, f16, ampere, f16_ampere, f16, f16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)

#pragma endregion F16

#pragma region E5M2

nk_define_cross_cuda_(angular, e5m2, ampere, e5m2_ampere, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, e5m2, ampere, e5m2_ampere, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_cuda_(angular, e4m3, ampere, e4m3_ampere, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, e4m3, ampere, e4m3_ampere, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_cuda_(angular, e3m2, ampere, e3m2_ampere, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, e3m2, ampere, e3m2_ampere, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_packed_cuda_(angular, e2m3, ampere, e2m3_ampere, e2m3, i8, f32, /*depth_simd_dimensions=*/16,
                             /*dimensions_per_value=*/1)
nk_define_cross_symmetric_cuda_(angular, e2m3, ampere, e2m3_symmetric_ampere, e2m3, i8, f32,
                                /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_cuda_(euclidean, e2m3, ampere, e2m3_ampere, e2m3, i8, f32, /*depth_simd_dimensions=*/16,
                             /*dimensions_per_value=*/1)
nk_define_cross_symmetric_cuda_(euclidean, e2m3, ampere, e2m3_symmetric_ampere, e2m3, i8, f32,
                                /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_cuda_(angular, e2m1, ampere, e2m1_ampere, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_cuda_(euclidean, e2m1, ampere, e2m1_ampere, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion E2M1

#pragma region I8

nk_define_cross_cuda_(angular, i8, ampere, i8_ampere, i8, i8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, i8, ampere, i8_ampere, i8, i8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion I8

#pragma region I4

nk_define_cross_cuda_(angular, i4, ampere, i4_ampere, i4x2, i4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_cuda_(euclidean, i4, ampere, i4_ampere, i4x2, i4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion I4

#pragma region U8

nk_define_cross_cuda_(angular, u8, ampere, u8_ampere, u8, u8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, u8, ampere, u8_ampere, u8, u8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion U8

#pragma region U4

nk_define_cross_cuda_(angular, u4, ampere, u4_ampere, u4x2, u4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_cuda_(euclidean, u4, ampere, u4_ampere, u4x2, u4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion U4

#endif // NUMKONG_TARGET_AMPERE

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_AMPERE_
#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_SPATIALS_AMPERE_CUH
