/**
 *  @file include/numkong/spatials/hopper.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief SIMD-accelerated batched spatial distances for NVIDIA Hopper, compute capability 9.0.
 *
 *  @sa include/numkong/spatials.h
 *  @sa include/numkong/dots/hopper.cuh
 *
 *  Warpgroup `wgmma.mma_async` over shared-memory descriptors, which only the "90a" code carries,
 *  so no other generation compiles these kernels. The dots tile with a metric in its epilogue:
 *  squared norms of A, and for @c symmetric of the column vectors too, accumulate from each staged
 *  slab through Ampere's norm updates before any widening, and @c packed reads the column norms
 *  its pack stored. Norm precision, zero-norm handling and the triangle @c symmetric writes follow
 *  the Ampere kernels.
 */
#ifndef NUMKONG_SPATIALS_HOPPER_CUH
#define NUMKONG_SPATIALS_HOPPER_CUH

#if NUMKONG_ARCH_CUDA_
#if NUMKONG_TARGET_HOPPER

#include "numkong/dots/hopper.cuh"
#include "numkong/spatials/ampere.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region BF16

nk_define_cross_cuda_(angular, bf16, hopper, bf16_hopper, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, bf16, hopper, bf16_hopper, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)

#pragma endregion BF16

#pragma region F16

nk_define_cross_cuda_(angular, f16, hopper, f16_hopper, f16, f16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, f16, hopper, f16_hopper, f16, f16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)

#pragma endregion F16

#pragma region E2M3

nk_define_cross_cuda_(angular, e2m3, hopper, e2m3_hopper, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, e2m3, hopper, e2m3_hopper, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_cuda_(angular, e2m1, hopper, e2m1_hopper, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_cuda_(euclidean, e2m1, hopper, e2m1_hopper, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion E2M1

#pragma region I8

nk_define_cross_cuda_(angular, i8, hopper, i8_hopper, i8, i8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, i8, hopper, i8_hopper, i8, i8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion I8

#pragma region I4

nk_define_cross_cuda_(angular, i4, hopper, i4_hopper, i4x2, i4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_cuda_(euclidean, i4, hopper, i4_hopper, i4x2, i4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion I4

#pragma region U8

nk_define_cross_cuda_(angular, u8, hopper, u8_hopper, u8, u8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, u8, hopper, u8_hopper, u8, u8, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion U8

#pragma region U4

nk_define_cross_cuda_(angular, u4, hopper, u4_hopper, u4x2, u4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_cuda_(euclidean, u4, hopper, u4_hopper, u4x2, u4x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion U4

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_HOPPER
#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_SPATIALS_HOPPER_CUH
