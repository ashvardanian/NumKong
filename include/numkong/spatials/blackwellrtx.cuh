/**
 *  @file include/numkong/spatials/blackwellrtx.cuh
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief SIMD-accelerated batched spatial distances for the NVIDIA compute capability 12.x family.
 *
 *  @sa include/numkong/spatials.h
 *  @sa include/numkong/dots/blackwellrtx.cuh
 *
 *  The native Float8, Float6 and Float4 products with the Ampere norm updates, whose widenings read
 *  a Float6 code from its low 6 bits exactly as the MMA does, so both see the same values.
 */
#ifndef NUMKONG_SPATIALS_BLACKWELLRTX_CUH
#define NUMKONG_SPATIALS_BLACKWELLRTX_CUH

#if NUMKONG_ARCH_CUDA_
#if NUMKONG_TARGET_BLACKWELLRTX

#include "numkong/dots/blackwellrtx.cuh"
#include "numkong/spatials/ampere.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region E5M2

nk_define_cross_cuda_(angular, e5m2, blackwellrtx, e5m2_blackwellrtx, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, e5m2, blackwellrtx, e5m2_blackwellrtx, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_cuda_(angular, e4m3, blackwellrtx, e4m3_blackwellrtx, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, e4m3, blackwellrtx, e4m3_blackwellrtx, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_cuda_(angular, e3m2, blackwellrtx, e3m2_blackwellrtx, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, e3m2, blackwellrtx, e3m2_blackwellrtx, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_cuda_(angular, e2m3, blackwellrtx, e2m3_blackwellrtx, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, e2m3, blackwellrtx, e2m3_blackwellrtx, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_cuda_(angular, e2m1, blackwellrtx, e2m1_blackwellrtx, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_cuda_(euclidean, e2m1, blackwellrtx, e2m1_blackwellrtx, e2m1x2, e2m1x2, f32,
                      /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion E2M1

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_BLACKWELLRTX
#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_SPATIALS_BLACKWELLRTX_CUH
