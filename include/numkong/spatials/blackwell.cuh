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

#if NUMKONG_ARCH_CUDA_
#if NUMKONG_ARCH_CUDA_BLACKWELL_

#include "numkong/dots/blackwell.cuh"
#include "numkong/spatials/ampere.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#if NUMKONG_TARGET_BLACKWELL
#pragma region BF16

nk_define_cross_tma_blackwell_(angular, bf16, blackwell, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*paired_dot=*/1)
nk_define_cross_tma_blackwell_(euclidean, bf16, blackwell, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*paired_dot=*/1)

#pragma endregion BF16

#pragma region F16

nk_define_cross_tma_blackwell_(angular, f16, blackwell, f16, f16, f32, /*depth_simd_dimensions=*/8,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*paired_dot=*/1)
nk_define_cross_tma_blackwell_(euclidean, f16, blackwell, f16, f16, f32, /*depth_simd_dimensions=*/8,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*paired_dot=*/1)

#pragma endregion F16

#pragma region E5M2

nk_define_cross_tma_blackwell_(angular, e5m2, blackwell, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*paired_dot=*/1)
nk_define_cross_tma_blackwell_(euclidean, e5m2, blackwell, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*paired_dot=*/1)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_tma_blackwell_(angular, e4m3, blackwell, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*paired_dot=*/1)
nk_define_cross_tma_blackwell_(euclidean, e4m3, blackwell, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*paired_dot=*/1)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_tma_blackwell_(angular, e3m2, blackwell, e3m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*paired_dot=*/0)
nk_define_cross_tma_blackwell_(euclidean, e3m2, blackwell, e3m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*paired_dot=*/0)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_tma_blackwell_(angular, e2m3, blackwell, e2m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*paired_dot=*/0)
nk_define_cross_tma_blackwell_(euclidean, e2m3, blackwell, e2m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*paired_dot=*/0)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_tma_blackwell_(angular, e2m1, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, /*box_bytes=*/128, /*paired_dot=*/0)
nk_define_cross_tma_blackwell_(euclidean, e2m1, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, /*box_bytes=*/128, /*paired_dot=*/0)

#pragma endregion E2M1

#pragma region I8

nk_define_cross_tma_blackwell_(angular, i8, blackwell, i8, f16, f32, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/1, /*box_bytes=*/64, /*paired_dot=*/0)
nk_define_cross_tma_blackwell_(euclidean, i8, blackwell, i8, f16, f32, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/1, /*box_bytes=*/64, /*paired_dot=*/0)

#pragma endregion I8

#pragma region I4

nk_define_cross_tma_blackwell_(angular, i4, blackwell, i4x2, i4x2, f32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, /*box_bytes=*/32, /*paired_dot=*/0)
nk_define_cross_tma_blackwell_(euclidean, i4, blackwell, i4x2, i4x2, f32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, /*box_bytes=*/32, /*paired_dot=*/0)

#pragma endregion I4

#pragma region U8

nk_define_cross_tma_blackwell_(angular, u8, blackwell, u8, f16, f32, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/1, /*box_bytes=*/64, /*paired_dot=*/0)
nk_define_cross_tma_blackwell_(euclidean, u8, blackwell, u8, f16, f32, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/1, /*box_bytes=*/64, /*paired_dot=*/0)

#pragma endregion U8

#pragma region U4

nk_define_cross_tma_blackwell_(angular, u4, blackwell, u4x2, u4x2, f32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, /*box_bytes=*/32, /*paired_dot=*/0)
nk_define_cross_tma_blackwell_(euclidean, u4, blackwell, u4x2, u4x2, f32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, /*box_bytes=*/32, /*paired_dot=*/0)

#pragma endregion U4

#pragma region Block Scaled Floats

nk_define_cross_tma_blackwell_(angular, nvfp4, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, /*box_bytes=*/128, /*paired_dot=*/0)
nk_define_cross_tma_blackwell_(euclidean, nvfp4, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, /*box_bytes=*/128, /*paired_dot=*/0)
nk_define_cross_tma_blackwell_(angular, mxfp4, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, /*box_bytes=*/128, /*paired_dot=*/0)
nk_define_cross_tma_blackwell_(euclidean, mxfp4, blackwell, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/2, /*box_bytes=*/128, /*paired_dot=*/0)
nk_define_cross_tma_blackwell_(angular, mxfp8e4m3, blackwell, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*paired_dot=*/0)
nk_define_cross_tma_blackwell_(euclidean, mxfp8e4m3, blackwell, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*paired_dot=*/0)
nk_define_cross_tma_blackwell_(angular, mxfp8e5m2, blackwell, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*paired_dot=*/0)
nk_define_cross_tma_blackwell_(euclidean, mxfp8e5m2, blackwell, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                               /*dimensions_per_value=*/1, /*box_bytes=*/128, /*paired_dot=*/0)

#pragma endregion Block Scaled Floats
#endif // NUMKONG_TARGET_BLACKWELL

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_BLACKWELL_
#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_SPATIALS_BLACKWELL_CUH
