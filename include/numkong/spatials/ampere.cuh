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
 *  sum. F32 squares its rows in Dot2 before the first slab, as the @c cuda tile orders them, and
 *  outputs F64. F64 squares its rows in Dot2 while it scales them onto their grids. Output
 *  precision, zero-norm handling and the triangle @c symmetric writes follow the serial kernels:
 *  F32 outputs for the other types, with I8 and I4 norms read as I32.
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

#pragma region F64

NUMKONG_API nk_status_t nk_angulars_packed_workspace_size_f64_ampere(nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                     nk_size_t *bytes) {
    return nk_cross_workspace_packed_f64_ampere_(rows, columns, depth, bytes);
}

NUMKONG_API nk_status_t nk_angulars_packed_f64_ampere(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *workspace,
                                                      nk_size_t workspace_bytes, nk_stream_t stream) {
    return nk_cross_packed_f64_ampere_(nk_cross_metric_angular_k, a, b_packed, c, rows, columns, depth, a_stride,
                                       c_stride, workspace, workspace_bytes, stream);
}

NUMKONG_API nk_status_t nk_angulars_symmetric_workspace_size_f64_ampere(nk_size_t vector_count, nk_size_t depth,
                                                                        nk_size_t rows_begin, nk_size_t rows_end,
                                                                        nk_size_t *bytes) {
    return nk_cross_workspace_symmetric_f64_ampere_(vector_count, depth, rows_begin, rows_end, bytes);
}

NUMKONG_API nk_status_t nk_angulars_symmetric_f64_ampere(nk_f64_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, void *workspace, nk_size_t workspace_bytes,
                                                         nk_stream_t stream) {
    return nk_cross_symmetric_f64_ampere_(nk_cross_metric_angular_k, vectors, vector_count, depth, stride, result,
                                          result_stride, rows_begin, rows_end, workspace, workspace_bytes, stream);
}

NUMKONG_API nk_status_t nk_euclideans_packed_workspace_size_f64_ampere(nk_size_t rows, nk_size_t columns,
                                                                       nk_size_t depth, nk_size_t *bytes) {
    return nk_cross_workspace_packed_f64_ampere_(rows, columns, depth, bytes);
}

NUMKONG_API nk_status_t nk_euclideans_packed_f64_ampere(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *workspace,
                                                        nk_size_t workspace_bytes, nk_stream_t stream) {
    return nk_cross_packed_f64_ampere_(nk_cross_metric_euclidean_k, a, b_packed, c, rows, columns, depth, a_stride,
                                       c_stride, workspace, workspace_bytes, stream);
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_workspace_size_f64_ampere(nk_size_t vector_count, nk_size_t depth,
                                                                          nk_size_t rows_begin, nk_size_t rows_end,
                                                                          nk_size_t *bytes) {
    return nk_cross_workspace_symmetric_f64_ampere_(vector_count, depth, rows_begin, rows_end, bytes);
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_ampere(nk_f64_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, void *workspace,
                                                           nk_size_t workspace_bytes, nk_stream_t stream) {
    return nk_cross_symmetric_f64_ampere_(nk_cross_metric_euclidean_k, vectors, vector_count, depth, stride, result,
                                          result_stride, rows_begin, rows_end, workspace, workspace_bytes, stream);
}

#pragma endregion F64

#pragma region F32

nk_define_cross_cuda_(angular, f32, ampere, f32_ampere, f32, f32, f64, /*depth_simd_dimensions=*/4,
                      /*dimensions_per_value=*/1)
nk_define_cross_cuda_(euclidean, f32, ampere, f32_ampere, f32, f32, f64, /*depth_simd_dimensions=*/4,
                      /*dimensions_per_value=*/1)

#pragma endregion F32

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
