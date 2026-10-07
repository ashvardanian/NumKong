/**
 *  @file include/numkong/spatials/cdna3.cuh
 *  @author Ash Vardanian
 *  @date October 7, 2026
 *  @brief Batched spatial distances for AMD Instinct MI300, gfx942, on its SIMT cores.
 *
 *  @sa include/numkong/spatials.h
 *  @sa include/numkong/dots/cdna3.cuh
 *
 *  The CDNA3 dots tile with a metric in its epilogue, for the types whose products and squares
 *  fold through CDNA's signed byte and F16 pair dots; the norms, outputs and triangles follow the
 *  @c rocm spatials, and every other type runs those.
 */
#ifndef NUMKONG_SPATIALS_CDNA3_CUH
#define NUMKONG_SPATIALS_CDNA3_CUH

#if NUMKONG_TARGET_CDNA3

#include "numkong/dots/cdna3.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

nk_define_cross_rocm_(angular, f16, cdna3, b32_cdna3, f16, f16, f32, 8, 1, nk_f16_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(euclidean, f16, cdna3, b32_cdna3, f16, f16, f32, 8, 1, nk_f16_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(angular, e5m2, cdna3, b32_cdna3, e5m2, e5m2, f32, 16, 1, nk_e5m2_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(euclidean, e5m2, cdna3, b32_cdna3, e5m2, e5m2, f32, 16, 1, nk_e5m2_k,
                      nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(angular, e4m3, cdna3, b32_cdna3, e4m3, e4m3, f32, 16, 1, nk_e4m3_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(euclidean, e4m3, cdna3, b32_cdna3, e4m3, e4m3, f32, 16, 1, nk_e4m3_k,
                      nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(angular, e3m2, cdna3, b32_cdna3, e3m2, e3m2, f32, 16, 1, nk_e3m2_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(euclidean, e3m2, cdna3, b32_cdna3, e3m2, e3m2, f32, 16, 1, nk_e3m2_k,
                      nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(angular, e2m3, cdna3, b32_cdna3, e2m3, e2m3, f32, 16, 1, nk_e2m3_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(euclidean, e2m3, cdna3, b32_cdna3, e2m3, e2m3, f32, 16, 1, nk_e2m3_k,
                      nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(angular, e2m1, cdna3, b32_cdna3, e2m1x2, e2m1x2, f32, 32, 2, nk_e2m1_k,
                      nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(euclidean, e2m1, cdna3, b32_cdna3, e2m1x2, e2m1x2, f32, 32, 2, nk_e2m1_k,
                      nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(angular, i8, cdna3, b32_cdna3, i8, i8, f32, 16, 1, nk_i8_k, nk_cross_accumulation_i8x4_k)
nk_define_cross_rocm_(euclidean, i8, cdna3, b32_cdna3, i8, i8, f32, 16, 1, nk_i8_k, nk_cross_accumulation_i8x4_k)
nk_define_cross_rocm_(angular, i4, cdna3, b32_cdna3, i4x2, i4x2, f32, 32, 2, nk_i4_k, nk_cross_accumulation_i4x8_k)
nk_define_cross_rocm_(euclidean, i4, cdna3, b32_cdna3, i4x2, i4x2, f32, 32, 2, nk_i4_k, nk_cross_accumulation_i4x8_k)

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_CDNA3
#endif // NUMKONG_SPATIALS_CDNA3_CUH
