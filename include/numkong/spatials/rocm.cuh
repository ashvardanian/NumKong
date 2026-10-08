/**
 *  @file include/numkong/spatials/rocm.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batched spatial distances on the SIMT cores of every ROCm device, the @c rocm exports.
 *
 *  @sa include/numkong/spatials.h
 *  @sa include/numkong/dots/rocm.cuh
 *
 *  The baseline dots tiles with a metric in their epilogue: squared norms of A, and for
 *  @c symmetric of the column vectors too, accumulate while the products do, and @c packed reads
 *  the column norms its pack stored. Norm types, output precision, zero-norm handling and the
 *  triangle @c symmetric writes follow the serial backends: F64 outputs for F64 and F32 inputs,
 *  and F32 outputs for every other type.
 */
#ifndef NUMKONG_SPATIALS_ROCM_CUH
#define NUMKONG_SPATIALS_ROCM_CUH

#if NUMKONG_ARCH_ROCM_

#include "numkong/dots/rocm.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#if NUMKONG_TARGET_ROCM
nk_define_cross_rocm_(angular, f64, rocm, f64_rocm, f64, f64, f64, 2, 1, nk_f64_k, nk_cross_accumulation_dot2_k)
nk_define_cross_rocm_(euclidean, f64, rocm, f64_rocm, f64, f64, f64, 2, 1, nk_f64_k, nk_cross_accumulation_dot2_k)
nk_define_cross_rocm_(angular, f32, rocm, f64_rocm, f32, f32, f64, 4, 1, nk_f32_k, nk_cross_accumulation_f64_k)
nk_define_cross_rocm_(euclidean, f32, rocm, f64_rocm, f32, f32, f64, 4, 1, nk_f32_k, nk_cross_accumulation_f64_k)
nk_define_cross_rocm_(angular, bf16, rocm, b32_rocm, bf16, bf16, f32, 8, 1, nk_bf16_k, nk_cross_accumulation_f32_k)
nk_define_cross_rocm_(euclidean, bf16, rocm, b32_rocm, bf16, bf16, f32, 8, 1, nk_bf16_k, nk_cross_accumulation_f32_k)
nk_define_cross_rocm_(angular, f16, rocm, b32_rocm, f16, f16, f32, 8, 1, nk_f16_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(euclidean, f16, rocm, b32_rocm, f16, f16, f32, 8, 1, nk_f16_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(angular, e5m2, rocm, b32_rocm, e5m2, e5m2, f32, 16, 1, nk_e5m2_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(euclidean, e5m2, rocm, b32_rocm, e5m2, e5m2, f32, 16, 1, nk_e5m2_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(angular, e4m3, rocm, b32_rocm, e4m3, e4m3, f32, 16, 1, nk_e4m3_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(euclidean, e4m3, rocm, b32_rocm, e4m3, e4m3, f32, 16, 1, nk_e4m3_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(angular, e3m2, rocm, b32_rocm, e3m2, e3m2, f32, 16, 1, nk_e3m2_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(euclidean, e3m2, rocm, b32_rocm, e3m2, e3m2, f32, 16, 1, nk_e3m2_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(angular, e2m3, rocm, b32_rocm, e2m3, e2m3, f32, 16, 1, nk_e2m3_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(euclidean, e2m3, rocm, b32_rocm, e2m3, e2m3, f32, 16, 1, nk_e2m3_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(angular, e2m1, rocm, b32_rocm, e2m1x2, e2m1x2, f32, 32, 2, nk_e2m1_k,
                      nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(euclidean, e2m1, rocm, b32_rocm, e2m1x2, e2m1x2, f32, 32, 2, nk_e2m1_k,
                      nk_cross_accumulation_f16x2_k)
nk_define_cross_rocm_(angular, i8, rocm, b32_rocm, i8, i8, f32, 16, 1, nk_i8_k, nk_cross_accumulation_i8x4_k)
nk_define_cross_rocm_(euclidean, i8, rocm, b32_rocm, i8, i8, f32, 16, 1, nk_i8_k, nk_cross_accumulation_i8x4_k)
nk_define_cross_rocm_(angular, u8, rocm, b32_rocm, u8, u8, f32, 16, 1, nk_u8_k, nk_cross_accumulation_u8x4_k)
nk_define_cross_rocm_(euclidean, u8, rocm, b32_rocm, u8, u8, f32, 16, 1, nk_u8_k, nk_cross_accumulation_u8x4_k)
nk_define_cross_rocm_(angular, i4, rocm, b32_rocm, i4x2, i4x2, f32, 32, 2, nk_i4_k, nk_cross_accumulation_i4x8_k)
nk_define_cross_rocm_(euclidean, i4, rocm, b32_rocm, i4x2, i4x2, f32, 32, 2, nk_i4_k, nk_cross_accumulation_i4x8_k)
nk_define_cross_rocm_(angular, u4, rocm, b32_rocm, u4x2, u4x2, f32, 32, 2, nk_u4_k, nk_cross_accumulation_u4x8_k)
nk_define_cross_rocm_(euclidean, u4, rocm, b32_rocm, u4x2, u4x2, f32, 32, 2, nk_u4_k, nk_cross_accumulation_u4x8_k)
#endif // NUMKONG_TARGET_ROCM

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_SPATIALS_ROCM_CUH
