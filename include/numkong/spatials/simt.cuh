/**
 *  @file include/numkong/spatials/simt.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batched spatial distances on the SIMT cores of every CUDA and ROCm device.
 *
 *  @sa include/numkong/spatials.h
 *  @sa include/numkong/dots/simt.cuh
 *
 *  The baseline dots tiles with a metric in their epilogue: squared norms of A, and for
 *  @c symmetric of the column vectors too, accumulate while the products do, and @c packed reads
 *  the column norms its pack stored. Norm types, output precision, zero-norm handling and the
 *  triangle @c symmetric writes follow the serial backends: F64 outputs for F64 and F32 inputs,
 *  and F32 outputs for every other type.
 */
#ifndef NUMKONG_SPATIALS_SIMT_CUH
#define NUMKONG_SPATIALS_SIMT_CUH

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#include "numkong/dots/simt.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#if NUMKONG_TARGET_CUDA
nk_define_cross_simt_(angular, f64, cuda, f64_simt, f64, f64, f64, 2, 1, nk_f64_k, nk_cross_accumulation_dot2_k)
nk_define_cross_simt_(euclidean, f64, cuda, f64_simt, f64, f64, f64, 2, 1, nk_f64_k, nk_cross_accumulation_dot2_k)
nk_define_cross_simt_(angular, f32, cuda, f64_simt, f32, f32, f64, 4, 1, nk_f32_k, nk_cross_accumulation_f64_k)
nk_define_cross_simt_(euclidean, f32, cuda, f64_simt, f32, f32, f64, 4, 1, nk_f32_k, nk_cross_accumulation_f64_k)
nk_define_cross_simt_(angular, bf16, cuda, b32_simt, bf16, bf16, f32, 8, 1, nk_bf16_k, nk_cross_accumulation_f32_k)
nk_define_cross_simt_(euclidean, bf16, cuda, b32_simt, bf16, bf16, f32, 8, 1, nk_bf16_k, nk_cross_accumulation_f32_k)
nk_define_cross_simt_(angular, f16, cuda, b32_simt, f16, f16, f32, 8, 1, nk_f16_k, nk_cross_accumulation_f32_k)
nk_define_cross_simt_(euclidean, f16, cuda, b32_simt, f16, f16, f32, 8, 1, nk_f16_k, nk_cross_accumulation_f32_k)
nk_define_cross_simt_(angular, e5m2, cuda, b32_simt, e5m2, e5m2, f32, 16, 1, nk_e5m2_k, nk_cross_accumulation_f32_k)
nk_define_cross_simt_(euclidean, e5m2, cuda, b32_simt, e5m2, e5m2, f32, 16, 1, nk_e5m2_k, nk_cross_accumulation_f32_k)
nk_define_cross_simt_(angular, e4m3, cuda, b32_simt, e4m3, e4m3, f32, 16, 1, nk_e4m3_k, nk_cross_accumulation_f32_k)
nk_define_cross_simt_(euclidean, e4m3, cuda, b32_simt, e4m3, e4m3, f32, 16, 1, nk_e4m3_k, nk_cross_accumulation_f32_k)
nk_define_cross_simt_(angular, e3m2, cuda, b32_simt, e3m2, e3m2, f32, 16, 1, nk_e3m2_k, nk_cross_accumulation_f32_k)
nk_define_cross_simt_(euclidean, e3m2, cuda, b32_simt, e3m2, e3m2, f32, 16, 1, nk_e3m2_k, nk_cross_accumulation_f32_k)
nk_define_cross_simt_(angular, e2m3, cuda, b32_simt, e2m3, e2m3, f32, 16, 1, nk_e2m3_k, nk_cross_accumulation_f32_k)
nk_define_cross_simt_(euclidean, e2m3, cuda, b32_simt, e2m3, e2m3, f32, 16, 1, nk_e2m3_k, nk_cross_accumulation_f32_k)
nk_define_cross_simt_(angular, e2m1, cuda, b32_simt, e2m1x2, e2m1x2, f32, 32, 2, nk_e2m1_k, nk_cross_accumulation_f32_k)
nk_define_cross_simt_(euclidean, e2m1, cuda, b32_simt, e2m1x2, e2m1x2, f32, 32, 2, nk_e2m1_k,
                      nk_cross_accumulation_f32_k)
nk_define_cross_simt_(angular, i8, cuda, b32_simt, i8, i8, f32, 16, 1, nk_i8_k, nk_cross_accumulation_i8x4_k)
nk_define_cross_simt_(euclidean, i8, cuda, b32_simt, i8, i8, f32, 16, 1, nk_i8_k, nk_cross_accumulation_i8x4_k)
nk_define_cross_simt_(angular, u8, cuda, b32_simt, u8, u8, f32, 16, 1, nk_u8_k, nk_cross_accumulation_u8x4_k)
nk_define_cross_simt_(euclidean, u8, cuda, b32_simt, u8, u8, f32, 16, 1, nk_u8_k, nk_cross_accumulation_u8x4_k)
nk_define_cross_simt_(angular, i4, cuda, b32_simt, i4x2, i4x2, f32, 32, 2, nk_i4_k, nk_cross_accumulation_i4x4_k)
nk_define_cross_simt_(euclidean, i4, cuda, b32_simt, i4x2, i4x2, f32, 32, 2, nk_i4_k, nk_cross_accumulation_i4x4_k)
nk_define_cross_simt_(angular, u4, cuda, b32_simt, u4x2, u4x2, f32, 32, 2, nk_u4_k, nk_cross_accumulation_u4x4_k)
nk_define_cross_simt_(euclidean, u4, cuda, b32_simt, u4x2, u4x2, f32, 32, 2, nk_u4_k, nk_cross_accumulation_u4x4_k)
nk_define_cross_simt_(angular, nvfp4, cuda, scaled_simt, e2m1x2, e2m1x2, f32, 32, 2, nk_nvfp4_k)
nk_define_cross_simt_(euclidean, nvfp4, cuda, scaled_simt, e2m1x2, e2m1x2, f32, 32, 2, nk_nvfp4_k)
nk_define_cross_simt_(angular, mxfp4, cuda, scaled_simt, e2m1x2, e2m1x2, f32, 32, 2, nk_mxfp4_k)
nk_define_cross_simt_(euclidean, mxfp4, cuda, scaled_simt, e2m1x2, e2m1x2, f32, 32, 2, nk_mxfp4_k)
nk_define_cross_simt_(angular, mxfp8e4m3, cuda, scaled_simt, e4m3, e4m3, f32, 16, 1, nk_mxfp8e4m3_k)
nk_define_cross_simt_(euclidean, mxfp8e4m3, cuda, scaled_simt, e4m3, e4m3, f32, 16, 1, nk_mxfp8e4m3_k)
nk_define_cross_simt_(angular, mxfp8e5m2, cuda, scaled_simt, e5m2, e5m2, f32, 16, 1, nk_mxfp8e5m2_k)
nk_define_cross_simt_(euclidean, mxfp8e5m2, cuda, scaled_simt, e5m2, e5m2, f32, 16, 1, nk_mxfp8e5m2_k)
#elif NUMKONG_TARGET_ROCM
nk_define_cross_simt_(angular, f64, rocm, f64_simt, f64, f64, f64, 2, 1, nk_f64_k, nk_cross_accumulation_dot2_k)
nk_define_cross_simt_(euclidean, f64, rocm, f64_simt, f64, f64, f64, 2, 1, nk_f64_k, nk_cross_accumulation_dot2_k)
nk_define_cross_simt_(angular, f32, rocm, f64_simt, f32, f32, f64, 4, 1, nk_f32_k, nk_cross_accumulation_f64_k)
nk_define_cross_simt_(euclidean, f32, rocm, f64_simt, f32, f32, f64, 4, 1, nk_f32_k, nk_cross_accumulation_f64_k)
nk_define_cross_simt_(angular, bf16, rocm, b32_simt, bf16, bf16, f32, 8, 1, nk_bf16_k, nk_cross_accumulation_f32_k)
nk_define_cross_simt_(euclidean, bf16, rocm, b32_simt, bf16, bf16, f32, 8, 1, nk_bf16_k, nk_cross_accumulation_f32_k)
nk_define_cross_simt_(angular, f16, rocm, b32_simt, f16, f16, f32, 8, 1, nk_f16_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_simt_(euclidean, f16, rocm, b32_simt, f16, f16, f32, 8, 1, nk_f16_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_simt_(angular, e5m2, rocm, b32_simt, e5m2, e5m2, f32, 16, 1, nk_e5m2_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_simt_(euclidean, e5m2, rocm, b32_simt, e5m2, e5m2, f32, 16, 1, nk_e5m2_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_simt_(angular, e4m3, rocm, b32_simt, e4m3, e4m3, f32, 16, 1, nk_e4m3_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_simt_(euclidean, e4m3, rocm, b32_simt, e4m3, e4m3, f32, 16, 1, nk_e4m3_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_simt_(angular, e3m2, rocm, b32_simt, e3m2, e3m2, f32, 16, 1, nk_e3m2_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_simt_(euclidean, e3m2, rocm, b32_simt, e3m2, e3m2, f32, 16, 1, nk_e3m2_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_simt_(angular, e2m3, rocm, b32_simt, e2m3, e2m3, f32, 16, 1, nk_e2m3_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_simt_(euclidean, e2m3, rocm, b32_simt, e2m3, e2m3, f32, 16, 1, nk_e2m3_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_simt_(angular, e2m1, rocm, b32_simt, e2m1x2, e2m1x2, f32, 32, 2, nk_e2m1_k,
                      nk_cross_accumulation_f16x2_k)
nk_define_cross_simt_(euclidean, e2m1, rocm, b32_simt, e2m1x2, e2m1x2, f32, 32, 2, nk_e2m1_k,
                      nk_cross_accumulation_f16x2_k)
nk_define_cross_simt_(angular, i8, rocm, b32_simt, i8, i8, f32, 16, 1, nk_i8_k, nk_cross_accumulation_i8x4_k)
nk_define_cross_simt_(euclidean, i8, rocm, b32_simt, i8, i8, f32, 16, 1, nk_i8_k, nk_cross_accumulation_i8x4_k)
nk_define_cross_simt_(angular, u8, rocm, b32_simt, u8, u8, f32, 16, 1, nk_u8_k, nk_cross_accumulation_u8x4_k)
nk_define_cross_simt_(euclidean, u8, rocm, b32_simt, u8, u8, f32, 16, 1, nk_u8_k, nk_cross_accumulation_u8x4_k)
nk_define_cross_simt_(angular, i4, rocm, b32_simt, i4x2, i4x2, f32, 32, 2, nk_i4_k, nk_cross_accumulation_i4x8_k)
nk_define_cross_simt_(euclidean, i4, rocm, b32_simt, i4x2, i4x2, f32, 32, 2, nk_i4_k, nk_cross_accumulation_i4x8_k)
nk_define_cross_simt_(angular, u4, rocm, b32_simt, u4x2, u4x2, f32, 32, 2, nk_u4_k, nk_cross_accumulation_u4x8_k)
nk_define_cross_simt_(euclidean, u4, rocm, b32_simt, u4x2, u4x2, f32, 32, 2, nk_u4_k, nk_cross_accumulation_u4x8_k)
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#endif // NUMKONG_SPATIALS_SIMT_CUH
