/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for NEON FP16.
 *  @file include/numkong/spatial/neonhalf.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_neonhalf_instructions ARM NEON FP16 Instructions (ARMv8.2-FP16)
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *                                                                              A76         M4+/V1+/Oryon
 *      vfmaq_f16                   FMLA (V.8H, V.8H, V.8H)         4cy         2/cy        4/cy
 *      vcvt_f32_f16                FCVTL (V.4S, V.4H)              3cy         2/cy        4/cy
 *      vld1q_f16                   LD1 (V.8H)                      4cy         2/cy        3/cy
 *      vsubq_f16                   FSUB (V.8H, V.8H, V.8H)         2cy         2/cy        4/cy
 *      vaddvq_f32                  FADDP+FADDP (V.4S)              4cy         1/cy        2/cy
 *
 *  The ARMv8.2-FP16 extension enables native half-precision arithmetic, doubling the element count
 *  per vector register (8x F16 vs 4x F32). For spatial distance computations like L2 and angular
 *  distance, this halves memory bandwidth requirements.
 *
 *  Inputs are widened from F16 to F32 for accumulation via FCVTL to preserve numerical precision
 *  during the squared difference summation. The subtraction and FMA operations use F32 precision
 *  in the accumulator to avoid catastrophic cancellation in distance computations.
 */
#ifndef NK_SPATIAL_NEONHALF_H
#define NK_SPATIAL_NEONHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF

#include "numkong/types.h"
#include "numkong/cast/serial.h"  // `nk_partial_load_b16x4_serial_`
#include "numkong/spatial/neon.h" // `nk_angular_normalize_f32_neon_`, `nk_f32_sqrt_neon`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

NK_PUBLIC void nk_sqeuclidean_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t a_f32x4, b_f32x4;
    float32x4_t distance_sq_f32x4 = vdupq_n_f32(0);

nk_sqeuclidean_f16_neonhalf_cycle:
    if (n < 4) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b16x4_serial_(a, &a_vec, n);
        nk_partial_load_b16x4_serial_(b, &b_vec, n);
        a_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(a_vec.u16x4));
        b_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(b_vec.u16x4));
        n = 0;
    }
    else {
        a_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)a));
        b_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)b));
        n -= 4, a += 4, b += 4;
    }
    float32x4_t diff_f32x4 = vsubq_f32(a_f32x4, b_f32x4);
    distance_sq_f32x4 = vfmaq_f32(distance_sq_f32x4, diff_f32x4, diff_f32x4);
    if (n) goto nk_sqeuclidean_f16_neonhalf_cycle;

    *result = vaddvq_f32(distance_sq_f32x4);
}
NK_PUBLIC void nk_euclidean_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_f16_neonhalf(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t dot_product_f32x4 = vdupq_n_f32(0), a_norm_sq_f32x4 = vdupq_n_f32(0), b_norm_sq_f32x4 = vdupq_n_f32(0);
    float32x4_t a_f32x4, b_f32x4;

nk_angular_f16_neonhalf_cycle:
    if (n < 4) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b16x4_serial_(a, &a_vec, n);
        nk_partial_load_b16x4_serial_(b, &b_vec, n);
        a_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(a_vec.u16x4));
        b_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(b_vec.u16x4));
        n = 0;
    }
    else {
        a_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)a));
        b_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)b));
        n -= 4, a += 4, b += 4;
    }
    dot_product_f32x4 = vfmaq_f32(dot_product_f32x4, a_f32x4, b_f32x4);
    a_norm_sq_f32x4 = vfmaq_f32(a_norm_sq_f32x4, a_f32x4, a_f32x4);
    b_norm_sq_f32x4 = vfmaq_f32(b_norm_sq_f32x4, b_f32x4, b_f32x4);
    if (n) goto nk_angular_f16_neonhalf_cycle;

    nk_f32_t dot_product_f32 = vaddvq_f32(dot_product_f32x4);
    nk_f32_t a_norm_sq_f32 = vaddvq_f32(a_norm_sq_f32x4);
    nk_f32_t b_norm_sq_f32 = vaddvq_f32(b_norm_sq_f32x4);
    *result = nk_angular_normalize_f32_neon_(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONHALF
#endif // NK_TARGET_ARM_
#endif // NK_SPATIAL_NEONHALF_H
