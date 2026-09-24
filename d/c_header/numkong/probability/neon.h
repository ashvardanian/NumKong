/**
 *  @brief NEON-accelerated Probability Distribution Similarity Measures.
 *  @file include/numkong/probability/neon.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/probability.h
 */
#ifndef NK_PROBABILITY_NEON_H
#define NK_PROBABILITY_NEON_H

#if NK_TARGET_ARM_

#include "numkong/types.h"
#include "numkong/cast/serial.h"  // `nk_partial_load_b16x4_serial_`, `nk_partial_load_b32x4_serial_`
#include "numkong/spatial/neon.h" // `nk_f32_sqrt_neon`

#if defined(__cplusplus)
extern "C" {
#endif

#if NK_TARGET_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#endif

NK_PUBLIC float32x4_t nk_log2_f32x4_neon_(float32x4_t x) {
    // Extracting the exponent
    int32x4_t bits_i32x4 = vreinterpretq_s32_f32(x);
    int32x4_t exponent_i32x4 = vsubq_s32(vshrq_n_s32(vandq_s32(bits_i32x4, vdupq_n_s32(0x7F800000)), 23),
                                         vdupq_n_s32(127));
    float32x4_t exponent_f32x4 = vcvtq_f32_s32(exponent_i32x4);

    // Extracting the mantissa
    float32x4_t mantissa_f32x4 = vreinterpretq_f32_s32(
        vorrq_s32(vandq_s32(bits_i32x4, vdupq_n_s32(0x007FFFFF)), vdupq_n_s32(0x3F800000)));

    // Constants for polynomial
    float32x4_t one_f32x4 = vdupq_n_f32(1.0f);
    float32x4_t poly_f32x4 = vdupq_n_f32(-3.4436006e-2f);

    // Compute polynomial using Horner's method
    poly_f32x4 = vmlaq_f32(vdupq_n_f32(3.1821337e-1f), mantissa_f32x4, poly_f32x4);
    poly_f32x4 = vmlaq_f32(vdupq_n_f32(-1.2315303f), mantissa_f32x4, poly_f32x4);
    poly_f32x4 = vmlaq_f32(vdupq_n_f32(2.5988452f), mantissa_f32x4, poly_f32x4);
    poly_f32x4 = vmlaq_f32(vdupq_n_f32(-3.3241990f), mantissa_f32x4, poly_f32x4);
    poly_f32x4 = vmlaq_f32(vdupq_n_f32(3.1157899f), mantissa_f32x4, poly_f32x4);

    // Final computation
    float32x4_t result_f32x4 = vaddq_f32(vmulq_f32(poly_f32x4, vsubq_f32(mantissa_f32x4, one_f32x4)), exponent_f32x4);
    return result_f32x4;
}

NK_PUBLIC void nk_kld_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    float32x4_t epsilon_f32x4 = vdupq_n_f32(epsilon);
    float64x2_t sum_lower_f64x2 = vdupq_n_f64(0.0);
    float64x2_t sum_upper_f64x2 = vdupq_n_f64(0.0);
    float32x4_t a_f32x4, b_f32x4;

nk_kld_f32_neon_cycle:
    if (n < 4) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b32x4_serial_(a, &a_vec, n);
        nk_partial_load_b32x4_serial_(b, &b_vec, n);
        a_f32x4 = a_vec.f32x4;
        b_f32x4 = b_vec.f32x4;
        n = 0;
    }
    else {
        a_f32x4 = vld1q_f32(a);
        b_f32x4 = vld1q_f32(b);
        n -= 4, a += 4, b += 4;
    }

    float32x4_t ratio_f32x4 = vdivq_f32(vaddq_f32(a_f32x4, epsilon_f32x4), vaddq_f32(b_f32x4, epsilon_f32x4));
    float32x4_t log_ratio_f32x4 = nk_log2_f32x4_neon_(ratio_f32x4);
    float32x4_t contribution_f32x4 = vmulq_f32(a_f32x4, log_ratio_f32x4);
    sum_lower_f64x2 = vaddq_f64(sum_lower_f64x2, vcvt_f64_f32(vget_low_f32(contribution_f32x4)));
    sum_upper_f64x2 = vaddq_f64(sum_upper_f64x2, vcvt_f64_f32(vget_high_f32(contribution_f32x4)));
    if (n != 0) goto nk_kld_f32_neon_cycle;

    nk_f64_t log2_normalizer = 0.6931471805599453;
    nk_f64_t sum = vaddvq_f64(vaddq_f64(sum_lower_f64x2, sum_upper_f64x2)) * log2_normalizer;
    *result = sum;
}

NK_PUBLIC void nk_jsd_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    float32x4_t epsilon_f32x4 = vdupq_n_f32(epsilon);
    float64x2_t sum_lower_f64x2 = vdupq_n_f64(0.0);
    float64x2_t sum_upper_f64x2 = vdupq_n_f64(0.0);
    float32x4_t a_f32x4, b_f32x4;

nk_jsd_f32_neon_cycle:
    if (n < 4) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b32x4_serial_(a, &a_vec, n);
        nk_partial_load_b32x4_serial_(b, &b_vec, n);
        a_f32x4 = a_vec.f32x4;
        b_f32x4 = b_vec.f32x4;
        n = 0;
    }
    else {
        a_f32x4 = vld1q_f32(a);
        b_f32x4 = vld1q_f32(b);
        n -= 4, a += 4, b += 4;
    }

    float32x4_t mean_f32x4 = vmulq_n_f32(vaddq_f32(a_f32x4, b_f32x4), 0.5f);
    float32x4_t ratio_a_f32x4 = vdivq_f32(vaddq_f32(a_f32x4, epsilon_f32x4), vaddq_f32(mean_f32x4, epsilon_f32x4));
    float32x4_t ratio_b_f32x4 = vdivq_f32(vaddq_f32(b_f32x4, epsilon_f32x4), vaddq_f32(mean_f32x4, epsilon_f32x4));
    float32x4_t log_ratio_a_f32x4 = nk_log2_f32x4_neon_(ratio_a_f32x4);
    float32x4_t log_ratio_b_f32x4 = nk_log2_f32x4_neon_(ratio_b_f32x4);
    float32x4_t contribution_a_f32x4 = vmulq_f32(a_f32x4, log_ratio_a_f32x4);
    float32x4_t contribution_b_f32x4 = vmulq_f32(b_f32x4, log_ratio_b_f32x4);
    float32x4_t contribution_f32x4 = vaddq_f32(contribution_a_f32x4, contribution_b_f32x4);
    sum_lower_f64x2 = vaddq_f64(sum_lower_f64x2, vcvt_f64_f32(vget_low_f32(contribution_f32x4)));
    sum_upper_f64x2 = vaddq_f64(sum_upper_f64x2, vcvt_f64_f32(vget_high_f32(contribution_f32x4)));
    if (n != 0) goto nk_jsd_f32_neon_cycle;

    nk_f64_t log2_normalizer = 0.6931471805599453;
    nk_f64_t sum = vaddvq_f64(vaddq_f64(sum_lower_f64x2, sum_upper_f64x2)) * log2_normalizer / 2.0;
    *result = sum > 0 ? nk_f64_sqrt_neon(sum) : 0;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

NK_PUBLIC void nk_kld_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    float32x4_t epsilon_f32x4 = vdupq_n_f32(epsilon);
    float32x4_t a_f32x4, b_f32x4;

nk_kld_f16_neonhalf_cycle:
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

    float32x4_t ratio_f32x4 = vdivq_f32(vaddq_f32(a_f32x4, epsilon_f32x4), vaddq_f32(b_f32x4, epsilon_f32x4));
    float32x4_t log_ratio_f32x4 = nk_log2_f32x4_neon_(ratio_f32x4);
    float32x4_t contribution_f32x4 = vmulq_f32(a_f32x4, log_ratio_f32x4);
    sum_f32x4 = vaddq_f32(sum_f32x4, contribution_f32x4);
    if (n) goto nk_kld_f16_neonhalf_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    nk_f32_t sum = vaddvq_f32(sum_f32x4) * log2_normalizer;
    *result = sum;
}

NK_PUBLIC void nk_jsd_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    float32x4_t epsilon_f32x4 = vdupq_n_f32(epsilon);
    float32x4_t a_f32x4, b_f32x4;

nk_jsd_f16_neonhalf_cycle:
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

    float32x4_t mean_f32x4 = vmulq_n_f32(vaddq_f32(a_f32x4, b_f32x4), 0.5f);
    float32x4_t ratio_a_f32x4 = vdivq_f32(vaddq_f32(a_f32x4, epsilon_f32x4), vaddq_f32(mean_f32x4, epsilon_f32x4));
    float32x4_t ratio_b_f32x4 = vdivq_f32(vaddq_f32(b_f32x4, epsilon_f32x4), vaddq_f32(mean_f32x4, epsilon_f32x4));
    float32x4_t log_ratio_a_f32x4 = nk_log2_f32x4_neon_(ratio_a_f32x4);
    float32x4_t log_ratio_b_f32x4 = nk_log2_f32x4_neon_(ratio_b_f32x4);
    float32x4_t contribution_a_f32x4 = vmulq_f32(a_f32x4, log_ratio_a_f32x4);
    float32x4_t contribution_b_f32x4 = vmulq_f32(b_f32x4, log_ratio_b_f32x4);
    sum_f32x4 = vaddq_f32(sum_f32x4, vaddq_f32(contribution_a_f32x4, contribution_b_f32x4));
    if (n) goto nk_jsd_f16_neonhalf_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    nk_f32_t sum = vaddvq_f32(sum_f32x4) * log2_normalizer / 2;
    *result = sum > 0 ? nk_f32_sqrt_neon(sum) : 0;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEONHALF

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_ARM_
#endif // NK_PROBABILITY_NEON_H
