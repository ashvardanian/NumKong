/**
 *  @file include/numkong/probability/neon.h
 *  @author Ash Vardanian
 *  @date October 20, 2023
 *  @brief NEON-accelerated probability distribution similarity measures.
 *
 *  @sa c/numkong/probability.h
 */
#ifndef NUMKONG_PROBABILITY_NEON_H
#define NUMKONG_PROBABILITY_NEON_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_NEON_

#include "numkong/types.h"
#include "numkong/cast/serial.h"        // `nk_partial_load_b64x2_serial_`, `nk_partial_load_b32x4_serial_`
#include "numkong/probability/serial.h" // `NUMKONG_F32_DIVISION_EPSILON`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#endif

NUMKONG_INLINE float32x4_t nk_log2_f32x4_neon_(float32x4_t x) {
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

NUMKONG_INLINE float64x2_t nk_log2_f64x2_neon_(float64x2_t x) {
    // Extract the exponent and the mantissa ∈ [1, 2)
    uint64x2_t bits_u64x2 = vreinterpretq_u64_f64(x);
    int64x2_t exponent_i64x2 = vsubq_s64(vreinterpretq_s64_u64(vshrq_n_u64(bits_u64x2, 52)), vdupq_n_s64(1023));
    float64x2_t exponent_f64x2 = vcvtq_f64_s64(exponent_i64x2);
    float64x2_t mantissa_f64x2 = vreinterpretq_f64_u64(
        vorrq_u64(vandq_u64(bits_u64x2, vdupq_n_u64(0x000FFFFFFFFFFFFFULL)), vdupq_n_u64(0x3FF0000000000000ULL)));

    // s-series: s = (m-1)/(m+1), log2(m) = 2 × s × P(s²) × log2(e)
    float64x2_t one_f64x2 = vdupq_n_f64(1.0);
    float64x2_t s_f64x2 = vdivq_f64(vsubq_f64(mantissa_f64x2, one_f64x2), vaddq_f64(mantissa_f64x2, one_f64x2));
    float64x2_t s2_f64x2 = vmulq_f64(s_f64x2, s_f64x2);

    // 14-term Horner: P(s²) = 1 + s²/3 + s⁴/5 + ... + s²⁶/27
    float64x2_t poly_f64x2 = vdupq_n_f64(1.0 / 27.0);
    poly_f64x2 = vfmaq_f64(vdupq_n_f64(1.0 / 25.0), s2_f64x2, poly_f64x2);
    poly_f64x2 = vfmaq_f64(vdupq_n_f64(1.0 / 23.0), s2_f64x2, poly_f64x2);
    poly_f64x2 = vfmaq_f64(vdupq_n_f64(1.0 / 21.0), s2_f64x2, poly_f64x2);
    poly_f64x2 = vfmaq_f64(vdupq_n_f64(1.0 / 19.0), s2_f64x2, poly_f64x2);
    poly_f64x2 = vfmaq_f64(vdupq_n_f64(1.0 / 17.0), s2_f64x2, poly_f64x2);
    poly_f64x2 = vfmaq_f64(vdupq_n_f64(1.0 / 15.0), s2_f64x2, poly_f64x2);
    poly_f64x2 = vfmaq_f64(vdupq_n_f64(1.0 / 13.0), s2_f64x2, poly_f64x2);
    poly_f64x2 = vfmaq_f64(vdupq_n_f64(1.0 / 11.0), s2_f64x2, poly_f64x2);
    poly_f64x2 = vfmaq_f64(vdupq_n_f64(1.0 / 9.0), s2_f64x2, poly_f64x2);
    poly_f64x2 = vfmaq_f64(vdupq_n_f64(1.0 / 7.0), s2_f64x2, poly_f64x2);
    poly_f64x2 = vfmaq_f64(vdupq_n_f64(1.0 / 5.0), s2_f64x2, poly_f64x2);
    poly_f64x2 = vfmaq_f64(vdupq_n_f64(1.0 / 3.0), s2_f64x2, poly_f64x2);
    poly_f64x2 = vfmaq_f64(one_f64x2, s2_f64x2, poly_f64x2);

    float64x2_t ln_m_f64x2 = vmulq_f64(vmulq_n_f64(s_f64x2, 2.0), poly_f64x2);
    return vaddq_f64(exponent_f64x2, vmulq_n_f64(ln_m_f64x2, NUMKONG_F64_LOG2E_));
}

#if NUMKONG_TARGET_NEON

NUMKONG_API nk_status_t nk_kld_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                        nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t epsilon = NUMKONG_F32_DIVISION_EPSILON;
    float32x4_t epsilon_f32x4 = vdupq_n_f32(epsilon);
    float64x2_t sum_low_f64x2 = vdupq_n_f64(0.0);
    float64x2_t sum_high_f64x2 = vdupq_n_f64(0.0);
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

    float32x4_t ratio_f32x4 = vdivq_f32(vmaxq_f32(a_f32x4, epsilon_f32x4), vmaxq_f32(b_f32x4, epsilon_f32x4));
    float32x4_t log_ratio_f32x4 = nk_log2_f32x4_neon_(ratio_f32x4);
    float32x4_t contribution_f32x4 = vmulq_f32(a_f32x4, log_ratio_f32x4);
    sum_low_f64x2 = vaddq_f64(sum_low_f64x2, vcvt_f64_f32(vget_low_f32(contribution_f32x4)));
    sum_high_f64x2 = vaddq_f64(sum_high_f64x2, vcvt_high_f64_f32(contribution_f32x4));
    if (n != 0) goto nk_kld_f32_neon_cycle;

    nk_f64_t log2_normalizer = NUMKONG_F64_LN2_;
    nk_f64_t sum = vaddvq_f64(vaddq_f64(sum_low_f64x2, sum_high_f64x2)) * log2_normalizer;
    *result = sum;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_jsd_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                        nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t epsilon = NUMKONG_F32_DIVISION_EPSILON;
    float32x4_t epsilon_f32x4 = vdupq_n_f32(epsilon);
    float64x2_t sum_low_f64x2 = vdupq_n_f64(0.0);
    float64x2_t sum_high_f64x2 = vdupq_n_f64(0.0);
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
    float32x4_t ratio_a_f32x4 = vdivq_f32(vmaxq_f32(a_f32x4, epsilon_f32x4), vmaxq_f32(mean_f32x4, epsilon_f32x4));
    float32x4_t ratio_b_f32x4 = vdivq_f32(vmaxq_f32(b_f32x4, epsilon_f32x4), vmaxq_f32(mean_f32x4, epsilon_f32x4));
    float32x4_t log_ratio_a_f32x4 = nk_log2_f32x4_neon_(ratio_a_f32x4);
    float32x4_t log_ratio_b_f32x4 = nk_log2_f32x4_neon_(ratio_b_f32x4);
    float32x4_t contribution_a_f32x4 = vmulq_f32(a_f32x4, log_ratio_a_f32x4);
    float32x4_t contribution_b_f32x4 = vmulq_f32(b_f32x4, log_ratio_b_f32x4);
    float32x4_t contribution_f32x4 = vaddq_f32(contribution_a_f32x4, contribution_b_f32x4);
    sum_low_f64x2 = vaddq_f64(sum_low_f64x2, vcvt_f64_f32(vget_low_f32(contribution_f32x4)));
    sum_high_f64x2 = vaddq_f64(sum_high_f64x2, vcvt_high_f64_f32(contribution_f32x4));
    if (n != 0) goto nk_jsd_f32_neon_cycle;

    nk_f64_t log2_normalizer = NUMKONG_F64_LN2_;
    nk_f64_t sum = vaddvq_f64(vaddq_f64(sum_low_f64x2, sum_high_f64x2)) * log2_normalizer / 2.0;
    *result = sum < 0 ? 0 : vget_lane_f64(vsqrt_f64(vdup_n_f64(sum)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_kld_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                        nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    float64x2_t epsilon_f64x2 = vdupq_n_f64(NUMKONG_F64_DIVISION_EPSILON);
    float64x2_t sum_f64x2 = vdupq_n_f64(0.0), compensation_f64x2 = vdupq_n_f64(0.0);
    float64x2_t a_f64x2, b_f64x2;

nk_kld_f64_neon_cycle:
    if (n < 2) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b64x2_serial_(a, &a_vec, n);
        nk_partial_load_b64x2_serial_(b, &b_vec, n);
        a_f64x2 = a_vec.f64x2;
        b_f64x2 = b_vec.f64x2;
        n = 0;
    }
    else {
        a_f64x2 = vld1q_f64(a);
        b_f64x2 = vld1q_f64(b);
        n -= 2, a += 2, b += 2;
    }

    float64x2_t ratio_f64x2 = vdivq_f64(vmaxq_f64(a_f64x2, epsilon_f64x2), vmaxq_f64(b_f64x2, epsilon_f64x2));
    float64x2_t contribution_f64x2 = vmulq_f64(a_f64x2, nk_log2_f64x2_neon_(ratio_f64x2));
    // Kahan compensated summation
    float64x2_t compensated_f64x2 = vsubq_f64(contribution_f64x2, compensation_f64x2);
    float64x2_t tentative_f64x2 = vaddq_f64(sum_f64x2, compensated_f64x2);
    compensation_f64x2 = vsubq_f64(vsubq_f64(tentative_f64x2, sum_f64x2), compensated_f64x2);
    sum_f64x2 = tentative_f64x2;
    if (n != 0) goto nk_kld_f64_neon_cycle;

    *result = vaddvq_f64(vsubq_f64(sum_f64x2, compensation_f64x2)) * NUMKONG_F64_LN2_;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_jsd_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                        nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    float64x2_t epsilon_f64x2 = vdupq_n_f64(NUMKONG_F64_DIVISION_EPSILON);
    float64x2_t sum_f64x2 = vdupq_n_f64(0.0), compensation_f64x2 = vdupq_n_f64(0.0);
    float64x2_t a_f64x2, b_f64x2;

nk_jsd_f64_neon_cycle:
    if (n < 2) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b64x2_serial_(a, &a_vec, n);
        nk_partial_load_b64x2_serial_(b, &b_vec, n);
        a_f64x2 = a_vec.f64x2;
        b_f64x2 = b_vec.f64x2;
        n = 0;
    }
    else {
        a_f64x2 = vld1q_f64(a);
        b_f64x2 = vld1q_f64(b);
        n -= 2, a += 2, b += 2;
    }

    float64x2_t mean_f64x2 = vmaxq_f64(vmulq_n_f64(vaddq_f64(a_f64x2, b_f64x2), 0.5), epsilon_f64x2);
    float64x2_t ratio_a_f64x2 = vdivq_f64(vmaxq_f64(a_f64x2, epsilon_f64x2), mean_f64x2);
    float64x2_t ratio_b_f64x2 = vdivq_f64(vmaxq_f64(b_f64x2, epsilon_f64x2), mean_f64x2);
    float64x2_t contribution_a_f64x2 = vmulq_f64(a_f64x2, nk_log2_f64x2_neon_(ratio_a_f64x2));
    float64x2_t contribution_b_f64x2 = vmulq_f64(b_f64x2, nk_log2_f64x2_neon_(ratio_b_f64x2));
    // Kahan compensated summation for contribution a
    float64x2_t compensated_a_f64x2 = vsubq_f64(contribution_a_f64x2, compensation_f64x2);
    float64x2_t tentative_a_f64x2 = vaddq_f64(sum_f64x2, compensated_a_f64x2);
    compensation_f64x2 = vsubq_f64(vsubq_f64(tentative_a_f64x2, sum_f64x2), compensated_a_f64x2);
    sum_f64x2 = tentative_a_f64x2;
    // Kahan compensated summation for contribution b
    float64x2_t compensated_b_f64x2 = vsubq_f64(contribution_b_f64x2, compensation_f64x2);
    float64x2_t tentative_b_f64x2 = vaddq_f64(sum_f64x2, compensated_b_f64x2);
    compensation_f64x2 = vsubq_f64(vsubq_f64(tentative_b_f64x2, sum_f64x2), compensated_b_f64x2);
    sum_f64x2 = tentative_b_f64x2;
    if (n != 0) goto nk_jsd_f64_neon_cycle;

    nk_f64_t sum = vaddvq_f64(vsubq_f64(sum_f64x2, compensation_f64x2)) * NUMKONG_F64_LN2_ / 2.0;
    *result = sum < 0 ? 0 : vget_lane_f64(vsqrt_f64(vdup_n_f64(sum)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_kld_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                        nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_f32_t epsilon = NUMKONG_F32_DIVISION_EPSILON;
    float32x4_t epsilon_f32x4 = vdupq_n_f32(epsilon);
    float32x4_t a_low_f32x4, a_high_f32x4, b_low_f32x4, b_high_f32x4;

nk_kld_f16_neon_cycle:
    if (n < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b16x8_serial_(a, &a_vec, n);
        nk_partial_load_b16x8_serial_(b, &b_vec, n);
        float16x8_t a_f16x8 = vreinterpretq_f16_u16(a_vec.u16x8);
        float16x8_t b_f16x8 = vreinterpretq_f16_u16(b_vec.u16x8);
        a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
        a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
        b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
        b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
        n = 0;
    }
    else {
        float16x8_t a_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)a));
        float16x8_t b_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)b));
        a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
        a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
        b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
        b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
        n -= 8, a += 8, b += 8;
    }

    float32x4_t ratio_low_f32x4 = vdivq_f32(vmaxq_f32(a_low_f32x4, epsilon_f32x4),
                                            vmaxq_f32(b_low_f32x4, epsilon_f32x4));
    float32x4_t ratio_high_f32x4 = vdivq_f32(vmaxq_f32(a_high_f32x4, epsilon_f32x4),
                                             vmaxq_f32(b_high_f32x4, epsilon_f32x4));
    float32x4_t log_ratio_low_f32x4 = nk_log2_f32x4_neon_(ratio_low_f32x4);
    float32x4_t log_ratio_high_f32x4 = nk_log2_f32x4_neon_(ratio_high_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_low_f32x4, log_ratio_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_high_f32x4, log_ratio_high_f32x4);
    if (n) goto nk_kld_f16_neon_cycle;

    nk_f32_t log2_normalizer = NUMKONG_F32_LN2_;
    nk_f32_t sum = vaddvq_f32(sum_f32x4) * log2_normalizer;
    *result = sum;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_jsd_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                        nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_f32_t epsilon = NUMKONG_F32_DIVISION_EPSILON;
    float32x4_t epsilon_f32x4 = vdupq_n_f32(epsilon);
    float32x4_t a_low_f32x4, a_high_f32x4, b_low_f32x4, b_high_f32x4;

nk_jsd_f16_neon_cycle:
    if (n < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b16x8_serial_(a, &a_vec, n);
        nk_partial_load_b16x8_serial_(b, &b_vec, n);
        float16x8_t a_f16x8 = vreinterpretq_f16_u16(a_vec.u16x8);
        float16x8_t b_f16x8 = vreinterpretq_f16_u16(b_vec.u16x8);
        a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
        a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
        b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
        b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
        n = 0;
    }
    else {
        float16x8_t a_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)a));
        float16x8_t b_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)b));
        a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
        a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
        b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
        b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
        n -= 8, a += 8, b += 8;
    }

    float32x4_t mean_low_f32x4 = vmulq_n_f32(vaddq_f32(a_low_f32x4, b_low_f32x4), 0.5f);
    float32x4_t mean_high_f32x4 = vmulq_n_f32(vaddq_f32(a_high_f32x4, b_high_f32x4), 0.5f);
    float32x4_t ratio_a_low_f32x4 = vdivq_f32(vmaxq_f32(a_low_f32x4, epsilon_f32x4),
                                              vmaxq_f32(mean_low_f32x4, epsilon_f32x4));
    float32x4_t ratio_a_high_f32x4 = vdivq_f32(vmaxq_f32(a_high_f32x4, epsilon_f32x4),
                                               vmaxq_f32(mean_high_f32x4, epsilon_f32x4));
    float32x4_t ratio_b_low_f32x4 = vdivq_f32(vmaxq_f32(b_low_f32x4, epsilon_f32x4),
                                              vmaxq_f32(mean_low_f32x4, epsilon_f32x4));
    float32x4_t ratio_b_high_f32x4 = vdivq_f32(vmaxq_f32(b_high_f32x4, epsilon_f32x4),
                                               vmaxq_f32(mean_high_f32x4, epsilon_f32x4));
    float32x4_t log_ratio_a_low_f32x4 = nk_log2_f32x4_neon_(ratio_a_low_f32x4);
    float32x4_t log_ratio_a_high_f32x4 = nk_log2_f32x4_neon_(ratio_a_high_f32x4);
    float32x4_t log_ratio_b_low_f32x4 = nk_log2_f32x4_neon_(ratio_b_low_f32x4);
    float32x4_t log_ratio_b_high_f32x4 = nk_log2_f32x4_neon_(ratio_b_high_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_low_f32x4, log_ratio_a_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_high_f32x4, log_ratio_a_high_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, b_low_f32x4, log_ratio_b_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, b_high_f32x4, log_ratio_b_high_f32x4);
    if (n) goto nk_jsd_f16_neon_cycle;

    nk_f32_t log2_normalizer = NUMKONG_F32_LN2_;
    nk_f32_t sum = vaddvq_f32(sum_f32x4) * log2_normalizer / 2;
    *result = sum < 0 ? 0 : vget_lane_f32(vsqrt_f32(vdup_n_f32(sum)), 0);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_NEON

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ARM64_NEON_
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_PROBABILITY_NEON_H
