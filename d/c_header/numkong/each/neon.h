/**
 *  @brief SIMD-accelerated Elementwise Arithmetic for NEON.
 *  @file include/numkong/each/neon.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/each.h
 *
 *  @section elementwise_neon_instructions ARM NEON Instructions
 *
 *      Intrinsic         Instruction                   Latency     Throughput
 *                                                                  A76     M4+/V1+/Oryon
 *      vld1q_f32         LD1 (V.4S)                    4cy         2/cy    2/cy
 *      vst1q_f32         ST1 (V.4S)                    2cy         2/cy    2/cy
 *      vaddq_f32         FADD (V.4S, V.4S, V.4S)       2cy         2/cy    4/cy
 *      vmulq_f32         FMUL (V.4S, V.4S, V.4S)       3cy         2/cy    4/cy
 *      vfmaq_f32         FMLA (V.4S, V.4S, V.4S)       4cy         2/cy    4/cy
 *      vaddq_f64         FADD (V.2D, V.2D, V.2D)       2cy         2/cy    4/cy
 *      vmulq_f64         FMUL (V.2D, V.2D, V.2D)       3cy         2/cy    4/cy
 *      vfmaq_f64         FMLA (V.2D, V.2D, V.2D)       4cy         2/cy    4/cy
 *      vqaddq_s16        SQADD (V.8H, V.8H, V.8H)      2cy         2/cy    4/cy
 *      vcvtq_f32_s32     SCVTF (V.4S, V.4S)            3cy         2/cy    2/cy
 *      vcvtnq_s32_f32    FCVTNS (V.4S, V.4S)           3cy         2/cy    2/cy
 *      vqmovn_s32        SQXTN (V.4H, V.4S)            3cy         2/cy    2/cy
 *
 *  Elementwise operations are throughput-bound rather than latency-bound. FP arithmetic
 *  throughput doubles on 4-pipe cores (Apple M4+, Graviton3+, Oryon) from 2/cy to 4/cy.
 *
 *  Memory bandwidth (LD1/ST1) typically becomes the bottleneck for large arrays, as load/store
 *  throughput remains at 2/cy across all cores.
 */
#ifndef NK_EACH_NEON_H
#define NK_EACH_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/types.h"
#include "numkong/cast/neon.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

NK_PUBLIC void nk_each_sum_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4 = vld1q_f32(a + i);
        float32x4_t b_f32x4 = vld1q_f32(b + i);
        float32x4_t sum_f32x4 = vaddq_f32(a_f32x4, b_f32x4);
        vst1q_f32(result + i, sum_f32x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

NK_PUBLIC void nk_each_scale_f32_neon(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                      nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    float32x4_t beta_f32x4 = vdupq_n_f32(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4 = vld1q_f32(a + i);
        float32x4_t result_f32x4 = vfmaq_n_f32(beta_f32x4, a_f32x4, alpha_val);
        vst1q_f32(result + i, result_f32x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val;
}

NK_PUBLIC void nk_each_blend_f32_neon(                 //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_each_sum_f32_neon(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_f32_neon(a, n, alpha, &zero, result); }
        else { nk_each_scale_f32_neon(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4 = vld1q_f32(a + i);
        float32x4_t b_f32x4 = vld1q_f32(b + i);
        float32x4_t a_scaled_f32x4 = vmulq_n_f32(a_f32x4, alpha_val);
        float32x4_t result_f32x4 = vfmaq_n_f32(a_scaled_f32x4, b_f32x4, beta_val);
        vst1q_f32(result + i, result_f32x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val * b[i];
}

NK_PUBLIC void nk_each_fma_f32_neon(                         //
    nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_f32x4 = vld1q_f32(a + i);
        float32x4_t b_f32x4 = vld1q_f32(b + i);
        float32x4_t c_f32x4 = vld1q_f32(c + i);
        float32x4_t ab_f32x4 = vmulq_f32(a_f32x4, b_f32x4);
        float32x4_t ab_scaled_f32x4 = vmulq_n_f32(ab_f32x4, alpha_val);
        float32x4_t result_f32x4 = vfmaq_n_f32(ab_scaled_f32x4, c_f32x4, beta_val);
        vst1q_f32(result + i, result_f32x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] * b[i] + beta_val * c[i];
}

NK_PUBLIC void nk_each_sum_i16_neon(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_i16_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int16x8_t a_s16x8 = vld1q_s16(a + i);
        int16x8_t b_s16x8 = vld1q_s16(b + i);
        int16x8_t sum_s16x8 = vqaddq_s16(a_s16x8, b_s16x8);
        vst1q_s16(result + i, sum_s16x8);
    }

    // The tail:
    for (; i < n; ++i) result[i] = nk_i16_saturating_add_serial(a[i], b[i]);
}

NK_PUBLIC void nk_each_scale_i16_neon(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                      nk_i16_t *result) {
    float32_t alpha_f32 = *alpha;
    float32_t beta_f32 = *beta;
    float32x4_t alpha_f32x4 = vdupq_n_f32(alpha_f32);
    float32x4_t beta_f32x4 = vdupq_n_f32(beta_f32);
    float32x4_t min_f32x4 = vdupq_n_f32(-32768.0f);
    float32x4_t max_f32x4 = vdupq_n_f32(32767.0f);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int16x4_t a_i16x4 = vld1_s16(a + i);
        float32x4_t a_f32x4 = vcvtq_f32_s32(vmovl_s16(a_i16x4));
        float32x4_t result_f32x4 = vfmaq_f32(beta_f32x4, a_f32x4, alpha_f32x4);
        result_f32x4 = vmaxq_f32(vminq_f32(result_f32x4, max_f32x4), min_f32x4);
        int16x4_t result_i16x4 = vqmovn_s32(vcvtnq_s32_f32(result_f32x4));
        vst1_s16(result + i, result_i16x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f32 * a[i] + beta_f32;
        nk_f32_to_i16_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_fma_i16_neon(                         //
    nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result) {
    float32_t alpha_f32 = *alpha;
    float32_t beta_f32 = *beta;
    float32x4_t min_f32x4 = vdupq_n_f32(-32768.0f);
    float32x4_t max_f32x4 = vdupq_n_f32(32767.0f);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int16x4_t a_i16x4 = vld1_s16(a + i);
        int16x4_t b_i16x4 = vld1_s16(b + i);
        int16x4_t c_i16x4 = vld1_s16(c + i);
        float32x4_t a_f32x4 = vcvtq_f32_s32(vmovl_s16(a_i16x4));
        float32x4_t b_f32x4 = vcvtq_f32_s32(vmovl_s16(b_i16x4));
        float32x4_t c_f32x4 = vcvtq_f32_s32(vmovl_s16(c_i16x4));
        float32x4_t ab_f32x4 = vmulq_f32(a_f32x4, b_f32x4);
        float32x4_t ab_scaled_f32x4 = vmulq_n_f32(ab_f32x4, alpha_f32);
        float32x4_t result_f32x4 = vfmaq_n_f32(ab_scaled_f32x4, c_f32x4, beta_f32);
        result_f32x4 = vmaxq_f32(vminq_f32(result_f32x4, max_f32x4), min_f32x4);
        int16x4_t result_i16x4 = vqmovn_s32(vcvtnq_s32_f32(result_f32x4));
        vst1_s16(result + i, result_i16x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f32 * a[i] * b[i] + beta_f32 * c[i];
        nk_f32_to_i16_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_sum_u16_neon(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_u16_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t a_u16x8 = vld1q_u16(a + i);
        uint16x8_t b_u16x8 = vld1q_u16(b + i);
        uint16x8_t sum_u16x8 = vqaddq_u16(a_u16x8, b_u16x8);
        vst1q_u16(result + i, sum_u16x8);
    }

    // The tail:
    for (; i < n; ++i) result[i] = nk_u16_saturating_add_serial(a[i], b[i]);
}

NK_PUBLIC void nk_each_scale_u16_neon(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                      nk_u16_t *result) {
    float32_t alpha_f32 = *alpha;
    float32_t beta_f32 = *beta;
    float32x4_t alpha_f32x4 = vdupq_n_f32(alpha_f32);
    float32x4_t beta_f32x4 = vdupq_n_f32(beta_f32);
    float32x4_t min_f32x4 = vdupq_n_f32(0.0f);
    float32x4_t max_f32x4 = vdupq_n_f32(65535.0f);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint16x4_t a_u16x4 = vld1_u16(a + i);
        float32x4_t a_f32x4 = vcvtq_f32_u32(vmovl_u16(a_u16x4));
        float32x4_t result_f32x4 = vfmaq_f32(beta_f32x4, a_f32x4, alpha_f32x4);
        result_f32x4 = vmaxq_f32(vminq_f32(result_f32x4, max_f32x4), min_f32x4);
        uint16x4_t result_u16x4 = vqmovn_u32(vcvtnq_u32_f32(result_f32x4));
        vst1_u16(result + i, result_u16x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f32 * a[i] + beta_f32;
        nk_f32_to_u16_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_fma_u16_neon(                         //
    nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result) {
    float32_t alpha_f32 = *alpha;
    float32_t beta_f32 = *beta;
    float32x4_t min_f32x4 = vdupq_n_f32(0.0f);
    float32x4_t max_f32x4 = vdupq_n_f32(65535.0f);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint16x4_t a_u16x4 = vld1_u16(a + i);
        uint16x4_t b_u16x4 = vld1_u16(b + i);
        uint16x4_t c_u16x4 = vld1_u16(c + i);
        float32x4_t a_f32x4 = vcvtq_f32_u32(vmovl_u16(a_u16x4));
        float32x4_t b_f32x4 = vcvtq_f32_u32(vmovl_u16(b_u16x4));
        float32x4_t c_f32x4 = vcvtq_f32_u32(vmovl_u16(c_u16x4));
        float32x4_t ab_f32x4 = vmulq_f32(a_f32x4, b_f32x4);
        float32x4_t ab_scaled_f32x4 = vmulq_n_f32(ab_f32x4, alpha_f32);
        float32x4_t result_f32x4 = vfmaq_n_f32(ab_scaled_f32x4, c_f32x4, beta_f32);
        result_f32x4 = vmaxq_f32(vminq_f32(result_f32x4, max_f32x4), min_f32x4);
        uint16x4_t result_u16x4 = vqmovn_u32(vcvtnq_u32_f32(result_f32x4));
        vst1_u16(result + i, result_u16x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_f32 * a[i] * b[i] + beta_f32 * c[i];
        nk_f32_to_u16_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_sum_i32_neon(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_i32_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t a_s32x4 = vld1q_s32(a + i);
        int32x4_t b_s32x4 = vld1q_s32(b + i);
        int32x4_t sum_s32x4 = vqaddq_s32(a_s32x4, b_s32x4);
        vst1q_s32(result + i, sum_s32x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = nk_i32_saturating_add_serial(a[i], b[i]);
}

NK_PUBLIC void nk_each_scale_i32_neon(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                      nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t alpha_f64x2 = vdupq_n_f64(alpha_val);
    float64x2_t beta_f64x2 = vdupq_n_f64(beta_val);
    float64x2_t min_f64x2 = vdupq_n_f64(-2147483648.0);
    float64x2_t max_f64x2 = vdupq_n_f64(2147483647.0);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int32x2_t a_i32x2 = vld1_s32(a + i);
        float64x2_t a_f64x2 = vcvtq_f64_s64(vmovl_s32(a_i32x2));
        float64x2_t result_f64x2 = vfmaq_f64(beta_f64x2, a_f64x2, alpha_f64x2);
        result_f64x2 = vmaxq_f64(vminq_f64(result_f64x2, max_f64x2), min_f64x2);
        int32x2_t result_i32x2 = vqmovn_s64(vcvtnq_s64_f64(result_f64x2));
        vst1_s32(result + i, result_i32x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] + beta_val;
        nk_f64_to_i32_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_fma_i32_neon(                         //
    nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c, //
    nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t min_f64x2 = vdupq_n_f64(-2147483648.0);
    float64x2_t max_f64x2 = vdupq_n_f64(2147483647.0);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int32x2_t a_i32x2 = vld1_s32(a + i);
        int32x2_t b_i32x2 = vld1_s32(b + i);
        int32x2_t c_i32x2 = vld1_s32(c + i);
        float64x2_t a_f64x2 = vcvtq_f64_s64(vmovl_s32(a_i32x2));
        float64x2_t b_f64x2 = vcvtq_f64_s64(vmovl_s32(b_i32x2));
        float64x2_t c_f64x2 = vcvtq_f64_s64(vmovl_s32(c_i32x2));
        float64x2_t ab_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
        float64x2_t ab_scaled_f64x2 = vmulq_n_f64(ab_f64x2, alpha_val);
        float64x2_t result_f64x2 = vfmaq_n_f64(ab_scaled_f64x2, c_f64x2, beta_val);
        result_f64x2 = vmaxq_f64(vminq_f64(result_f64x2, max_f64x2), min_f64x2);
        int32x2_t result_i32x2 = vqmovn_s64(vcvtnq_s64_f64(result_f64x2));
        vst1_s32(result + i, result_i32x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] * b[i] + beta_val * c[i];
        nk_f64_to_i32_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_sum_u32_neon(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_u32_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint32x4_t a_u32x4 = vld1q_u32(a + i);
        uint32x4_t b_u32x4 = vld1q_u32(b + i);
        uint32x4_t sum_u32x4 = vqaddq_u32(a_u32x4, b_u32x4);
        vst1q_u32(result + i, sum_u32x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = nk_u32_saturating_add_serial(a[i], b[i]);
}

NK_PUBLIC void nk_each_scale_u32_neon(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                      nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t alpha_f64x2 = vdupq_n_f64(alpha_val);
    float64x2_t beta_f64x2 = vdupq_n_f64(beta_val);
    float64x2_t min_f64x2 = vdupq_n_f64(0.0);
    float64x2_t max_f64x2 = vdupq_n_f64(4294967295.0);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint32x2_t a_u32x2 = vld1_u32(a + i);
        float64x2_t a_f64x2 = vcvtq_f64_u64(vmovl_u32(a_u32x2));
        float64x2_t result_f64x2 = vfmaq_f64(beta_f64x2, a_f64x2, alpha_f64x2);
        result_f64x2 = vmaxq_f64(vminq_f64(result_f64x2, max_f64x2), min_f64x2);
        uint32x2_t result_u32x2 = vqmovn_u64(vcvtnq_u64_f64(result_f64x2));
        vst1_u32(result + i, result_u32x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] + beta_val;
        nk_f64_to_u32_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_fma_u32_neon(                         //
    nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c, //
    nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t min_f64x2 = vdupq_n_f64(0.0);
    float64x2_t max_f64x2 = vdupq_n_f64(4294967295.0);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint32x2_t a_u32x2 = vld1_u32(a + i);
        uint32x2_t b_u32x2 = vld1_u32(b + i);
        uint32x2_t c_u32x2 = vld1_u32(c + i);
        float64x2_t a_f64x2 = vcvtq_f64_u64(vmovl_u32(a_u32x2));
        float64x2_t b_f64x2 = vcvtq_f64_u64(vmovl_u32(b_u32x2));
        float64x2_t c_f64x2 = vcvtq_f64_u64(vmovl_u32(c_u32x2));
        float64x2_t ab_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
        float64x2_t ab_scaled_f64x2 = vmulq_n_f64(ab_f64x2, alpha_val);
        float64x2_t result_f64x2 = vfmaq_n_f64(ab_scaled_f64x2, c_f64x2, beta_val);
        result_f64x2 = vmaxq_f64(vminq_f64(result_f64x2, max_f64x2), min_f64x2);
        uint32x2_t result_u32x2 = vqmovn_u64(vcvtnq_u64_f64(result_f64x2));
        vst1_u32(result + i, result_u32x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] * b[i] + beta_val * c[i];
        nk_f64_to_u32_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_sum_i64_neon(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_i64_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t a_s64x2 = vld1q_s64(a + i);
        int64x2_t b_s64x2 = vld1q_s64(b + i);
        int64x2_t sum_s64x2 = vqaddq_s64(a_s64x2, b_s64x2);
        vst1q_s64(result + i, sum_s64x2);
    }

    // The tail:
    for (; i < n; ++i) result[i] = nk_i64_saturating_add_serial(a[i], b[i]);
}

NK_PUBLIC void nk_each_scale_i64_neon(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                      nk_i64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t alpha_f64x2 = vdupq_n_f64(alpha_val);
    float64x2_t beta_f64x2 = vdupq_n_f64(beta_val);
    float64x2_t min_f64x2 = vdupq_n_f64((nk_f64_t)NK_I64_MIN);
    float64x2_t max_f64x2 = vdupq_n_f64((nk_f64_t)NK_I64_MAX);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t a_i64x2 = vld1q_s64(a + i);
        float64x2_t a_f64x2 = vcvtq_f64_s64(a_i64x2);
        float64x2_t result_f64x2 = vfmaq_f64(beta_f64x2, a_f64x2, alpha_f64x2);
        result_f64x2 = vmaxq_f64(vminq_f64(result_f64x2, max_f64x2), min_f64x2);
        int64x2_t result_i64x2 = vcvtnq_s64_f64(result_f64x2);
        vst1q_s64(result + i, result_i64x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] + beta_val;
        nk_f64_to_i64_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_fma_i64_neon(                         //
    nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c, //
    nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_i64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t min_f64x2 = vdupq_n_f64((nk_f64_t)NK_I64_MIN);
    float64x2_t max_f64x2 = vdupq_n_f64((nk_f64_t)NK_I64_MAX);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t a_i64x2 = vld1q_s64(a + i);
        int64x2_t b_i64x2 = vld1q_s64(b + i);
        int64x2_t c_i64x2 = vld1q_s64(c + i);
        float64x2_t a_f64x2 = vcvtq_f64_s64(a_i64x2);
        float64x2_t b_f64x2 = vcvtq_f64_s64(b_i64x2);
        float64x2_t c_f64x2 = vcvtq_f64_s64(c_i64x2);
        float64x2_t ab_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
        float64x2_t ab_scaled_f64x2 = vmulq_n_f64(ab_f64x2, alpha_val);
        float64x2_t result_f64x2 = vfmaq_n_f64(ab_scaled_f64x2, c_f64x2, beta_val);
        result_f64x2 = vmaxq_f64(vminq_f64(result_f64x2, max_f64x2), min_f64x2);
        int64x2_t result_i64x2 = vcvtnq_s64_f64(result_f64x2);
        vst1q_s64(result + i, result_i64x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] * b[i] + beta_val * c[i];
        nk_f64_to_i64_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_sum_u64_neon(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_u64_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint64x2_t a_u64x2 = vld1q_u64(a + i);
        uint64x2_t b_u64x2 = vld1q_u64(b + i);
        uint64x2_t sum_u64x2 = vqaddq_u64(a_u64x2, b_u64x2);
        vst1q_u64(result + i, sum_u64x2);
    }

    // The tail:
    for (; i < n; ++i) result[i] = nk_u64_saturating_add_serial(a[i], b[i]);
}

NK_PUBLIC void nk_each_scale_u64_neon(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                      nk_u64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t alpha_f64x2 = vdupq_n_f64(alpha_val);
    float64x2_t beta_f64x2 = vdupq_n_f64(beta_val);
    float64x2_t min_f64x2 = vdupq_n_f64(0.0);
    float64x2_t max_f64x2 = vdupq_n_f64((nk_f64_t)NK_U64_MAX);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint64x2_t a_u64x2 = vld1q_u64(a + i);
        float64x2_t a_f64x2 = vcvtq_f64_u64(a_u64x2);
        float64x2_t result_f64x2 = vfmaq_f64(beta_f64x2, a_f64x2, alpha_f64x2);
        result_f64x2 = vmaxq_f64(vminq_f64(result_f64x2, max_f64x2), min_f64x2);
        uint64x2_t result_u64x2 = vcvtnq_u64_f64(result_f64x2);
        vst1q_u64(result + i, result_u64x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] + beta_val;
        nk_f64_to_u64_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_fma_u64_neon(                         //
    nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c, //
    nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_u64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t min_f64x2 = vdupq_n_f64(0.0);
    float64x2_t max_f64x2 = vdupq_n_f64((nk_f64_t)NK_U64_MAX);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint64x2_t a_u64x2 = vld1q_u64(a + i);
        uint64x2_t b_u64x2 = vld1q_u64(b + i);
        uint64x2_t c_u64x2 = vld1q_u64(c + i);
        float64x2_t a_f64x2 = vcvtq_f64_u64(a_u64x2);
        float64x2_t b_f64x2 = vcvtq_f64_u64(b_u64x2);
        float64x2_t c_f64x2 = vcvtq_f64_u64(c_u64x2);
        float64x2_t ab_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
        float64x2_t ab_scaled_f64x2 = vmulq_n_f64(ab_f64x2, alpha_val);
        float64x2_t result_f64x2 = vfmaq_n_f64(ab_scaled_f64x2, c_f64x2, beta_val);
        result_f64x2 = vmaxq_f64(vminq_f64(result_f64x2, max_f64x2), min_f64x2);
        uint64x2_t result_u64x2 = vcvtnq_u64_f64(result_f64x2);
        vst1q_u64(result + i, result_u64x2);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t sum = alpha_val * a[i] * b[i] + beta_val * c[i];
        nk_f64_to_u64_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_sum_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t a_f64x2 = vld1q_f64(a + i);
        float64x2_t b_f64x2 = vld1q_f64(b + i);
        float64x2_t sum_f64x2 = vaddq_f64(a_f64x2, b_f64x2);
        vst1q_f64(result + i, sum_f64x2);
    }

    // The tail:
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

NK_PUBLIC void nk_each_scale_f64_neon(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                      nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    float64x2_t alpha_f64x2 = vdupq_n_f64(alpha_val);
    float64x2_t beta_f64x2 = vdupq_n_f64(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t a_f64x2 = vld1q_f64(a + i);
        float64x2_t result_f64x2 = vfmaq_f64(beta_f64x2, a_f64x2, alpha_f64x2);
        vst1q_f64(result + i, result_f64x2);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val;
}

NK_PUBLIC void nk_each_blend_f64_neon(                 //
    nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result) {

    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // There are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_each_sum_f64_neon(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f64_t zero = 0;
        if (beta_val == 0) { nk_each_scale_f64_neon(a, n, alpha, &zero, result); }
        else { nk_each_scale_f64_neon(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t a_f64x2 = vld1q_f64(a + i);
        float64x2_t b_f64x2 = vld1q_f64(b + i);
        float64x2_t a_scaled_f64x2 = vmulq_n_f64(a_f64x2, alpha_val);
        float64x2_t b_scaled_f64x2 = vmulq_n_f64(b_f64x2, beta_val);
        float64x2_t result_f64x2 = vaddq_f64(a_scaled_f64x2, b_scaled_f64x2);
        vst1q_f64(result + i, result_f64x2);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val * b[i];
}

NK_PUBLIC void nk_each_fma_f64_neon(                         //
    nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, //
    nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // The main loop:
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t a_f64x2 = vld1q_f64(a + i);
        float64x2_t b_f64x2 = vld1q_f64(b + i);
        float64x2_t c_f64x2 = vld1q_f64(c + i);
        float64x2_t ab_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
        float64x2_t ab_scaled_f64x2 = vmulq_n_f64(ab_f64x2, alpha_val);
        float64x2_t result_f64x2 = vfmaq_n_f64(ab_scaled_f64x2, c_f64x2, beta_val);
        vst1q_f64(result + i, result_f64x2);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] * b[i] + beta_val * c[i];
}

NK_PUBLIC void nk_each_sum_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_e4m3_t *result) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(a + i));
        float16x8_t b_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(b + i));
        float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
        float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
        float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
        float32x4_t b_high_f32x4 = vcvt_f32_f16(vget_high_f16(b_f16x8));
        float32x4_t result_low_f32x4 = vaddq_f32(a_low_f32x4, b_low_f32x4);
        float32x4_t result_high_f32x4 = vaddq_f32(a_high_f32x4, b_high_f32x4);
        nk_b32_vec_t result_low_vec = nk_f32x4_to_e4m3x4_neon_(result_low_f32x4);
        nk_b32_vec_t result_high_vec = nk_f32x4_to_e4m3x4_neon_(result_high_f32x4);
        vst1_u8(result + i, vcreate_u8((nk_u64_t)result_low_vec.u32 | ((nk_u64_t)result_high_vec.u32 << 32)));
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi, sum;
        nk_e4m3_to_f32_serial(a + i, &ai);
        nk_e4m3_to_f32_serial(b + i, &bi);
        sum = ai + bi;
        nk_f32_to_e4m3_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_sum_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_e5m2_t *result) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8(a + i));
        float16x8_t b_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8(b + i));
        float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
        float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
        float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
        float32x4_t b_high_f32x4 = vcvt_f32_f16(vget_high_f16(b_f16x8));
        float32x4_t result_low_f32x4 = vaddq_f32(a_low_f32x4, b_low_f32x4);
        float32x4_t result_high_f32x4 = vaddq_f32(a_high_f32x4, b_high_f32x4);
        nk_b32_vec_t result_low_vec = nk_f32x4_to_e5m2x4_neon_(result_low_f32x4);
        nk_b32_vec_t result_high_vec = nk_f32x4_to_e5m2x4_neon_(result_high_f32x4);
        vst1_u8(result + i, vcreate_u8((nk_u64_t)result_low_vec.u32 | ((nk_u64_t)result_high_vec.u32 << 32)));
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi, sum;
        nk_e5m2_to_f32_serial(a + i, &ai);
        nk_e5m2_to_f32_serial(b + i, &bi);
        sum = ai + bi;
        nk_f32_to_e5m2_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_scale_e4m3_neon(nk_e4m3_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                       nk_e4m3_t *result) {
    float32x4_t alpha_f32x4 = vdupq_n_f32(*alpha);
    float32x4_t beta_f32x4 = vdupq_n_f32(*beta);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(a + i));
        float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
        float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
        float32x4_t result_low_f32x4 = vfmaq_f32(beta_f32x4, a_low_f32x4, alpha_f32x4);
        float32x4_t result_high_f32x4 = vfmaq_f32(beta_f32x4, a_high_f32x4, alpha_f32x4);
        nk_b32_vec_t result_low_vec = nk_f32x4_to_e4m3x4_neon_(result_low_f32x4);
        nk_b32_vec_t result_high_vec = nk_f32x4_to_e4m3x4_neon_(result_high_f32x4);
        vst1_u8(result + i, vcreate_u8((nk_u64_t)result_low_vec.u32 | ((nk_u64_t)result_high_vec.u32 << 32)));
    }
    for (; i < n; ++i) {
        nk_f32_t ai, scaled;
        nk_e4m3_to_f32_serial(a + i, &ai);
        scaled = *alpha * ai + *beta;
        nk_f32_to_e4m3_serial(&scaled, result + i);
    }
}

NK_PUBLIC void nk_each_scale_e5m2_neon(nk_e5m2_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                       nk_e5m2_t *result) {
    float32x4_t alpha_f32x4 = vdupq_n_f32(*alpha);
    float32x4_t beta_f32x4 = vdupq_n_f32(*beta);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8(a + i));
        float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
        float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
        float32x4_t result_low_f32x4 = vfmaq_f32(beta_f32x4, a_low_f32x4, alpha_f32x4);
        float32x4_t result_high_f32x4 = vfmaq_f32(beta_f32x4, a_high_f32x4, alpha_f32x4);
        nk_b32_vec_t result_low_vec = nk_f32x4_to_e5m2x4_neon_(result_low_f32x4);
        nk_b32_vec_t result_high_vec = nk_f32x4_to_e5m2x4_neon_(result_high_f32x4);
        vst1_u8(result + i, vcreate_u8((nk_u64_t)result_low_vec.u32 | ((nk_u64_t)result_high_vec.u32 << 32)));
    }
    for (; i < n; ++i) {
        nk_f32_t ai, scaled;
        nk_e5m2_to_f32_serial(a + i, &ai);
        scaled = *alpha * ai + *beta;
        nk_f32_to_e5m2_serial(&scaled, result + i);
    }
}

NK_PUBLIC void nk_each_blend_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                       nk_f32_t const *beta, nk_e4m3_t *result) {
    float32x4_t alpha_f32x4 = vdupq_n_f32(*alpha);
    float32x4_t beta_f32x4 = vdupq_n_f32(*beta);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(a + i));
        float16x8_t b_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(b + i));
        float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
        float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
        float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
        float32x4_t b_high_f32x4 = vcvt_f32_f16(vget_high_f16(b_f16x8));
        float32x4_t a_scaled_low_f32x4 = vmulq_f32(a_low_f32x4, alpha_f32x4);
        float32x4_t a_scaled_high_f32x4 = vmulq_f32(a_high_f32x4, alpha_f32x4);
        float32x4_t result_low_f32x4 = vfmaq_f32(a_scaled_low_f32x4, b_low_f32x4, beta_f32x4);
        float32x4_t result_high_f32x4 = vfmaq_f32(a_scaled_high_f32x4, b_high_f32x4, beta_f32x4);
        nk_b32_vec_t result_low_vec = nk_f32x4_to_e4m3x4_neon_(result_low_f32x4);
        nk_b32_vec_t result_high_vec = nk_f32x4_to_e4m3x4_neon_(result_high_f32x4);
        vst1_u8(result + i, vcreate_u8((nk_u64_t)result_low_vec.u32 | ((nk_u64_t)result_high_vec.u32 << 32)));
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi, blended;
        nk_e4m3_to_f32_serial(a + i, &ai);
        nk_e4m3_to_f32_serial(b + i, &bi);
        blended = *alpha * ai + *beta * bi;
        nk_f32_to_e4m3_serial(&blended, result + i);
    }
}

NK_PUBLIC void nk_each_blend_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                       nk_f32_t const *beta, nk_e5m2_t *result) {
    float32x4_t alpha_f32x4 = vdupq_n_f32(*alpha);
    float32x4_t beta_f32x4 = vdupq_n_f32(*beta);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8(a + i));
        float16x8_t b_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8(b + i));
        float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
        float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
        float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
        float32x4_t b_high_f32x4 = vcvt_f32_f16(vget_high_f16(b_f16x8));
        float32x4_t a_scaled_low_f32x4 = vmulq_f32(a_low_f32x4, alpha_f32x4);
        float32x4_t a_scaled_high_f32x4 = vmulq_f32(a_high_f32x4, alpha_f32x4);
        float32x4_t result_low_f32x4 = vfmaq_f32(a_scaled_low_f32x4, b_low_f32x4, beta_f32x4);
        float32x4_t result_high_f32x4 = vfmaq_f32(a_scaled_high_f32x4, b_high_f32x4, beta_f32x4);
        nk_b32_vec_t result_low_vec = nk_f32x4_to_e5m2x4_neon_(result_low_f32x4);
        nk_b32_vec_t result_high_vec = nk_f32x4_to_e5m2x4_neon_(result_high_f32x4);
        vst1_u8(result + i, vcreate_u8((nk_u64_t)result_low_vec.u32 | ((nk_u64_t)result_high_vec.u32 << 32)));
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi, blended;
        nk_e5m2_to_f32_serial(a + i, &ai);
        nk_e5m2_to_f32_serial(b + i, &bi);
        blended = *alpha * ai + *beta * bi;
        nk_f32_to_e5m2_serial(&blended, result + i);
    }
}

NK_PUBLIC void nk_each_fma_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_e4m3_t const *c, nk_size_t n,
                                     nk_f32_t const *alpha, nk_f32_t const *beta, nk_e4m3_t *result) {
    float32x4_t alpha_f32x4 = vdupq_n_f32(*alpha);
    float32x4_t beta_f32x4 = vdupq_n_f32(*beta);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(a + i));
        float16x8_t b_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(b + i));
        float16x8_t c_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(c + i));
        float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
        float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
        float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
        float32x4_t b_high_f32x4 = vcvt_f32_f16(vget_high_f16(b_f16x8));
        float32x4_t c_low_f32x4 = vcvt_f32_f16(vget_low_f16(c_f16x8));
        float32x4_t c_high_f32x4 = vcvt_f32_f16(vget_high_f16(c_f16x8));
        float32x4_t ab_low_f32x4 = vmulq_f32(a_low_f32x4, b_low_f32x4);
        float32x4_t ab_high_f32x4 = vmulq_f32(a_high_f32x4, b_high_f32x4);
        float32x4_t ab_scaled_low_f32x4 = vmulq_f32(ab_low_f32x4, alpha_f32x4);
        float32x4_t ab_scaled_high_f32x4 = vmulq_f32(ab_high_f32x4, alpha_f32x4);
        float32x4_t result_low_f32x4 = vfmaq_f32(ab_scaled_low_f32x4, c_low_f32x4, beta_f32x4);
        float32x4_t result_high_f32x4 = vfmaq_f32(ab_scaled_high_f32x4, c_high_f32x4, beta_f32x4);
        nk_b32_vec_t result_low_vec = nk_f32x4_to_e4m3x4_neon_(result_low_f32x4);
        nk_b32_vec_t result_high_vec = nk_f32x4_to_e4m3x4_neon_(result_high_f32x4);
        vst1_u8(result + i, vcreate_u8((nk_u64_t)result_low_vec.u32 | ((nk_u64_t)result_high_vec.u32 << 32)));
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi, ci, fma;
        nk_e4m3_to_f32_serial(a + i, &ai);
        nk_e4m3_to_f32_serial(b + i, &bi);
        nk_e4m3_to_f32_serial(c + i, &ci);
        fma = *alpha * ai * bi + *beta * ci;
        nk_f32_to_e4m3_serial(&fma, result + i);
    }
}

NK_PUBLIC void nk_each_fma_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_e5m2_t const *c, nk_size_t n,
                                     nk_f32_t const *alpha, nk_f32_t const *beta, nk_e5m2_t *result) {
    float32x4_t alpha_f32x4 = vdupq_n_f32(*alpha);
    float32x4_t beta_f32x4 = vdupq_n_f32(*beta);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8(a + i));
        float16x8_t b_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8(b + i));
        float16x8_t c_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8(c + i));
        float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
        float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
        float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
        float32x4_t b_high_f32x4 = vcvt_f32_f16(vget_high_f16(b_f16x8));
        float32x4_t c_low_f32x4 = vcvt_f32_f16(vget_low_f16(c_f16x8));
        float32x4_t c_high_f32x4 = vcvt_f32_f16(vget_high_f16(c_f16x8));
        float32x4_t ab_low_f32x4 = vmulq_f32(a_low_f32x4, b_low_f32x4);
        float32x4_t ab_high_f32x4 = vmulq_f32(a_high_f32x4, b_high_f32x4);
        float32x4_t ab_scaled_low_f32x4 = vmulq_f32(ab_low_f32x4, alpha_f32x4);
        float32x4_t ab_scaled_high_f32x4 = vmulq_f32(ab_high_f32x4, alpha_f32x4);
        float32x4_t result_low_f32x4 = vfmaq_f32(ab_scaled_low_f32x4, c_low_f32x4, beta_f32x4);
        float32x4_t result_high_f32x4 = vfmaq_f32(ab_scaled_high_f32x4, c_high_f32x4, beta_f32x4);
        nk_b32_vec_t result_low_vec = nk_f32x4_to_e5m2x4_neon_(result_low_f32x4);
        nk_b32_vec_t result_high_vec = nk_f32x4_to_e5m2x4_neon_(result_high_f32x4);
        vst1_u8(result + i, vcreate_u8((nk_u64_t)result_low_vec.u32 | ((nk_u64_t)result_high_vec.u32 << 32)));
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi, ci, fma;
        nk_e5m2_to_f32_serial(a + i, &ai);
        nk_e5m2_to_f32_serial(b + i, &bi);
        nk_e5m2_to_f32_serial(c + i, &ci);
        fma = *alpha * ai * bi + *beta * ci;
        nk_f32_to_e5m2_serial(&fma, result + i);
    }
}

NK_PUBLIC void nk_each_scale_f32c_neon(nk_f32c_t const *a, nk_size_t n, nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                       nk_f32c_t *result) {
    float32x4_t alpha_real_f32x4 = vdupq_n_f32(alpha->real);
    float32x4_t alpha_imag_f32x4 = vdupq_n_f32(alpha->imag);
    float32x4_t beta_real_f32x4 = vdupq_n_f32(beta->real);
    float32x4_t beta_imag_f32x4 = vdupq_n_f32(beta->imag);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t a_f32x4x2 = vld2q_f32((nk_f32_t const *)(a + i));
        float32x4_t y_real_f32x4 = vfmaq_f32(beta_real_f32x4, alpha_real_f32x4, a_f32x4x2.val[0]);
        y_real_f32x4 = vfmsq_f32(y_real_f32x4, alpha_imag_f32x4, a_f32x4x2.val[1]);
        float32x4_t y_imag_f32x4 = vfmaq_f32(beta_imag_f32x4, alpha_real_f32x4, a_f32x4x2.val[1]);
        y_imag_f32x4 = vfmaq_f32(y_imag_f32x4, alpha_imag_f32x4, a_f32x4x2.val[0]);
        float32x4x2_t out = {y_real_f32x4, y_imag_f32x4};
        vst2q_f32((nk_f32_t *)(result + i), out);
    }
    for (; i < n; i++) {
        nk_f32_t a_real = a[i].real, a_imag = a[i].imag;
        result[i].real = alpha->real * a_real - alpha->imag * a_imag + beta->real;
        result[i].imag = alpha->real * a_imag + alpha->imag * a_real + beta->imag;
    }
}

NK_PUBLIC void nk_each_scale_f64c_neon(nk_f64c_t const *a, nk_size_t n, nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                       nk_f64c_t *result) {
    float64x2_t alpha_real_f64x2 = vdupq_n_f64(alpha->real);
    float64x2_t alpha_imag_f64x2 = vdupq_n_f64(alpha->imag);
    float64x2_t beta_real_f64x2 = vdupq_n_f64(beta->real);
    float64x2_t beta_imag_f64x2 = vdupq_n_f64(beta->imag);
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2x2_t a_f64x2x2 = vld2q_f64((nk_f64_t const *)(a + i));
        float64x2_t y_real_f64x2 = vfmaq_f64(beta_real_f64x2, alpha_real_f64x2, a_f64x2x2.val[0]);
        y_real_f64x2 = vfmsq_f64(y_real_f64x2, alpha_imag_f64x2, a_f64x2x2.val[1]);
        float64x2_t y_imag_f64x2 = vfmaq_f64(beta_imag_f64x2, alpha_real_f64x2, a_f64x2x2.val[1]);
        y_imag_f64x2 = vfmaq_f64(y_imag_f64x2, alpha_imag_f64x2, a_f64x2x2.val[0]);
        float64x2x2_t out = {y_real_f64x2, y_imag_f64x2};
        vst2q_f64((nk_f64_t *)(result + i), out);
    }
    for (; i < n; i++) {
        nk_f64_t a_real = a[i].real, a_imag = a[i].imag;
        result[i].real = alpha->real * a_real - alpha->imag * a_imag + beta->real;
        result[i].imag = alpha->real * a_imag + alpha->imag * a_real + beta->imag;
    }
}

NK_PUBLIC void nk_each_blend_f32c_neon(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t const *alpha,
                                       nk_f32c_t const *beta, nk_f32c_t *result) {
    float32x4_t alpha_real_f32x4 = vdupq_n_f32(alpha->real);
    float32x4_t alpha_imag_f32x4 = vdupq_n_f32(alpha->imag);
    float32x4_t beta_real_f32x4 = vdupq_n_f32(beta->real);
    float32x4_t beta_imag_f32x4 = vdupq_n_f32(beta->imag);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t a_f32x4x2 = vld2q_f32((nk_f32_t const *)(a + i));
        float32x4x2_t b_f32x4x2 = vld2q_f32((nk_f32_t const *)(b + i));
        float32x4_t ya_real_f32x4 = vmulq_f32(alpha_real_f32x4, a_f32x4x2.val[0]);
        ya_real_f32x4 = vfmsq_f32(ya_real_f32x4, alpha_imag_f32x4, a_f32x4x2.val[1]);
        float32x4_t ya_imag_f32x4 = vmulq_f32(alpha_real_f32x4, a_f32x4x2.val[1]);
        ya_imag_f32x4 = vfmaq_f32(ya_imag_f32x4, alpha_imag_f32x4, a_f32x4x2.val[0]);
        float32x4_t y_real_f32x4 = vfmaq_f32(ya_real_f32x4, beta_real_f32x4, b_f32x4x2.val[0]);
        y_real_f32x4 = vfmsq_f32(y_real_f32x4, beta_imag_f32x4, b_f32x4x2.val[1]);
        float32x4_t y_imag_f32x4 = vfmaq_f32(ya_imag_f32x4, beta_real_f32x4, b_f32x4x2.val[1]);
        y_imag_f32x4 = vfmaq_f32(y_imag_f32x4, beta_imag_f32x4, b_f32x4x2.val[0]);
        float32x4x2_t out = {y_real_f32x4, y_imag_f32x4};
        vst2q_f32((nk_f32_t *)(result + i), out);
    }
    for (; i < n; i++) {
        nk_f32_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f32_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f32_t ar = alpha->real * a_real - alpha->imag * a_imag;
        nk_f32_t ai = alpha->real * a_imag + alpha->imag * a_real;
        nk_f32_t br = beta->real * b_real - beta->imag * b_imag;
        nk_f32_t bi = beta->real * b_imag + beta->imag * b_real;
        result[i].real = ar + br;
        result[i].imag = ai + bi;
    }
}

NK_PUBLIC void nk_each_blend_f64c_neon(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t const *alpha,
                                       nk_f64c_t const *beta, nk_f64c_t *result) {
    float64x2_t alpha_real_f64x2 = vdupq_n_f64(alpha->real);
    float64x2_t alpha_imag_f64x2 = vdupq_n_f64(alpha->imag);
    float64x2_t beta_real_f64x2 = vdupq_n_f64(beta->real);
    float64x2_t beta_imag_f64x2 = vdupq_n_f64(beta->imag);
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2x2_t a_f64x2x2 = vld2q_f64((nk_f64_t const *)(a + i));
        float64x2x2_t b_f64x2x2 = vld2q_f64((nk_f64_t const *)(b + i));
        float64x2_t ya_real_f64x2 = vmulq_f64(alpha_real_f64x2, a_f64x2x2.val[0]);
        ya_real_f64x2 = vfmsq_f64(ya_real_f64x2, alpha_imag_f64x2, a_f64x2x2.val[1]);
        float64x2_t ya_imag_f64x2 = vmulq_f64(alpha_real_f64x2, a_f64x2x2.val[1]);
        ya_imag_f64x2 = vfmaq_f64(ya_imag_f64x2, alpha_imag_f64x2, a_f64x2x2.val[0]);
        float64x2_t y_real_f64x2 = vfmaq_f64(ya_real_f64x2, beta_real_f64x2, b_f64x2x2.val[0]);
        y_real_f64x2 = vfmsq_f64(y_real_f64x2, beta_imag_f64x2, b_f64x2x2.val[1]);
        float64x2_t y_imag_f64x2 = vfmaq_f64(ya_imag_f64x2, beta_real_f64x2, b_f64x2x2.val[1]);
        y_imag_f64x2 = vfmaq_f64(y_imag_f64x2, beta_imag_f64x2, b_f64x2x2.val[0]);
        float64x2x2_t out = {y_real_f64x2, y_imag_f64x2};
        vst2q_f64((nk_f64_t *)(result + i), out);
    }
    for (; i < n; i++) {
        nk_f64_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f64_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f64_t ar = alpha->real * a_real - alpha->imag * a_imag;
        nk_f64_t ai = alpha->real * a_imag + alpha->imag * a_real;
        nk_f64_t br = beta->real * b_real - beta->imag * b_imag;
        nk_f64_t bi = beta->real * b_imag + beta->imag * b_real;
        result[i].real = ar + br;
        result[i].imag = ai + bi;
    }
}

NK_PUBLIC void nk_each_fma_f32c_neon(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                     nk_f32c_t const *alpha, nk_f32c_t const *beta, nk_f32c_t *result) {
    float32x4_t alpha_real_f32x4 = vdupq_n_f32(alpha->real);
    float32x4_t alpha_imag_f32x4 = vdupq_n_f32(alpha->imag);
    float32x4_t beta_real_f32x4 = vdupq_n_f32(beta->real);
    float32x4_t beta_imag_f32x4 = vdupq_n_f32(beta->imag);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t a_f32x4x2 = vld2q_f32((nk_f32_t const *)(a + i));
        float32x4x2_t b_f32x4x2 = vld2q_f32((nk_f32_t const *)(b + i));
        float32x4x2_t c_f32x4x2 = vld2q_f32((nk_f32_t const *)(c + i));
        float32x4_t ab_real_f32x4 = vmulq_f32(a_f32x4x2.val[0], b_f32x4x2.val[0]);
        ab_real_f32x4 = vfmsq_f32(ab_real_f32x4, a_f32x4x2.val[1], b_f32x4x2.val[1]);
        float32x4_t ab_imag_f32x4 = vmulq_f32(a_f32x4x2.val[0], b_f32x4x2.val[1]);
        ab_imag_f32x4 = vfmaq_f32(ab_imag_f32x4, a_f32x4x2.val[1], b_f32x4x2.val[0]);
        float32x4_t y_real_f32x4 = vmulq_f32(alpha_real_f32x4, ab_real_f32x4);
        y_real_f32x4 = vfmsq_f32(y_real_f32x4, alpha_imag_f32x4, ab_imag_f32x4);
        float32x4_t y_imag_f32x4 = vmulq_f32(alpha_real_f32x4, ab_imag_f32x4);
        y_imag_f32x4 = vfmaq_f32(y_imag_f32x4, alpha_imag_f32x4, ab_real_f32x4);
        y_real_f32x4 = vfmaq_f32(y_real_f32x4, beta_real_f32x4, c_f32x4x2.val[0]);
        y_real_f32x4 = vfmsq_f32(y_real_f32x4, beta_imag_f32x4, c_f32x4x2.val[1]);
        y_imag_f32x4 = vfmaq_f32(y_imag_f32x4, beta_real_f32x4, c_f32x4x2.val[1]);
        y_imag_f32x4 = vfmaq_f32(y_imag_f32x4, beta_imag_f32x4, c_f32x4x2.val[0]);
        float32x4x2_t out = {y_real_f32x4, y_imag_f32x4};
        vst2q_f32((nk_f32_t *)(result + i), out);
    }
    for (; i < n; i++) {
        nk_f32_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f32_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f32_t c_real = c[i].real, c_imag = c[i].imag;
        nk_f32_t ab_real = a_real * b_real - a_imag * b_imag;
        nk_f32_t ab_imag = a_real * b_imag + a_imag * b_real;
        nk_f32_t aab_real = alpha->real * ab_real - alpha->imag * ab_imag;
        nk_f32_t aab_imag = alpha->real * ab_imag + alpha->imag * ab_real;
        nk_f32_t bc_real = beta->real * c_real - beta->imag * c_imag;
        nk_f32_t bc_imag = beta->real * c_imag + beta->imag * c_real;
        result[i].real = aab_real + bc_real;
        result[i].imag = aab_imag + bc_imag;
    }
}

NK_PUBLIC void nk_each_fma_f64c_neon(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                     nk_f64c_t const *alpha, nk_f64c_t const *beta, nk_f64c_t *result) {
    float64x2_t alpha_real_f64x2 = vdupq_n_f64(alpha->real);
    float64x2_t alpha_imag_f64x2 = vdupq_n_f64(alpha->imag);
    float64x2_t beta_real_f64x2 = vdupq_n_f64(beta->real);
    float64x2_t beta_imag_f64x2 = vdupq_n_f64(beta->imag);
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2x2_t a_f64x2x2 = vld2q_f64((nk_f64_t const *)(a + i));
        float64x2x2_t b_f64x2x2 = vld2q_f64((nk_f64_t const *)(b + i));
        float64x2x2_t c_f64x2x2 = vld2q_f64((nk_f64_t const *)(c + i));
        float64x2_t ab_real_f64x2 = vmulq_f64(a_f64x2x2.val[0], b_f64x2x2.val[0]);
        ab_real_f64x2 = vfmsq_f64(ab_real_f64x2, a_f64x2x2.val[1], b_f64x2x2.val[1]);
        float64x2_t ab_imag_f64x2 = vmulq_f64(a_f64x2x2.val[0], b_f64x2x2.val[1]);
        ab_imag_f64x2 = vfmaq_f64(ab_imag_f64x2, a_f64x2x2.val[1], b_f64x2x2.val[0]);
        float64x2_t y_real_f64x2 = vmulq_f64(alpha_real_f64x2, ab_real_f64x2);
        y_real_f64x2 = vfmsq_f64(y_real_f64x2, alpha_imag_f64x2, ab_imag_f64x2);
        float64x2_t y_imag_f64x2 = vmulq_f64(alpha_real_f64x2, ab_imag_f64x2);
        y_imag_f64x2 = vfmaq_f64(y_imag_f64x2, alpha_imag_f64x2, ab_real_f64x2);
        y_real_f64x2 = vfmaq_f64(y_real_f64x2, beta_real_f64x2, c_f64x2x2.val[0]);
        y_real_f64x2 = vfmsq_f64(y_real_f64x2, beta_imag_f64x2, c_f64x2x2.val[1]);
        y_imag_f64x2 = vfmaq_f64(y_imag_f64x2, beta_real_f64x2, c_f64x2x2.val[1]);
        y_imag_f64x2 = vfmaq_f64(y_imag_f64x2, beta_imag_f64x2, c_f64x2x2.val[0]);
        float64x2x2_t out = {y_real_f64x2, y_imag_f64x2};
        vst2q_f64((nk_f64_t *)(result + i), out);
    }
    for (; i < n; i++) {
        nk_f64_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f64_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f64_t c_real = c[i].real, c_imag = c[i].imag;
        nk_f64_t ab_real = a_real * b_real - a_imag * b_imag;
        nk_f64_t ab_imag = a_real * b_imag + a_imag * b_real;
        nk_f64_t aab_real = alpha->real * ab_real - alpha->imag * ab_imag;
        nk_f64_t aab_imag = alpha->real * ab_imag + alpha->imag * ab_real;
        nk_f64_t bc_real = beta->real * c_real - beta->imag * c_imag;
        nk_f64_t bc_imag = beta->real * c_imag + beta->imag * c_real;
        result[i].real = aab_real + bc_real;
        result[i].imag = aab_imag + bc_imag;
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_
#endif // NK_EACH_NEON_H
