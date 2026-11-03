/**
 *  @file include/numkong/curved/neon.h
 *  @author Ash Vardanian
 *  @date August 29, 2024
 *  @brief SIMD-accelerated curved-space similarity for NEON.
 *
 *  @sa c/numkong/curved.h
 *
 *  Implements f32 bilinear forms and Mahalanobis distance using ARM NEON SIMD.
 *  Accumulates f32 inputs in f64 precision to avoid catastrophic cancellation.
 *
 *  @section neon_curved_instructions Key NEON Instructions
 *
 *  @verbatim
 *  Intrinsic     Instruction                  A76       M5
 *  vfmaq_f64     FMLA (V.2D, V.2D, V.2D)      4cy @ 2p  3cy @ 4p
 *  vcvt_f64_f32  FCVTL (V.2D, V.2S)           3cy @ 2p  3cy @ 4p
 *  vaddvq_f64    FADDP (V.2D to scalar)       3cy @ 1p  3cy @ 2p
 *  vld1_f32      LD1 ({Vt.2S}, [Xn])          4cy @ 2p  4cy @ 3p
 *  vld2_f32      LD2 ({Vt.2S, Vt2.2S}, [Xn])  4cy @ 1p  4cy @ 1p
 *  @endverbatim
 *
 *  For f32 bilinear and Mahalanobis, we upcast to f64 for accumulation to preserve precision and
 *  avoid catastrophic cancellation in large-magnitude sums. Rows stay in lanes up to one reduction.
 */
#ifndef NUMKONG_CURVED_NEON_H
#define NUMKONG_CURVED_NEON_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_NEON_

#include "numkong/types.h"
#include "numkong/cast/neon.h" // `nk_partial_load_b16x8_neon_`
#include "numkong/dot/neon.h"  // `nk_dot2_f64x2_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

/** Widens one F16 into all four F32 lanes, through the same conversion as the row bodies. */
NUMKONG_INLINE float32x4_t nk_load_f16_as_f32x4_neon_(nk_f16_t const *value) {
    return vcvt_f32_f16(vreinterpret_f16_u16(vld1_dup_u16((nk_u16_t const *)value)));
}

/** Deinterleaves up to 7 complex F16 pairs without byte stores, leaving zeros past them. */
NUMKONG_INLINE uint16x8x2_t nk_partial_load_f16c_neon_(nk_f16c_t const *pairs, nk_size_t count) {
    nk_b128_vec_t first_vec, second_vec;
    nk_size_t const values = count * 2;
    nk_partial_load_b16x8_neon_(pairs, &first_vec, values < 8 ? values : 8);
    nk_partial_load_b16x8_neon_(pairs + 4, &second_vec, values > 8 ? values - 8 : 0);
    uint16x8x2_t result_u16x8x2;
    result_u16x8x2.val[0] = vuzp1q_u16(first_vec.u16x8, second_vec.u16x8);
    result_u16x8x2.val[1] = vuzp2q_u16(first_vec.u16x8, second_vec.u16x8);
    return result_u16x8x2;
}

/** Adds the complex products of 8 deinterleaved F16 pairs of @p c and @p b onto F32 lanes. */
NUMKONG_INLINE void nk_fma_f16cx8_neon_(uint16x8x2_t c_u16x8x2, uint16x8x2_t b_u16x8x2, float32x4_t *real_f32x4,
                                        float32x4_t *imag_f32x4) {
    float16x8_t b_real_f16x8 = vreinterpretq_f16_u16(b_u16x8x2.val[0]);
    float16x8_t b_imag_f16x8 = vreinterpretq_f16_u16(b_u16x8x2.val[1]);
    float16x8_t c_real_f16x8 = vreinterpretq_f16_u16(c_u16x8x2.val[0]);
    float16x8_t c_imag_f16x8 = vreinterpretq_f16_u16(c_u16x8x2.val[1]);
    float32x4_t b_real_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_real_f16x8));
    float32x4_t b_real_high_f32x4 = vcvt_high_f32_f16(b_real_f16x8);
    float32x4_t b_imag_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_imag_f16x8));
    float32x4_t b_imag_high_f32x4 = vcvt_high_f32_f16(b_imag_f16x8);
    float32x4_t c_real_low_f32x4 = vcvt_f32_f16(vget_low_f16(c_real_f16x8));
    float32x4_t c_real_high_f32x4 = vcvt_high_f32_f16(c_real_f16x8);
    float32x4_t c_imag_low_f32x4 = vcvt_f32_f16(vget_low_f16(c_imag_f16x8));
    float32x4_t c_imag_high_f32x4 = vcvt_high_f32_f16(c_imag_f16x8);
    *real_f32x4 = vfmaq_f32(*real_f32x4, c_real_low_f32x4, b_real_low_f32x4);
    *real_f32x4 = vfmsq_f32(*real_f32x4, c_imag_low_f32x4, b_imag_low_f32x4);
    *real_f32x4 = vfmaq_f32(*real_f32x4, c_real_high_f32x4, b_real_high_f32x4);
    *real_f32x4 = vfmsq_f32(*real_f32x4, c_imag_high_f32x4, b_imag_high_f32x4);
    *imag_f32x4 = vfmaq_f32(*imag_f32x4, c_real_low_f32x4, b_imag_low_f32x4);
    *imag_f32x4 = vfmaq_f32(*imag_f32x4, c_imag_low_f32x4, b_real_low_f32x4);
    *imag_f32x4 = vfmaq_f32(*imag_f32x4, c_real_high_f32x4, b_imag_high_f32x4);
    *imag_f32x4 = vfmaq_f32(*imag_f32x4, c_imag_high_f32x4, b_real_high_f32x4);
}

#if NUMKONG_TARGET_NEON

NUMKONG_API nk_status_t nk_bilinear_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                             nk_f64_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    float64x2_t const zero_f64x2 = vdupq_n_f64(0);
    float64x2_t outer_sum_f64x2 = zero_f64x2, outer_compensation_f64x2 = zero_f64x2;
    for (nk_size_t row = 0; row != n; ++row) {
        nk_f64_t const *c_row = c + row * n;
        float64x2_t inner_sum_f64x2 = zero_f64x2, inner_compensation_f64x2 = zero_f64x2;
        nk_size_t column = 0;
        for (; column + 2 <= n; column += 2)
            nk_dot2_f64x2_neon_(&inner_sum_f64x2, &inner_compensation_f64x2, vld1q_f64(c_row + column),
                                vld1q_f64(b + column));
        if (column < n)
            nk_dot2_f64x2_neon_(&inner_sum_f64x2, &inner_compensation_f64x2,
                                vld1q_lane_f64(c_row + column, zero_f64x2, 0),
                                vld1q_lane_f64(b + column, zero_f64x2, 0));
        nk_dot2_f64x2_neon_(&outer_sum_f64x2, &outer_compensation_f64x2, vdupq_n_f64(a[row]),
                            vaddq_f64(inner_sum_f64x2, inner_compensation_f64x2));
    }
    *result = nk_dot_stable_sum_f64x2_neon_(outer_sum_f64x2, outer_compensation_f64x2);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_mahalanobis_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                                nk_f64_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    float64x2_t const zero_f64x2 = vdupq_n_f64(0);
    float64x2_t outer_sum_f64x2 = zero_f64x2, outer_compensation_f64x2 = zero_f64x2;
    for (nk_size_t row = 0; row != n; ++row) {
        nk_f64_t const *c_row = c + row * n;
        float64x2_t inner_sum_f64x2 = zero_f64x2, inner_compensation_f64x2 = zero_f64x2;
        nk_size_t column = 0;
        for (; column + 2 <= n; column += 2)
            nk_dot2_f64x2_neon_(&inner_sum_f64x2, &inner_compensation_f64x2, vld1q_f64(c_row + column),
                                vsubq_f64(vld1q_f64(a + column), vld1q_f64(b + column)));
        if (column < n)
            nk_dot2_f64x2_neon_(&inner_sum_f64x2, &inner_compensation_f64x2,
                                vld1q_lane_f64(c_row + column, zero_f64x2, 0),
                                vsetq_lane_f64(a[column] - b[column], zero_f64x2, 0));
        nk_dot2_f64x2_neon_(&outer_sum_f64x2, &outer_compensation_f64x2, vdupq_n_f64(a[row] - b[row]),
                            vaddq_f64(inner_sum_f64x2, inner_compensation_f64x2));
    }
    nk_f64_t quadratic = nk_dot_stable_sum_f64x2_neon_(outer_sum_f64x2, outer_compensation_f64x2);
    *result = vget_lane_f64(vsqrt_f64(vdup_n_f64(quadratic > 0 ? quadratic : 0)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_bilinear_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                             nk_f64_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const tail_length = n % 4, tail_start = n - tail_length;
    nk_b128_vec_t b_tail_vec;
    nk_partial_load_b8x16_neon_(b + tail_start, &b_tail_vec, tail_length * sizeof(nk_f32_t));
    float64x2_t const b_tail_low_f64x2 = vcvt_f64_f32(vget_low_f32(b_tail_vec.f32x4));
    float64x2_t const b_tail_high_f64x2 = vcvt_high_f64_f32(b_tail_vec.f32x4);
    float64x2_t outer_low_f64x2 = vdupq_n_f64(0), outer_high_f64x2 = vdupq_n_f64(0);
    for (nk_size_t row = 0; row != n; ++row) {
        nk_f32_t const *c_row = c + row * n;
        float64x2_t inner_low_f64x2 = vdupq_n_f64(0), inner_high_f64x2 = vdupq_n_f64(0);
        for (nk_size_t column = 0; column != tail_start; column += 4) {
            float32x4_t b_f32x4 = vld1q_f32(b + column), c_f32x4 = vld1q_f32(c_row + column);
            inner_low_f64x2 = vfmaq_f64(inner_low_f64x2, vcvt_f64_f32(vget_low_f32(c_f32x4)),
                                        vcvt_f64_f32(vget_low_f32(b_f32x4)));
            inner_high_f64x2 = vfmaq_f64(inner_high_f64x2, vcvt_high_f64_f32(c_f32x4), vcvt_high_f64_f32(b_f32x4));
        }
        if (tail_length) {
            nk_b128_vec_t c_tail_vec;
            nk_partial_load_b8x16_neon_(c_row + tail_start, &c_tail_vec, tail_length * sizeof(nk_f32_t));
            inner_low_f64x2 = vfmaq_f64(inner_low_f64x2, vcvt_f64_f32(vget_low_f32(c_tail_vec.f32x4)),
                                        b_tail_low_f64x2);
            inner_high_f64x2 = vfmaq_f64(inner_high_f64x2, vcvt_high_f64_f32(c_tail_vec.f32x4), b_tail_high_f64x2);
        }
        outer_low_f64x2 = vfmaq_n_f64(outer_low_f64x2, inner_low_f64x2, (nk_f64_t)a[row]);
        outer_high_f64x2 = vfmaq_n_f64(outer_high_f64x2, inner_high_f64x2, (nk_f64_t)a[row]);
    }
    *result = vaddvq_f64(vaddq_f64(outer_low_f64x2, outer_high_f64x2));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_mahalanobis_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                                nk_f64_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const tail_length = n % 4, tail_start = n - tail_length;
    nk_b128_vec_t a_tail_vec, b_tail_vec;
    nk_partial_load_b8x16_neon_(a + tail_start, &a_tail_vec, tail_length * sizeof(nk_f32_t));
    nk_partial_load_b8x16_neon_(b + tail_start, &b_tail_vec, tail_length * sizeof(nk_f32_t));
    float64x2_t const diff_tail_low_f64x2 = vsubq_f64(vcvt_f64_f32(vget_low_f32(a_tail_vec.f32x4)),
                                                      vcvt_f64_f32(vget_low_f32(b_tail_vec.f32x4)));
    float64x2_t const diff_tail_high_f64x2 = vsubq_f64(vcvt_high_f64_f32(a_tail_vec.f32x4),
                                                       vcvt_high_f64_f32(b_tail_vec.f32x4));
    float64x2_t outer_low_f64x2 = vdupq_n_f64(0), outer_high_f64x2 = vdupq_n_f64(0);
    for (nk_size_t row = 0; row != n; ++row) {
        nk_f32_t const *c_row = c + row * n;
        nk_f64_t const diff_row_f64 = (nk_f64_t)a[row] - (nk_f64_t)b[row];
        float64x2_t inner_low_f64x2 = vdupq_n_f64(0), inner_high_f64x2 = vdupq_n_f64(0);
        for (nk_size_t column = 0; column != tail_start; column += 4) {
            float32x4_t a_f32x4 = vld1q_f32(a + column), b_f32x4 = vld1q_f32(b + column);
            float32x4_t c_f32x4 = vld1q_f32(c_row + column);
            float64x2_t diff_low_f64x2 = vsubq_f64(vcvt_f64_f32(vget_low_f32(a_f32x4)),
                                                   vcvt_f64_f32(vget_low_f32(b_f32x4)));
            float64x2_t diff_high_f64x2 = vsubq_f64(vcvt_high_f64_f32(a_f32x4), vcvt_high_f64_f32(b_f32x4));
            inner_low_f64x2 = vfmaq_f64(inner_low_f64x2, vcvt_f64_f32(vget_low_f32(c_f32x4)), diff_low_f64x2);
            inner_high_f64x2 = vfmaq_f64(inner_high_f64x2, vcvt_high_f64_f32(c_f32x4), diff_high_f64x2);
        }
        if (tail_length) {
            nk_b128_vec_t c_tail_vec;
            nk_partial_load_b8x16_neon_(c_row + tail_start, &c_tail_vec, tail_length * sizeof(nk_f32_t));
            inner_low_f64x2 = vfmaq_f64(inner_low_f64x2, vcvt_f64_f32(vget_low_f32(c_tail_vec.f32x4)),
                                        diff_tail_low_f64x2);
            inner_high_f64x2 = vfmaq_f64(inner_high_f64x2, vcvt_high_f64_f32(c_tail_vec.f32x4), diff_tail_high_f64x2);
        }
        outer_low_f64x2 = vfmaq_n_f64(outer_low_f64x2, inner_low_f64x2, diff_row_f64);
        outer_high_f64x2 = vfmaq_n_f64(outer_high_f64x2, inner_high_f64x2, diff_row_f64);
    }
    nk_f64_t quadratic = vaddvq_f64(vaddq_f64(outer_low_f64x2, outer_high_f64x2));
    *result = vget_lane_f64(vsqrt_f64(vdup_n_f64(quadratic > 0 ? quadratic : 0)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_bilinear_f32c_neon(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs,
                                              nk_f32c_t const *c_pairs, nk_size_t n, nk_f64c_t *results,
                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // ARMv8.3-A FCMLA (`vcmlaq_f32`) was benchmarked for this complex inner loop.
    // The deinterleave+4FMA pattern is 2.3x faster on Apple M4 — see `dot/neon.h` comment.
    nk_f64_t outer_sum_real_f64 = 0;
    nk_f64_t outer_sum_imag_f64 = 0;

    for (nk_size_t i = 0; i != n; ++i) {
        // Convert a[i] to f64 for precision
        nk_f64_t a_real_f64 = (nk_f64_t)a_pairs[i].real;
        nk_f64_t a_imag_f64 = (nk_f64_t)a_pairs[i].imag;

        // Inner loop: accumulate Σⱼ cᵢⱼ × bⱼ in f64
        float64x2_t sum_real_real_f64x2 = vdupq_n_f64(0), sum_imag_imag_f64x2 = vdupq_n_f64(0);
        float64x2_t sum_real_imag_f64x2 = vdupq_n_f64(0), sum_imag_real_f64x2 = vdupq_n_f64(0);
        nk_size_t j = 0;

        // Vectorized inner loop: process 2 complex elements at a time
        for (; j + 2 <= n; j += 2) {
            // Load b[j:j+2] as interleaved complex pairs (real, imag, real, imag)
            float32x2x2_t b_f32x2x2 = vld2_f32((nk_f32_t const *)(b_pairs + j));
            float64x2_t b_real_f64x2 = vcvt_f64_f32(b_f32x2x2.val[0]);
            float64x2_t b_imag_f64x2 = vcvt_f64_f32(b_f32x2x2.val[1]);

            // Load c[i*n+j : i*n+j+2] as interleaved complex pairs
            float32x2x2_t c_f32x2x2 = vld2_f32((nk_f32_t const *)(c_pairs + i * n + j));
            float64x2_t c_real_f64x2 = vcvt_f64_f32(c_f32x2x2.val[0]);
            float64x2_t c_imag_f64x2 = vcvt_f64_f32(c_f32x2x2.val[1]);

            // Complex multiply, each product on its own chain
            sum_real_real_f64x2 = vfmaq_f64(sum_real_real_f64x2, c_real_f64x2, b_real_f64x2);
            sum_imag_imag_f64x2 = vfmaq_f64(sum_imag_imag_f64x2, c_imag_f64x2, b_imag_f64x2);
            sum_real_imag_f64x2 = vfmaq_f64(sum_real_imag_f64x2, c_real_f64x2, b_imag_f64x2);
            sum_imag_real_f64x2 = vfmaq_f64(sum_imag_real_f64x2, c_imag_f64x2, b_real_f64x2);
        }

        // Reduce the f64x2 accumulators to scalars
        nk_f64_t inner_sum_real_f64 = vaddvq_f64(vsubq_f64(sum_real_real_f64x2, sum_imag_imag_f64x2));
        nk_f64_t inner_sum_imag_f64 = vaddvq_f64(vaddq_f64(sum_real_imag_f64x2, sum_imag_real_f64x2));

        // Handle tail elements
        for (; j < n; ++j) {
            nk_f64_t b_real = (nk_f64_t)b_pairs[j].real;
            nk_f64_t b_imag = (nk_f64_t)b_pairs[j].imag;
            nk_f64_t c_real = (nk_f64_t)c_pairs[i * n + j].real;
            nk_f64_t c_imag = (nk_f64_t)c_pairs[i * n + j].imag;
            // Complex multiply: c × b
            inner_sum_real_f64 += c_real * b_real - c_imag * b_imag;
            inner_sum_imag_f64 += c_real * b_imag + c_imag * b_real;
        }

        // Outer accumulation
        outer_sum_real_f64 += a_real_f64 * inner_sum_real_f64 - a_imag_f64 * inner_sum_imag_f64;
        outer_sum_imag_f64 += a_real_f64 * inner_sum_imag_f64 + a_imag_f64 * inner_sum_real_f64;
    }

    results->real = outer_sum_real_f64;
    results->imag = outer_sum_imag_f64;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_bilinear_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                             nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const tail_length = n % 8, tail_start = n - tail_length;
    nk_b128_vec_t b_tail_vec;
    nk_partial_load_b16x8_neon_(b + tail_start, &b_tail_vec, tail_length);
    float16x8_t const b_tail_f16x8 = vreinterpretq_f16_u16(b_tail_vec.u16x8);
    float32x4_t const b_tail_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_tail_f16x8));
    float32x4_t const b_tail_high_f32x4 = vcvt_high_f32_f16(b_tail_f16x8);
    float32x4_t outer_low_f32x4 = vdupq_n_f32(0), outer_high_f32x4 = vdupq_n_f32(0);
    for (nk_size_t row = 0; row != n; ++row) {
        nk_f16_t const *c_row = c + row * n;
        float32x4_t a_row_f32x4 = nk_load_f16_as_f32x4_neon_(a + row);
        float32x4_t inner_low_f32x4 = vdupq_n_f32(0), inner_high_f32x4 = vdupq_n_f32(0);
        for (nk_size_t column = 0; column != tail_start; column += 8) {
            float16x8_t b_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)(b + column)));
            float16x8_t c_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)(c_row + column)));
            inner_low_f32x4 = vfmaq_f32(inner_low_f32x4, vcvt_f32_f16(vget_low_f16(c_f16x8)),
                                        vcvt_f32_f16(vget_low_f16(b_f16x8)));
            inner_high_f32x4 = vfmaq_f32(inner_high_f32x4, vcvt_high_f32_f16(c_f16x8), vcvt_high_f32_f16(b_f16x8));
        }
        if (tail_length) {
            nk_b128_vec_t c_tail_vec;
            nk_partial_load_b16x8_neon_(c_row + tail_start, &c_tail_vec, tail_length);
            float16x8_t c_f16x8 = vreinterpretq_f16_u16(c_tail_vec.u16x8);
            inner_low_f32x4 = vfmaq_f32(inner_low_f32x4, vcvt_f32_f16(vget_low_f16(c_f16x8)), b_tail_low_f32x4);
            inner_high_f32x4 = vfmaq_f32(inner_high_f32x4, vcvt_high_f32_f16(c_f16x8), b_tail_high_f32x4);
        }
        outer_low_f32x4 = vfmaq_f32(outer_low_f32x4, inner_low_f32x4, a_row_f32x4);
        outer_high_f32x4 = vfmaq_f32(outer_high_f32x4, inner_high_f32x4, a_row_f32x4);
    }
    *result = vaddvq_f32(vaddq_f32(outer_low_f32x4, outer_high_f32x4));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_mahalanobis_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                                nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const tail_length = n % 8, tail_start = n - tail_length;
    nk_b128_vec_t a_tail_vec, b_tail_vec;
    nk_partial_load_b16x8_neon_(a + tail_start, &a_tail_vec, tail_length);
    nk_partial_load_b16x8_neon_(b + tail_start, &b_tail_vec, tail_length);
    float16x8_t const a_tail_f16x8 = vreinterpretq_f16_u16(a_tail_vec.u16x8);
    float16x8_t const b_tail_f16x8 = vreinterpretq_f16_u16(b_tail_vec.u16x8);
    float32x4_t const diff_tail_low_f32x4 = vsubq_f32(vcvt_f32_f16(vget_low_f16(a_tail_f16x8)),
                                                      vcvt_f32_f16(vget_low_f16(b_tail_f16x8)));
    float32x4_t const diff_tail_high_f32x4 = vsubq_f32(vcvt_high_f32_f16(a_tail_f16x8),
                                                       vcvt_high_f32_f16(b_tail_f16x8));
    float32x4_t outer_low_f32x4 = vdupq_n_f32(0), outer_high_f32x4 = vdupq_n_f32(0);
    for (nk_size_t row = 0; row != n; ++row) {
        nk_f16_t const *c_row = c + row * n;
        float32x4_t diff_row_f32x4 = vsubq_f32(nk_load_f16_as_f32x4_neon_(a + row),
                                               nk_load_f16_as_f32x4_neon_(b + row));
        float32x4_t inner_low_f32x4 = vdupq_n_f32(0), inner_high_f32x4 = vdupq_n_f32(0);
        for (nk_size_t column = 0; column != tail_start; column += 8) {
            float16x8_t a_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)(a + column)));
            float16x8_t b_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)(b + column)));
            float16x8_t c_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)(c_row + column)));
            float32x4_t diff_low_f32x4 = vsubq_f32(vcvt_f32_f16(vget_low_f16(a_f16x8)),
                                                   vcvt_f32_f16(vget_low_f16(b_f16x8)));
            float32x4_t diff_high_f32x4 = vsubq_f32(vcvt_high_f32_f16(a_f16x8), vcvt_high_f32_f16(b_f16x8));
            inner_low_f32x4 = vfmaq_f32(inner_low_f32x4, vcvt_f32_f16(vget_low_f16(c_f16x8)), diff_low_f32x4);
            inner_high_f32x4 = vfmaq_f32(inner_high_f32x4, vcvt_high_f32_f16(c_f16x8), diff_high_f32x4);
        }
        if (tail_length) {
            nk_b128_vec_t c_tail_vec;
            nk_partial_load_b16x8_neon_(c_row + tail_start, &c_tail_vec, tail_length);
            float16x8_t c_f16x8 = vreinterpretq_f16_u16(c_tail_vec.u16x8);
            inner_low_f32x4 = vfmaq_f32(inner_low_f32x4, vcvt_f32_f16(vget_low_f16(c_f16x8)), diff_tail_low_f32x4);
            inner_high_f32x4 = vfmaq_f32(inner_high_f32x4, vcvt_high_f32_f16(c_f16x8), diff_tail_high_f32x4);
        }
        outer_low_f32x4 = vfmaq_f32(outer_low_f32x4, inner_low_f32x4, diff_row_f32x4);
        outer_high_f32x4 = vfmaq_f32(outer_high_f32x4, inner_high_f32x4, diff_row_f32x4);
    }
    nk_f32_t quadratic = vaddvq_f32(vaddq_f32(outer_low_f32x4, outer_high_f32x4));
    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32(quadratic > 0 ? quadratic : 0)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_bilinear_f16c_neon(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs,
                                              nk_f16c_t const *c_pairs, nk_size_t n, nk_f32c_t *results,
                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const tail_length = n % 8, tail_start = n - tail_length;
    uint16x8x2_t const b_tail_u16x8x2 = nk_partial_load_f16c_neon_(b_pairs + tail_start, tail_length);
    float32x4_t outer_real_f32x4 = vdupq_n_f32(0), outer_imag_f32x4 = vdupq_n_f32(0);
    for (nk_size_t row = 0; row != n; ++row) {
        nk_f16c_t const *c_row = c_pairs + row * n;
        float32x4_t a_real_f32x4 = nk_load_f16_as_f32x4_neon_(&(a_pairs + row)->real);
        float32x4_t a_imag_f32x4 = nk_load_f16_as_f32x4_neon_(&(a_pairs + row)->imag);
        float32x4_t inner_real_f32x4 = vdupq_n_f32(0), inner_imag_f32x4 = vdupq_n_f32(0);
        for (nk_size_t column = 0; column != tail_start; column += 8)
            nk_fma_f16cx8_neon_(vld2q_u16((nk_u16_t const *)(c_row + column)),
                                vld2q_u16((nk_u16_t const *)(b_pairs + column)), &inner_real_f32x4, &inner_imag_f32x4);
        if (tail_length)
            nk_fma_f16cx8_neon_(nk_partial_load_f16c_neon_(c_row + tail_start, tail_length), b_tail_u16x8x2,
                                &inner_real_f32x4, &inner_imag_f32x4);
        outer_real_f32x4 = vfmaq_f32(outer_real_f32x4, a_real_f32x4, inner_real_f32x4);
        outer_real_f32x4 = vfmsq_f32(outer_real_f32x4, a_imag_f32x4, inner_imag_f32x4);
        outer_imag_f32x4 = vfmaq_f32(outer_imag_f32x4, a_real_f32x4, inner_imag_f32x4);
        outer_imag_f32x4 = vfmaq_f32(outer_imag_f32x4, a_imag_f32x4, inner_real_f32x4);
    }
    results->real = vaddvq_f32(outer_real_f32x4);
    results->imag = vaddvq_f32(outer_imag_f32x4);
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
#endif // NUMKONG_CURVED_NEON_H
