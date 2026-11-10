/**
 *  @file include/numkong/curved/neonbfdot.h
 *  @author Ash Vardanian
 *  @date August 29, 2024
 *  @brief SIMD-accelerated curved-space similarity for NEON BF16.
 *
 *  @sa c/numkong/curved.h
 *
 *  Implements bf16 bilinear forms and Mahalanobis distance using ARM NEON with BF16 extensions.
 *
 *  @section curved_neonbfdot_instructions ARM NEON BF16 Instructions (ARMv8.6-BF16)
 *
 *  @verbatim
 *  Intrinsic      Instruction               A76       M5
 *  vbfdotq_f32    BFDOT (V.4S, V.8H, V.8H)  3cy @ 2p  2cy @ 1p
 *  vcvt_f32_bf16  BFCVTN (V.4H, V.4S)       3cy @ 2p  3cy @ 4p
 *  vld1q_bf16     LD1 (V.8H)                4cy @ 2p  4cy @ 3p
 *  vaddvq_f32     FADDP+FADDP (V.4S)        5cy @ 1p  8cy @ 1p
 *  vfmaq_f32      FMLA (V.4S, V.4S, V.4S)   4cy @ 2p  3cy @ 4p
 *  @endverbatim
 *
 *  For bilinear forms, BFDOT enables efficient inner-product computation by processing 8 bf16 pairs
 *  into 4 f32 results per instruction. For Mahalanobis distance, bf16 inputs are converted to f32
 *  for subtraction, then accumulated using FMA for numerical stability.
 */
#ifndef NUMKONG_CURVED_NEONBFDOT_H
#define NUMKONG_CURVED_NEONBFDOT_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONBFDOT

#include "numkong/types.h"     // `nk_bf16_t`
#include "numkong/cast/neon.h" // `nk_partial_load_b16x8_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#endif

/** Widens one BF16 into all four F32 lanes. */
NUMKONG_INLINE float32x4_t nk_load_bf16_as_f32x4_neonbfdot_(nk_bf16_t const *value) {
    return nk_bf16x4_to_f32x4_neon_(vld1_dup_u16((nk_u16_t const *)value));
}

/** Widens up to 3 BF16 values without byte stores, leaving zeros past them. */
NUMKONG_INLINE float32x4_t nk_partial_load_bf16x3_neonbfdot_(nk_bf16_t const *values, nk_size_t count) {
    nk_u16_t const *source = (nk_u16_t const *)values;
    uint16x4_t bits_u16x4 = vdup_n_u16(0);
    if (count > 0) bits_u16x4 = vld1_lane_u16(source, bits_u16x4, 0);
    if (count > 1) bits_u16x4 = vld1_lane_u16(source + 1, bits_u16x4, 1);
    if (count > 2) bits_u16x4 = vld1_lane_u16(source + 2, bits_u16x4, 2);
    return nk_bf16x4_to_f32x4_neon_(bits_u16x4);
}

/** Deinterleaves up to 3 complex BF16 pairs without byte stores, leaving zeros past them. */
NUMKONG_INLINE int16x4x2_t nk_partial_load_bf16c_neonbfdot_(nk_bf16c_t const *pairs, nk_size_t count) {
    int16x4x2_t result_i16x4x2 = {{vdup_n_s16(0), vdup_n_s16(0)}};
    if (count > 0) result_i16x4x2 = vld2_lane_s16((short const *)pairs, result_i16x4x2, 0);
    if (count > 1) result_i16x4x2 = vld2_lane_s16((short const *)(pairs + 1), result_i16x4x2, 1);
    if (count > 2) result_i16x4x2 = vld2_lane_s16((short const *)(pairs + 2), result_i16x4x2, 2);
    return result_i16x4x2;
}

/** Adds the complex products of 4 deinterleaved BF16 pairs of @p c and @p b onto F32 lanes. */
NUMKONG_INLINE void nk_fma_bf16cx4_neonbfdot_(int16x4x2_t c_i16x4x2, int16x4x2_t b_i16x4x2, float32x4_t *real_f32x4,
                                              float32x4_t *imag_f32x4) {
    float32x4_t b_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[0]));
    float32x4_t b_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[1]));
    float32x4_t c_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(c_i16x4x2.val[0]));
    float32x4_t c_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(c_i16x4x2.val[1]));
    *real_f32x4 = vfmaq_f32(*real_f32x4, c_real_f32x4, b_real_f32x4);
    *real_f32x4 = vfmsq_f32(*real_f32x4, c_imag_f32x4, b_imag_f32x4);
    *imag_f32x4 = vfmaq_f32(*imag_f32x4, c_real_f32x4, b_imag_f32x4);
    *imag_f32x4 = vfmaq_f32(*imag_f32x4, c_imag_f32x4, b_real_f32x4);
}

NUMKONG_API nk_status_t nk_bilinear_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                   nk_size_t n, nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const tail_length = n % 8, tail_start = n - tail_length;
    nk_b128_vec_t b_tail_vec;
    nk_partial_load_b16x8_neon_(b + tail_start, &b_tail_vec, tail_length);
    bfloat16x8_t const b_tail_bf16x8 = vreinterpretq_bf16_u16(b_tail_vec.u16x8);
    float32x4_t outer_sum_f32x4 = vdupq_n_f32(0);
    for (nk_size_t row = 0; row != n; ++row) {
        nk_bf16_t const *c_row = c + row * n;
        float32x4_t inner_sum_f32x4 = vdupq_n_f32(0);
        for (nk_size_t column = 0; column != tail_start; column += 8) {
            bfloat16x8_t b_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)(b + column));
            bfloat16x8_t c_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)(c_row + column));
            inner_sum_f32x4 = vbfdotq_f32(inner_sum_f32x4, c_bf16x8, b_bf16x8);
        }
        if (tail_length) {
            nk_b128_vec_t c_tail_vec;
            nk_partial_load_b16x8_neon_(c_row + tail_start, &c_tail_vec, tail_length);
            inner_sum_f32x4 = vbfdotq_f32(inner_sum_f32x4, vreinterpretq_bf16_u16(c_tail_vec.u16x8), b_tail_bf16x8);
        }
        outer_sum_f32x4 = vfmaq_f32(outer_sum_f32x4, nk_load_bf16_as_f32x4_neonbfdot_(a + row), inner_sum_f32x4);
    }
    *result = vaddvq_f32(outer_sum_f32x4);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_mahalanobis_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                      nk_size_t n, nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const tail_length = n % 4, tail_start = n - tail_length;
    float32x4_t const diff_tail_f32x4 = vsubq_f32(nk_partial_load_bf16x3_neonbfdot_(a + tail_start, tail_length),
                                                  nk_partial_load_bf16x3_neonbfdot_(b + tail_start, tail_length));
    float32x4_t outer_sum_f32x4 = vdupq_n_f32(0);
    for (nk_size_t row = 0; row != n; ++row) {
        nk_bf16_t const *c_row = c + row * n;
        float32x4_t inner_sum_f32x4 = vdupq_n_f32(0);
        for (nk_size_t column = 0; column != tail_start; column += 4) {
            float32x4_t a_f32x4 = nk_bf16x4_to_f32x4_neon_(vld1_u16((nk_u16_t const *)(a + column)));
            float32x4_t b_f32x4 = nk_bf16x4_to_f32x4_neon_(vld1_u16((nk_u16_t const *)(b + column)));
            float32x4_t c_f32x4 = nk_bf16x4_to_f32x4_neon_(vld1_u16((nk_u16_t const *)(c_row + column)));
            inner_sum_f32x4 = vfmaq_f32(inner_sum_f32x4, c_f32x4, vsubq_f32(a_f32x4, b_f32x4));
        }
        if (tail_length)
            inner_sum_f32x4 = vfmaq_f32(
                inner_sum_f32x4, nk_partial_load_bf16x3_neonbfdot_(c_row + tail_start, tail_length), diff_tail_f32x4);
        float32x4_t diff_row_f32x4 = vsubq_f32(nk_load_bf16_as_f32x4_neonbfdot_(a + row),
                                               nk_load_bf16_as_f32x4_neonbfdot_(b + row));
        outer_sum_f32x4 = vfmaq_f32(outer_sum_f32x4, diff_row_f32x4, inner_sum_f32x4);
    }
    nk_f32_t quadratic = vaddvq_f32(outer_sum_f32x4);
    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32(quadratic < 0 ? 0 : quadratic)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_bilinear_bf16c_neonbfdot(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs,
                                                    nk_bf16c_t const *c_pairs, nk_size_t n, nk_f32c_t *result,
                                                    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // ARMv8.3-A FCMLA was benchmarked for this complex multiply pattern.
    // The deinterleave+4FMA approach is 2.3x faster on Apple M4 — see `dot/neon.h` comment.
    nk_size_t const tail_length = n % 4, tail_start = n - tail_length;
    int16x4x2_t const b_tail_i16x4x2 = nk_partial_load_bf16c_neonbfdot_(b_pairs + tail_start, tail_length);
    float32x4_t outer_real_f32x4 = vdupq_n_f32(0), outer_imag_f32x4 = vdupq_n_f32(0);
    for (nk_size_t row = 0; row != n; ++row) {
        nk_bf16c_t const *c_row = c_pairs + row * n;
        float32x4_t a_real_f32x4 = nk_load_bf16_as_f32x4_neonbfdot_(&a_pairs[row].real);
        float32x4_t a_imag_f32x4 = nk_load_bf16_as_f32x4_neonbfdot_(&a_pairs[row].imag);
        float32x4_t inner_real_f32x4 = vdupq_n_f32(0), inner_imag_f32x4 = vdupq_n_f32(0);
        // MSVC doesn't support vld2_bf16, so load as s16 and reinterpret
        for (nk_size_t column = 0; column != tail_start; column += 4)
            nk_fma_bf16cx4_neonbfdot_(vld2_s16((short const *)(c_row + column)),
                                      vld2_s16((short const *)(b_pairs + column)), &inner_real_f32x4,
                                      &inner_imag_f32x4);
        if (tail_length)
            nk_fma_bf16cx4_neonbfdot_(nk_partial_load_bf16c_neonbfdot_(c_row + tail_start, tail_length), b_tail_i16x4x2,
                                      &inner_real_f32x4, &inner_imag_f32x4);
        outer_real_f32x4 = vfmaq_f32(outer_real_f32x4, a_real_f32x4, inner_real_f32x4);
        outer_real_f32x4 = vfmsq_f32(outer_real_f32x4, a_imag_f32x4, inner_imag_f32x4);
        outer_imag_f32x4 = vfmaq_f32(outer_imag_f32x4, a_real_f32x4, inner_imag_f32x4);
        outer_imag_f32x4 = vfmaq_f32(outer_imag_f32x4, a_imag_f32x4, inner_real_f32x4);
    }
    result->real = vaddvq_f32(outer_real_f32x4);
    result->imag = vaddvq_f32(outer_imag_f32x4);
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_NEONBFDOT
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_CURVED_NEONBFDOT_H
