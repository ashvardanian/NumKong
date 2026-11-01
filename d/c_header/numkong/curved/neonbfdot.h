/**
 *  @brief SIMD-accelerated Curved Space Similarity for NEON BF16.
 *  @file include/numkong/curved/neonbfdot.h
 *  @author Ash Vardanian
 *  @date January 14, 2026
 *
 *  @sa include/numkong/curved.h
 *
 *  Implements bf16 bilinear forms and Mahalanobis distance using ARM NEON with BF16 extensions.
 *
 *  @section curved_neonbfdot_instructions ARM NEON BF16 Instructions (ARMv8.6-BF16)
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *                                                                              A76         M4+/V1+/Oryon
 *      vbfdotq_f32                 BFDOT (V.4S, V.8H, V.8H)        3cy         2/cy        4/cy
 *      vcvt_f32_bf16               BFCVTN (V.4H, V.4S)             3cy         2/cy        4/cy
 *      vld1q_bf16                  LD1 (V.8H)                      4cy         2/cy        3/cy
 *      vaddvq_f32                  FADDP+FADDP (V.4S)              4cy         1/cy        2/cy
 *      vfmaq_f32                   FMLA (V.4S, V.4S, V.4S)         4cy         2/cy        4/cy
 *
 *  For bilinear forms, BFDOT enables efficient inner-product computation by processing 8 bf16
 *  pairs into 4 f32 results per instruction. For Mahalanobis distance, bf16 inputs are converted
 *  to f32 for subtraction, then accumulated using FMA for numerical stability.
 */
#ifndef NK_CURVED_NEONBFDOT_H
#define NK_CURVED_NEONBFDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONBFDOT

#include "numkong/types.h"        // `nk_bf16_t`
#include "numkong/spatial/neon.h" // `nk_f32_sqrt_neon`
#include "numkong/cast/serial.h"  // `nk_bf16_to_f32_serial`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#endif

NK_PUBLIC void nk_bilinear_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                          nk_f32_t *result) {
    float32x4_t outer_sum_f32x4 = vdupq_n_f32(0);

    for (nk_size_t i = 0; i != n; ++i) {
        // Load a[i] and broadcast to f32
        nk_f32_t a_i_f32;
        nk_bf16_to_f32_serial(a + i, &a_i_f32);
        float32x4_t a_i_f32x4 = vdupq_n_f32(a_i_f32);

        // Inner sum
        float32x4_t inner_sum_f32x4 = vdupq_n_f32(0);
        nk_size_t j = 0;

        // Process 8 elements at a time using BFDOT
        for (; j + 8 <= n; j += 8) {
            bfloat16x8_t b_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)(b + j));
            bfloat16x8_t c_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)(c + i * n + j));
            inner_sum_f32x4 = vbfdotq_f32(inner_sum_f32x4, c_bf16x8, b_bf16x8);
        }

        // Handle tail elements (less than 8)
        if (j < n) {
            nk_b128_vec_t b_vec, c_vec;
            nk_partial_load_b16x8_serial_(b + j, &b_vec, n - j);
            nk_partial_load_b16x8_serial_(c + i * n + j, &c_vec, n - j);
            bfloat16x8_t b_bf16x8 = vreinterpretq_bf16_u16(b_vec.u16x8);
            bfloat16x8_t c_bf16x8 = vreinterpretq_bf16_u16(c_vec.u16x8);
            inner_sum_f32x4 = vbfdotq_f32(inner_sum_f32x4, c_bf16x8, b_bf16x8);
        }

        // Accumulate: outer_sum += a[i] * inner_sum
        outer_sum_f32x4 = vfmaq_f32(outer_sum_f32x4, a_i_f32x4, inner_sum_f32x4);
    }

    *result = vaddvq_f32(outer_sum_f32x4);
}

NK_PUBLIC void nk_mahalanobis_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                             nk_f32_t *result) {
    nk_f32_t outer_sum = 0;

    for (nk_size_t i = 0; i != n; ++i) {
        // Compute diff_i = a[i] - b[i] in f32
        nk_f32_t a_i_f32, b_i_f32;
        nk_bf16_to_f32_serial(a + i, &a_i_f32);
        nk_bf16_to_f32_serial(b + i, &b_i_f32);
        nk_f32_t diff_i = a_i_f32 - b_i_f32;

        // Inner sum
        float32x4_t inner_sum_f32x4 = vdupq_n_f32(0);
        nk_size_t j = 0;

        // Process 4 elements at a time (convert bf16->f32, subtract, then FMA)
        for (; j + 4 <= n; j += 4) {
            bfloat16x4_t a_j_bf16x4 = vld1_bf16((nk_bf16_for_arm_simd_t const *)(a + j));
            bfloat16x4_t b_j_bf16x4 = vld1_bf16((nk_bf16_for_arm_simd_t const *)(b + j));
            bfloat16x4_t c_bf16x4 = vld1_bf16((nk_bf16_for_arm_simd_t const *)(c + i * n + j));

            float32x4_t a_j_f32x4 = vcvt_f32_bf16(a_j_bf16x4);
            float32x4_t b_j_f32x4 = vcvt_f32_bf16(b_j_bf16x4);
            float32x4_t c_f32x4 = vcvt_f32_bf16(c_bf16x4);

            float32x4_t diff_j_f32x4 = vsubq_f32(a_j_f32x4, b_j_f32x4);
            inner_sum_f32x4 = vfmaq_f32(inner_sum_f32x4, c_f32x4, diff_j_f32x4);
        }

        // Handle tail elements
        nk_f32_t inner_sum_tail = 0;
        for (; j < n; ++j) {
            nk_f32_t a_j_f32, b_j_f32, c_f32;
            nk_bf16_to_f32_serial(a + j, &a_j_f32);
            nk_bf16_to_f32_serial(b + j, &b_j_f32);
            nk_bf16_to_f32_serial(c + i * n + j, &c_f32);
            inner_sum_tail += c_f32 * (a_j_f32 - b_j_f32);
        }

        // Reduce inner sum and add tail
        nk_f32_t inner_sum = vaddvq_f32(inner_sum_f32x4) + inner_sum_tail;

        // Accumulate: outer_sum += diff_i * inner_sum
        outer_sum += diff_i * inner_sum;
    }

    nk_f32_t quadratic = outer_sum;
    *result = nk_f32_sqrt_neon(quadratic > 0 ? quadratic : 0);
}

NK_PUBLIC void nk_bilinear_bf16c_neonbfdot(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs,
                                           nk_bf16c_t const *c_pairs, nk_size_t n, nk_f32c_t *result) {
    // ARMv8.3-A FCMLA was benchmarked for this complex multiply pattern.
    // The deinterleave+4FMA approach is 2.3x faster on Apple M4 — see `dot/neon.h` comment.
    nk_f32_t outer_sum_real = 0;
    nk_f32_t outer_sum_imag = 0;

    for (nk_size_t i = 0; i != n; ++i) {
        // Load a[i] as complex (real, imag) and convert to f32
        nk_f32_t a_real, a_imag;
        nk_bf16_to_f32_serial(&a_pairs[i].real, &a_real);
        nk_bf16_to_f32_serial(&a_pairs[i].imag, &a_imag);

        // Inner sums for real and imaginary parts of c[i,j] * b[j]
        float32x4_t inner_sum_real_f32x4 = vdupq_n_f32(0);
        float32x4_t inner_sum_imag_f32x4 = vdupq_n_f32(0);
        nk_size_t j = 0;

        // Process 4 complex pairs at a time
        for (; j + 4 <= n; j += 4) {
            // Deinterleave load: separate real and imaginary parts
            // MSVC doesn't support vld2_bf16, so load as s16 and reinterpret
            int16x4x2_t b_i16x4x2 = vld2_s16((short const *)(b_pairs + j));
            int16x4x2_t c_i16x4x2 = vld2_s16((short const *)(c_pairs + i * n + j));

            float32x4_t b_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[0]));
            float32x4_t b_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[1]));
            float32x4_t c_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(c_i16x4x2.val[0]));
            float32x4_t c_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(c_i16x4x2.val[1]));

            // Complex multiply: c * b = (c_real*b_real - c_imag*b_imag) + (c_real*b_imag + c_imag*b_real)*i
            // Real part: c_real*b_real - c_imag*b_imag
            inner_sum_real_f32x4 = vfmaq_f32(inner_sum_real_f32x4, c_real_f32x4, b_real_f32x4);
            inner_sum_real_f32x4 = vfmsq_f32(inner_sum_real_f32x4, c_imag_f32x4, b_imag_f32x4);
            // Imaginary part: c_real*b_imag + c_imag*b_real
            inner_sum_imag_f32x4 = vfmaq_f32(inner_sum_imag_f32x4, c_real_f32x4, b_imag_f32x4);
            inner_sum_imag_f32x4 = vfmaq_f32(inner_sum_imag_f32x4, c_imag_f32x4, b_real_f32x4);
        }

        // Handle tail elements
        nk_f32_t inner_sum_real_tail = 0, inner_sum_imag_tail = 0;
        for (; j < n; ++j) {
            nk_f32_t b_real, b_imag, c_real, c_imag;
            nk_bf16_to_f32_serial(&b_pairs[j].real, &b_real);
            nk_bf16_to_f32_serial(&b_pairs[j].imag, &b_imag);
            nk_bf16_to_f32_serial(&c_pairs[i * n + j].real, &c_real);
            nk_bf16_to_f32_serial(&c_pairs[i * n + j].imag, &c_imag);
            // Complex multiply: c * b
            inner_sum_real_tail += c_real * b_real - c_imag * b_imag;
            inner_sum_imag_tail += c_real * b_imag + c_imag * b_real;
        }

        // Reduce inner sums
        nk_f32_t inner_sum_real = vaddvq_f32(inner_sum_real_f32x4) + inner_sum_real_tail;
        nk_f32_t inner_sum_imag = vaddvq_f32(inner_sum_imag_f32x4) + inner_sum_imag_tail;

        // Complex multiply: a * inner_sum = (a_real*inner_real - a_imag*inner_imag) + (a_real*inner_imag +
        // a_imag*inner_real)*i
        outer_sum_real += a_real * inner_sum_real - a_imag * inner_sum_imag;
        outer_sum_imag += a_real * inner_sum_imag + a_imag * inner_sum_real;
    }

    result->real = outer_sum_real;
    result->imag = outer_sum_imag;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONBFDOT
#endif // NK_TARGET_ARM_
#endif // NK_CURVED_NEONBFDOT_H
