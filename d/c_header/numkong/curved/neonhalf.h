/**
 *  @brief SIMD-accelerated Curved Space Similarity for NEON FP16.
 *  @file include/numkong/curved/neonhalf.h
 *  @author Ash Vardanian
 *  @date January 14, 2026
 *
 *  @sa include/numkong/curved.h
 *
 *  Implements f16 bilinear forms and Mahalanobis distance using ARM NEON with FP16 extensions.
 *
 *  @section curved_neonhalf_instructions ARM NEON FP16 Instructions (ARMv8.2-FP16)
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *                                                                              A76         M4+/V1+/Oryon
 *      vfmaq_f32                   FMLA (V.4S, V.4S, V.4S)         4cy         2/cy        4/cy
 *      vcvt_f32_f16                FCVTL (V.4S, V.4H)              3cy         2/cy        4/cy
 *      vld1_f16                    LD1 (V.4H)                      4cy         2/cy        3/cy
 *      vsubq_f32                   FSUB (V.4S, V.4S, V.4S)         2cy         2/cy        4/cy
 *      vaddvq_f32                  FADDP+FADDP (V.4S)              4cy         1/cy        2/cy
 *
 *  Bilinear forms involve nested summation O(n^2) operations. For numerical stability,
 *  f16 inputs are widened to f32 for accumulation. The matrix C is accessed row-by-row
 *  to maintain cache locality.
 *
 *  Mathematical definitions:
 *  - Bilinear: result = ∑ᵢ ∑ⱼ aᵢ × cᵢⱼ × bⱼ
 *  - Mahalanobis: result = √((a - b)ᵀ × C × (a - b))
 */
#ifndef NK_CURVED_NEONHALF_H
#define NK_CURVED_NEONHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // `nk_f32_sqrt_neon`
#include "numkong/cast/serial.h"  // `nk_f16_to_f32_serial`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

NK_PUBLIC void nk_bilinear_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                        nk_f32_t *result) {
    nk_f32_t outer_sum = 0;

    // Process rows of the matrix
    for (nk_size_t row = 0; row != n; ++row) {
        nk_f16_t const *c_row = c + row * n;

        // Load a[row] as f32
        nk_f32_t a_row;
        nk_f16_to_f32_serial(a + row, &a_row);

        // Compute inner sum
        float32x4_t inner_sum_f32x4 = vdupq_n_f32(0);
        nk_size_t column = 0;

        // Process 4 elements at a time
        for (; column + 4 <= n; column += 4) {
            float32x4_t b_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)(b + column)));
            float32x4_t c_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)(c_row + column)));
            inner_sum_f32x4 = vfmaq_f32(inner_sum_f32x4, c_f32x4, b_f32x4);
        }

        // Reduce SIMD accumulator
        nk_f32_t inner_sum = vaddvq_f32(inner_sum_f32x4);

        // Handle tail elements with scalar code
        for (; column < n; ++column) {
            nk_f32_t b_val, c_val;
            nk_f16_to_f32_serial(b + column, &b_val);
            nk_f16_to_f32_serial(c_row + column, &c_val);
            inner_sum += c_val * b_val;
        }

        // Multiply by a[row] and accumulate
        outer_sum += a_row * inner_sum;
    }

    *result = outer_sum;
}

NK_PUBLIC void nk_mahalanobis_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                           nk_f32_t *result) {
    nk_f32_t outer_sum = 0;

    // Process rows of the matrix
    for (nk_size_t row = 0; row != n; ++row) {
        nk_f16_t const *c_row = c + row * n;

        // Compute diff_row = a[row] - b[row] in f32
        nk_f32_t a_row, b_row;
        nk_f16_to_f32_serial(a + row, &a_row);
        nk_f16_to_f32_serial(b + row, &b_row);
        nk_f32_t diff_row = a_row - b_row;

        // Compute inner sum
        float32x4_t inner_sum_f32x4 = vdupq_n_f32(0);
        nk_size_t column = 0;

        // Process 4 elements at a time
        for (; column + 4 <= n; column += 4) {
            float32x4_t a_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)(a + column)));
            float32x4_t b_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)(b + column)));
            float32x4_t c_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)(c_row + column)));
            float32x4_t diff_column_f32x4 = vsubq_f32(a_f32x4, b_f32x4);
            inner_sum_f32x4 = vfmaq_f32(inner_sum_f32x4, c_f32x4, diff_column_f32x4);
        }

        // Reduce SIMD accumulator
        nk_f32_t inner_sum = vaddvq_f32(inner_sum_f32x4);

        // Handle tail elements with scalar code
        for (; column < n; ++column) {
            nk_f32_t a_val, b_val, c_val;
            nk_f16_to_f32_serial(a + column, &a_val);
            nk_f16_to_f32_serial(b + column, &b_val);
            nk_f16_to_f32_serial(c_row + column, &c_val);
            inner_sum += c_val * (a_val - b_val);
        }

        // Multiply by diff_row and accumulate
        outer_sum += diff_row * inner_sum;
    }

    nk_f32_t quadratic = outer_sum;
    *result = nk_f32_sqrt_neon(quadratic > 0 ? quadratic : 0);
}

NK_PUBLIC void nk_bilinear_f16c_neonhalf(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_f16c_t const *c_pairs,
                                         nk_size_t n, nk_f32c_t *results) {
    nk_f32_t outer_sum_real = 0;
    nk_f32_t outer_sum_imag = 0;

    // Process rows of the matrix
    for (nk_size_t row = 0; row != n; ++row) {
        nk_f16c_t const *c_row = c_pairs + row * n;

        // Load a[row] complex value
        nk_f32_t a_real, a_imag;
        nk_f16_to_f32_serial(&(a_pairs + row)->real, &a_real);
        nk_f16_to_f32_serial(&(a_pairs + row)->imag, &a_imag);

        // Compute inner sum
        float32x4_t inner_sum_real_f32x4 = vdupq_n_f32(0);
        float32x4_t inner_sum_imag_f32x4 = vdupq_n_f32(0);
        nk_size_t column = 0;

        // Process 4 complex pairs at a time using deinterleaved loads
        for (; column + 4 <= n; column += 4) {
            // Deinterleave real/imaginary using vld2_s16 pattern from dot/neonhalf.h
            int16x4x2_t b_i16x4x2 = vld2_s16((short const *)(b_pairs + column));
            int16x4x2_t c_i16x4x2 = vld2_s16((short const *)(c_row + column));
            float32x4_t b_real_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(b_i16x4x2.val[0]));
            float32x4_t b_imag_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(b_i16x4x2.val[1]));
            float32x4_t c_real_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(c_i16x4x2.val[0]));
            float32x4_t c_imag_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(c_i16x4x2.val[1]));

            // Complex multiply
            inner_sum_real_f32x4 = vfmaq_f32(inner_sum_real_f32x4, c_real_f32x4, b_real_f32x4);
            inner_sum_real_f32x4 = vfmsq_f32(inner_sum_real_f32x4, c_imag_f32x4, b_imag_f32x4);
            inner_sum_imag_f32x4 = vfmaq_f32(inner_sum_imag_f32x4, c_real_f32x4, b_imag_f32x4);
            inner_sum_imag_f32x4 = vfmaq_f32(inner_sum_imag_f32x4, c_imag_f32x4, b_real_f32x4);
        }

        // Reduce SIMD accumulators
        nk_f32_t inner_sum_real = vaddvq_f32(inner_sum_real_f32x4);
        nk_f32_t inner_sum_imag = vaddvq_f32(inner_sum_imag_f32x4);

        // Handle tail elements with scalar code
        for (; column < n; ++column) {
            nk_f32_t b_real, b_imag, c_real, c_imag;
            nk_f16_to_f32_serial(&(b_pairs + column)->real, &b_real);
            nk_f16_to_f32_serial(&(b_pairs + column)->imag, &b_imag);
            nk_f16_to_f32_serial(&(c_row + column)->real, &c_real);
            nk_f16_to_f32_serial(&(c_row + column)->imag, &c_imag);

            // Complex multiply
            inner_sum_real += c_real * b_real - c_imag * b_imag;
            inner_sum_imag += c_real * b_imag + c_imag * b_real;
        }

        // Complex multiply
        outer_sum_real += a_real * inner_sum_real - a_imag * inner_sum_imag;
        outer_sum_imag += a_real * inner_sum_imag + a_imag * inner_sum_real;
    }

    results->real = outer_sum_real;
    results->imag = outer_sum_imag;
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
#endif // NK_CURVED_NEONHALF_H
