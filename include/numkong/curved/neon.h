/**
 *  @brief SIMD-accelerated Curved Space Similarity for NEON.
 *  @file include/numkong/curved/neon.h
 *  @author Ash Vardanian
 *  @date January 14, 2026
 *
 *  @sa include/numkong/curved.h
 *
 *  Implements f32 bilinear forms and Mahalanobis distance using ARM NEON SIMD.
 *  Accumulates f32 inputs in f64 precision to avoid catastrophic cancellation.
 *
 *  @section neon_curved_instructions Key NEON Instructions
 *
 *      Intrinsic     Instruction                  A76       M5
 *      vfmaq_f64     FMLA (V.2D, V.2D, V.2D)      4cy @ 2p  3cy @ 4p
 *      vcvt_f64_f32  FCVTL (V.2D, V.2S)           3cy @ 2p  3cy @ 4p
 *      vaddvq_f64    FADDP (V.2D to scalar)       3cy @ 1p  3cy @ 2p
 *      vld1_f32      LD1 ({Vt.2S}, [Xn])          4cy @ 2p  4cy @ 3p
 *      vld2_f32      LD2 ({Vt.2S, Vt2.2S}, [Xn])  4cy @ 1p  4cy @ 1p
 *
 *  For f32 bilinear and Mahalanobis, we upcast to f64 for accumulation to preserve
 *  precision and avoid catastrophic cancellation in large-magnitude sums.
 */
#ifndef NK_CURVED_NEON_H
#define NK_CURVED_NEON_H

#if NK_TARGET_ARM64_
#if NK_TARGET_NEON

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // nk_f64_sqrt_neon
#include "numkong/dot/serial.h"   // `nk_dot_f16c_serial`, `nk_vdot_f16c_serial`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

NK_PUBLIC void nk_bilinear_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                    nk_f64_t *result) {
    nk_f64_t outer_sum_f64 = 0;

    for (nk_size_t i = 0; i != n; ++i) {
        // Convert a[i] to f64 for precision
        nk_f64_t a_i_f64 = (nk_f64_t)a[i];

        // Inner loop: accumulate Σⱼ cᵢⱼ × bⱼ in f64
        float64x2_t inner_sum_f64x2 = vdupq_n_f64(0);
        nk_size_t j = 0;

        // Vectorized inner loop: process 2 elements at a time
        for (; j + 2 <= n; j += 2) {
            // Load b[j:j+2] as f32, upcast to f64
            float32x2_t b_f32x2 = vld1_f32(b + j);
            float64x2_t b_f64x2 = vcvt_f64_f32(b_f32x2);

            // Load c[i*n+j : i*n+j+2] as f32, upcast to f64
            float32x2_t c_f32x2 = vld1_f32(c + i * n + j);
            float64x2_t c_f64x2 = vcvt_f64_f32(c_f32x2);

            // FMA: inner_sum += c × b
            inner_sum_f64x2 = vfmaq_f64(inner_sum_f64x2, c_f64x2, b_f64x2);
        }

        // Reduce the f64x2 accumulator to scalar
        nk_f64_t inner_sum_f64 = vaddvq_f64(inner_sum_f64x2);

        // Handle tail elements
        for (; j < n; ++j) { inner_sum_f64 += (nk_f64_t)c[i * n + j] * (nk_f64_t)b[j]; }

        // Outer accumulation: outer_sum += aᵢ × inner_sum
        outer_sum_f64 += a_i_f64 * inner_sum_f64;
    }

    *result = outer_sum_f64;
}

NK_PUBLIC void nk_mahalanobis_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                       nk_f64_t *result) {
    nk_f64_t outer_sum_f64 = 0;

    for (nk_size_t i = 0; i != n; ++i) {
        // Compute difference (aᵢ - bᵢ) in f64 for precision
        nk_f64_t diff_i_f64 = (nk_f64_t)a[i] - (nk_f64_t)b[i];

        // Inner loop: accumulate Σⱼ cᵢⱼ × (aⱼ - bⱼ) in f64
        float64x2_t inner_sum_f64x2 = vdupq_n_f64(0);
        nk_size_t j = 0;

        // Vectorized inner loop: process 2 elements at a time
        for (; j + 2 <= n; j += 2) {
            // Load a[j:j+2] and b[j:j+2] as f32
            float32x2_t a_f32x2 = vld1_f32(a + j);
            float32x2_t b_f32x2 = vld1_f32(b + j);

            // Compute difference in f32, then upcast to f64
            float32x2_t diff_f32x2 = vsub_f32(a_f32x2, b_f32x2);
            float64x2_t diff_f64x2 = vcvt_f64_f32(diff_f32x2);

            // Load c[i*n+j : i*n+j+2] as f32, upcast to f64
            float32x2_t c_f32x2 = vld1_f32(c + i * n + j);
            float64x2_t c_f64x2 = vcvt_f64_f32(c_f32x2);

            // FMA: inner_sum += c × diff
            inner_sum_f64x2 = vfmaq_f64(inner_sum_f64x2, c_f64x2, diff_f64x2);
        }

        // Reduce the f64x2 accumulator to scalar
        nk_f64_t inner_sum_f64 = vaddvq_f64(inner_sum_f64x2);

        // Handle tail elements
        for (; j < n; ++j) {
            nk_f64_t diff_j_f64 = (nk_f64_t)a[j] - (nk_f64_t)b[j];
            inner_sum_f64 += (nk_f64_t)c[i * n + j] * diff_j_f64;
        }

        // Outer accumulation: outer_sum += diff_i × inner_sum
        outer_sum_f64 += diff_i_f64 * inner_sum_f64;
    }

    // Take square root of the result (clamp to 0 for numerical stability)
    *result = nk_f64_sqrt_neon(outer_sum_f64 > 0 ? outer_sum_f64 : 0);
}

NK_PUBLIC void nk_bilinear_f32c_neon(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_f32c_t const *c_pairs,
                                     nk_size_t n, nk_f64c_t *results) {
    // ARMv8.3-A FCMLA (`vcmlaq_f32`) was benchmarked for this complex inner loop.
    // The deinterleave+4FMA pattern is 2.3x faster on Apple M4 — see `dot/neon.h` comment.
    nk_f64_t outer_sum_real_f64 = 0;
    nk_f64_t outer_sum_imag_f64 = 0;

    for (nk_size_t i = 0; i != n; ++i) {
        // Convert a[i] to f64 for precision
        nk_f64_t a_real_f64 = (nk_f64_t)a_pairs[i].real;
        nk_f64_t a_imag_f64 = (nk_f64_t)a_pairs[i].imag;

        // Inner loop: accumulate Σⱼ cᵢⱼ × bⱼ in f64
        float64x2_t inner_sum_real_f64x2 = vdupq_n_f64(0);
        float64x2_t inner_sum_imag_f64x2 = vdupq_n_f64(0);
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

            // Complex multiply
            inner_sum_real_f64x2 = vfmaq_f64(inner_sum_real_f64x2, c_real_f64x2, b_real_f64x2);
            inner_sum_real_f64x2 = vfmsq_f64(inner_sum_real_f64x2, c_imag_f64x2, b_imag_f64x2);

            // Imaginary part: c_real×b_imag + c_imag×b_real
            inner_sum_imag_f64x2 = vfmaq_f64(inner_sum_imag_f64x2, c_real_f64x2, b_imag_f64x2);
            inner_sum_imag_f64x2 = vfmaq_f64(inner_sum_imag_f64x2, c_imag_f64x2, b_real_f64x2);
        }

        // Reduce the f64x2 accumulators to scalars
        nk_f64_t inner_sum_real_f64 = vaddvq_f64(inner_sum_real_f64x2);
        nk_f64_t inner_sum_imag_f64 = vaddvq_f64(inner_sum_imag_f64x2);

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
}

NK_PUBLIC void nk_bilinear_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                    nk_f32_t *result) {
    nk_f32_t outer_sum = 0;
    for (nk_size_t row = 0; row != n; ++row) {
        nk_f16_t const *c_row = c + row * n;
        nk_f32_t a_row;
        nk_f16_to_f32_serial(a + row, &a_row);
        float32x4_t inner_sum_f32x4 = vdupq_n_f32(0);
        nk_size_t column = 0;
        for (; column + 8 <= n; column += 8) {
            float16x8_t b_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)(b + column)));
            float16x8_t c_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)(c_row + column)));
            float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
            float32x4_t b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
            float32x4_t c_low_f32x4 = vcvt_f32_f16(vget_low_f16(c_f16x8));
            float32x4_t c_high_f32x4 = vcvt_high_f32_f16(c_f16x8);
            inner_sum_f32x4 = vfmaq_f32(inner_sum_f32x4, c_low_f32x4, b_low_f32x4);
            inner_sum_f32x4 = vfmaq_f32(inner_sum_f32x4, c_high_f32x4, b_high_f32x4);
        }
        nk_f32_t inner_sum = vaddvq_f32(inner_sum_f32x4);
        for (; column < n; ++column) {
            nk_f32_t b_val, c_val;
            nk_f16_to_f32_serial(b + column, &b_val);
            nk_f16_to_f32_serial(c_row + column, &c_val);
            inner_sum += c_val * b_val;
        }
        outer_sum += a_row * inner_sum;
    }
    *result = outer_sum;
}

NK_PUBLIC void nk_mahalanobis_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                       nk_f32_t *result) {
    nk_f32_t outer_sum = 0;
    for (nk_size_t row = 0; row != n; ++row) {
        nk_f16_t const *c_row = c + row * n;
        nk_f32_t a_row, b_row;
        nk_f16_to_f32_serial(a + row, &a_row);
        nk_f16_to_f32_serial(b + row, &b_row);
        nk_f32_t diff_row = a_row - b_row;
        float32x4_t inner_sum_f32x4 = vdupq_n_f32(0);
        nk_size_t column = 0;
        for (; column + 8 <= n; column += 8) {
            float16x8_t a_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)(a + column)));
            float16x8_t b_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)(b + column)));
            float16x8_t c_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)(c_row + column)));
            float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
            float32x4_t a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
            float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
            float32x4_t b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
            float32x4_t c_low_f32x4 = vcvt_f32_f16(vget_low_f16(c_f16x8));
            float32x4_t c_high_f32x4 = vcvt_high_f32_f16(c_f16x8);
            float32x4_t diff_low_f32x4 = vsubq_f32(a_low_f32x4, b_low_f32x4);
            float32x4_t diff_high_f32x4 = vsubq_f32(a_high_f32x4, b_high_f32x4);
            inner_sum_f32x4 = vfmaq_f32(inner_sum_f32x4, c_low_f32x4, diff_low_f32x4);
            inner_sum_f32x4 = vfmaq_f32(inner_sum_f32x4, c_high_f32x4, diff_high_f32x4);
        }
        nk_f32_t inner_sum = vaddvq_f32(inner_sum_f32x4);
        for (; column < n; ++column) {
            nk_f32_t a_val, b_val, c_val;
            nk_f16_to_f32_serial(a + column, &a_val);
            nk_f16_to_f32_serial(b + column, &b_val);
            nk_f16_to_f32_serial(c_row + column, &c_val);
            inner_sum += c_val * (a_val - b_val);
        }
        outer_sum += diff_row * inner_sum;
    }
    nk_f32_t quadratic = outer_sum;
    *result = nk_f32_sqrt_neon(quadratic > 0 ? quadratic : 0);
}

NK_PUBLIC void nk_bilinear_f16c_neon(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_f16c_t const *c_pairs,
                                     nk_size_t n, nk_f32c_t *results) {
    nk_f32_t outer_sum_real = 0;
    nk_f32_t outer_sum_imag = 0;
    for (nk_size_t row = 0; row != n; ++row) {
        nk_f16c_t const *c_row = c_pairs + row * n;
        nk_f32_t a_real, a_imag;
        nk_f16_to_f32_serial(&(a_pairs + row)->real, &a_real);
        nk_f16_to_f32_serial(&(a_pairs + row)->imag, &a_imag);
        float32x4_t inner_sum_real_f32x4 = vdupq_n_f32(0);
        float32x4_t inner_sum_imag_f32x4 = vdupq_n_f32(0);
        nk_size_t column = 0;
        for (; column + 8 <= n; column += 8) {
            int16x8x2_t b_i16x8x2 = vld2q_s16((short const *)(b_pairs + column));
            int16x8x2_t c_i16x8x2 = vld2q_s16((short const *)(c_row + column));
            float16x8_t b_real_f16x8 = vreinterpretq_f16_s16(b_i16x8x2.val[0]);
            float16x8_t b_imag_f16x8 = vreinterpretq_f16_s16(b_i16x8x2.val[1]);
            float16x8_t c_real_f16x8 = vreinterpretq_f16_s16(c_i16x8x2.val[0]);
            float16x8_t c_imag_f16x8 = vreinterpretq_f16_s16(c_i16x8x2.val[1]);
            float32x4_t b_real_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_real_f16x8));
            float32x4_t b_real_high_f32x4 = vcvt_high_f32_f16(b_real_f16x8);
            float32x4_t b_imag_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_imag_f16x8));
            float32x4_t b_imag_high_f32x4 = vcvt_high_f32_f16(b_imag_f16x8);
            float32x4_t c_real_low_f32x4 = vcvt_f32_f16(vget_low_f16(c_real_f16x8));
            float32x4_t c_real_high_f32x4 = vcvt_high_f32_f16(c_real_f16x8);
            float32x4_t c_imag_low_f32x4 = vcvt_f32_f16(vget_low_f16(c_imag_f16x8));
            float32x4_t c_imag_high_f32x4 = vcvt_high_f32_f16(c_imag_f16x8);
            inner_sum_real_f32x4 = vfmaq_f32(inner_sum_real_f32x4, c_real_low_f32x4, b_real_low_f32x4);
            inner_sum_real_f32x4 = vfmsq_f32(inner_sum_real_f32x4, c_imag_low_f32x4, b_imag_low_f32x4);
            inner_sum_real_f32x4 = vfmaq_f32(inner_sum_real_f32x4, c_real_high_f32x4, b_real_high_f32x4);
            inner_sum_real_f32x4 = vfmsq_f32(inner_sum_real_f32x4, c_imag_high_f32x4, b_imag_high_f32x4);
            inner_sum_imag_f32x4 = vfmaq_f32(inner_sum_imag_f32x4, c_real_low_f32x4, b_imag_low_f32x4);
            inner_sum_imag_f32x4 = vfmaq_f32(inner_sum_imag_f32x4, c_imag_low_f32x4, b_real_low_f32x4);
            inner_sum_imag_f32x4 = vfmaq_f32(inner_sum_imag_f32x4, c_real_high_f32x4, b_imag_high_f32x4);
            inner_sum_imag_f32x4 = vfmaq_f32(inner_sum_imag_f32x4, c_imag_high_f32x4, b_real_high_f32x4);
        }
        nk_f32_t inner_sum_real = vaddvq_f32(inner_sum_real_f32x4);
        nk_f32_t inner_sum_imag = vaddvq_f32(inner_sum_imag_f32x4);
        for (; column < n; ++column) {
            nk_f32_t b_real, b_imag, c_real, c_imag;
            nk_f16_to_f32_serial(&(b_pairs + column)->real, &b_real);
            nk_f16_to_f32_serial(&(b_pairs + column)->imag, &b_imag);
            nk_f16_to_f32_serial(&(c_row + column)->real, &c_real);
            nk_f16_to_f32_serial(&(c_row + column)->imag, &c_imag);
            inner_sum_real += c_real * b_real - c_imag * b_imag;
            inner_sum_imag += c_real * b_imag + c_imag * b_real;
        }
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

#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM64_
#endif // NK_CURVED_NEON_H
