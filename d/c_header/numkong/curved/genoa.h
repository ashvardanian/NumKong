/**
 *  @brief SIMD-accelerated Curved Space Similarity for Genoa.
 *  @file include/numkong/curved/genoa.h
 *  @author Ash Vardanian
 *  @date January 14, 2026
 *
 *  @sa include/numkong/curved.h
 *
 *  Implements bf16 bilinear forms using AVX-512 with BF16 extensions.
 */
#ifndef NK_CURVED_GENOA_H
#define NK_CURVED_GENOA_H

#if NK_TARGET_X86_
#if NK_TARGET_GENOA

#include "numkong/types.h"
#include "numkong/spatial/genoa.h"  // `nk_substract_bf16x32_genoa_`
#include "numkong/reduce/skylake.h" // `nk_reduce_add_f32x16_skylake_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512bf16,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512bf16", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC void nk_bilinear_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                      nk_f32_t *result) {
    nk_size_t const tail_length = n % 32;
    nk_size_t const tail_start = n - tail_length;
    __mmask32 const tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_length);
    __m512 sum_f32x16 = _mm512_setzero_ps();

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t a_f32;
        nk_bf16_to_f32_serial(a + i, &a_f32);
        __m512 a_f32x16 = _mm512_set1_ps(a_f32);
        __m512 cb_j_f32x16 = _mm512_setzero_ps();
        __m512i b_bf16x32, c_bf16x32;
        nk_size_t j = 0;

    nk_bilinear_bf16_genoa_cycle:
        if (j + 32 <= n) {
            b_bf16x32 = _mm512_loadu_epi16(b + j);
            c_bf16x32 = _mm512_loadu_epi16(c + i * n + j);
        }
        else {
            b_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, b + tail_start);
            c_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, c + i * n + tail_start);
        }
        cb_j_f32x16 = _mm512_dpbf16_ps(cb_j_f32x16, nk_m512bh_from_m512i_(b_bf16x32), nk_m512bh_from_m512i_(c_bf16x32));
        j += 32;
        if (j < n) goto nk_bilinear_bf16_genoa_cycle;
        sum_f32x16 = _mm512_fmadd_ps(a_f32x16, cb_j_f32x16, sum_f32x16);
    }

    *result = _mm512_reduce_add_ps(sum_f32x16);
}

NK_PUBLIC void nk_mahalanobis_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                         nk_f32_t *result) {
    nk_size_t const tail_length = n % 32;
    nk_size_t const tail_start = n - tail_length;
    __mmask32 const tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_length);
    __m512 sum_f32x16 = _mm512_setzero_ps();

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t a_i, b_i;
        nk_bf16_to_f32_serial(a + i, &a_i);
        nk_bf16_to_f32_serial(b + i, &b_i);
        __m512 diff_i_f32x16 = _mm512_set1_ps(a_i - b_i);
        __m512 cdiff_j_f32x16 = _mm512_setzero_ps();
        __m512i a_j_bf16x32, b_j_bf16x32, diff_j_bf16x32, c_bf16x32;
        nk_size_t j = 0;

        // The nested loop is cleaner to implement with a `goto` in this case:
    nk_mahalanobis_bf16_genoa_cycle:
        if (j + 32 <= n) {
            a_j_bf16x32 = _mm512_loadu_epi16(a + j);
            b_j_bf16x32 = _mm512_loadu_epi16(b + j);
            c_bf16x32 = _mm512_loadu_epi16(c + i * n + j);
        }
        else {
            a_j_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, a + tail_start);
            b_j_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, b + tail_start);
            c_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, c + i * n + tail_start);
        }
        diff_j_bf16x32 = nk_substract_bf16x32_genoa_(a_j_bf16x32, b_j_bf16x32);
        cdiff_j_f32x16 = _mm512_dpbf16_ps(cdiff_j_f32x16, nk_m512bh_from_m512i_(diff_j_bf16x32),
                                          nk_m512bh_from_m512i_(c_bf16x32));
        j += 32;
        if (j < n) goto nk_mahalanobis_bf16_genoa_cycle;
        sum_f32x16 = _mm512_fmadd_ps(diff_i_f32x16, cdiff_j_f32x16, sum_f32x16);
    }

    nk_f32_t quadratic = _mm512_reduce_add_ps(sum_f32x16);
    *result = nk_f32_sqrt_haswell(quadratic > 0 ? quadratic : 0);
}

NK_PUBLIC void nk_bilinear_bf16c_genoa(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c, nk_size_t n,
                                       nk_f32c_t *results) {

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_i32x16 = _mm512_set1_epi32(0x80000000);
    __m512i const swap_adjacent_i8x64 = _mm512_set_epi8(                //
        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50, // 4th 128-bit lane
        45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, // 3rd 128-bit lane
        29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, // 2nd 128-bit lane
        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2            // 1st 128-bit lane
    );

    // Default case for arbitrary size `n`
    nk_size_t const tail_length = n % 16;
    nk_size_t const tail_start = n - tail_length;
    __mmask32 const tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_length * 2);
    nk_f32_t sum_real = 0;
    nk_f32_t sum_imag = 0;

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t a_i_real, a_i_imag;
        nk_bf16_to_f32_serial(&a[i].real, &a_i_real);
        nk_bf16_to_f32_serial(&a[i].imag, &a_i_imag);
        __m512 cb_j_real_f32x16 = _mm512_setzero_ps();
        __m512 cb_j_imag_f32x16 = _mm512_setzero_ps();
        __m512i b_bf16x32, c_bf16x32;
        nk_size_t j = 0;

    nk_bilinear_bf16c_skylake_cycle:
        if (j + 16 <= n) {
            b_bf16x32 = _mm512_loadu_epi16((nk_i16_t const *)(b + j));
            c_bf16x32 = _mm512_loadu_epi16((nk_i16_t const *)(c + i * n + j));
        }
        else {
            b_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, (nk_i16_t const *)(b + tail_start));
            c_bf16x32 = _mm512_maskz_loadu_epi16(tail_mask, (nk_i16_t const *)(c + i * n + tail_start));
        }
        cb_j_real_f32x16 = _mm512_dpbf16_ps(                                      //
            cb_j_real_f32x16,                                                     //
            nk_m512bh_from_m512i_(_mm512_xor_si512(c_bf16x32, sign_flip_i32x16)), //
            nk_m512bh_from_m512i_(b_bf16x32));
        cb_j_imag_f32x16 = _mm512_dpbf16_ps(                                            //
            cb_j_imag_f32x16,                                                           //
            nk_m512bh_from_m512i_(_mm512_shuffle_epi8(c_bf16x32, swap_adjacent_i8x64)), //
            nk_m512bh_from_m512i_(b_bf16x32));
        j += 16;
        if (j < n) goto nk_bilinear_bf16c_skylake_cycle;
        // Horizontal sums are the expensive part of the computation:
        nk_f32_t const cb_j_real = nk_reduce_add_f32x16_skylake_(cb_j_real_f32x16);
        nk_f32_t const cb_j_imag = nk_reduce_add_f32x16_skylake_(cb_j_imag_f32x16);
        sum_real += a_i_real * cb_j_real - a_i_imag * cb_j_imag;
        sum_imag += a_i_real * cb_j_imag + a_i_imag * cb_j_real;
    }

    // Reduce horizontal sums:
    results->real = sum_real;
    results->imag = sum_imag;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_GENOA
#endif // NK_TARGET_X86_
#endif // NK_CURVED_GENOA_H
