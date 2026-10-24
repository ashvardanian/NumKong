/**
 *  @brief SIMD-accelerated Curved Space Similarity for Skylake.
 *  @file include/numkong/curved/skylake.h
 *  @author Ash Vardanian
 *  @date January 14, 2026
 *
 *  @sa include/numkong/curved.h
 *
 *  Implements f32 and f64 bilinear forms and Mahalanobis distance using AVX-512:
 *  - f32 inputs accumulate in f64 to avoid catastrophic cancellation
 *  - f64 inputs use Dot2 algorithm (Ogita-Rump-Oishi 2005) for error compensation
 */
#ifndef NK_CURVED_SKYLAKE_H
#define NK_CURVED_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE

#include "numkong/types.h"
#include "numkong/spatial/haswell.h" // `nk_f64_sqrt_haswell`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC void nk_bilinear_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                       nk_f64_t *result) {

    // Default case for arbitrary size `n`
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __mmask8 const tail_mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, tail_length);

    for (nk_size_t i = 0; i != n; ++i) {
        __m512d a_f64x8 = _mm512_set1_pd((nk_f64_t)a[i]);
        __m512d cb_j_f64x8 = _mm512_setzero_pd();
        __m256 b_f32x8, c_f32x8;
        nk_size_t j = 0;

    nk_bilinear_f32_skylake_cycle:
        if (j + 8 <= n) {
            b_f32x8 = _mm256_loadu_ps(b + j);
            c_f32x8 = _mm256_loadu_ps(c + i * n + j);
        }
        else {
            b_f32x8 = _mm256_maskz_loadu_ps(tail_mask, b + tail_start);
            c_f32x8 = _mm256_maskz_loadu_ps(tail_mask, c + i * n + tail_start);
        }
        cb_j_f64x8 = _mm512_fmadd_pd(_mm512_cvtps_pd(b_f32x8), _mm512_cvtps_pd(c_f32x8), cb_j_f64x8);
        j += 8;
        if (j < n) goto nk_bilinear_f32_skylake_cycle;
        sum_f64x8 = _mm512_fmadd_pd(a_f64x8, cb_j_f64x8, sum_f64x8);
    }

    *result = _mm512_reduce_add_pd(sum_f64x8);
}

NK_PUBLIC void nk_mahalanobis_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                          nk_f64_t *result) {
    // We use f64 accumulators to prevent catastrophic cancellation.
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __mmask8 const tail_mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, tail_length);

    for (nk_size_t i = 0; i != n; ++i) {
        __m512d diff_i_f64x8 = _mm512_set1_pd((nk_f64_t)a[i] - (nk_f64_t)b[i]);
        __m512d cdiff_j_f64x8 = _mm512_setzero_pd();
        __m256 a_j_f32x8, b_j_f32x8, c_f32x8;
        nk_size_t j = 0;

        // The nested loop is cleaner to implement with a `goto` in this case:
    nk_mahalanobis_f32_skylake_cycle:
        if (j + 8 <= n) {
            a_j_f32x8 = _mm256_loadu_ps(a + j);
            b_j_f32x8 = _mm256_loadu_ps(b + j);
            c_f32x8 = _mm256_loadu_ps(c + i * n + j);
        }
        else {
            a_j_f32x8 = _mm256_maskz_loadu_ps(tail_mask, a + tail_start);
            b_j_f32x8 = _mm256_maskz_loadu_ps(tail_mask, b + tail_start);
            c_f32x8 = _mm256_maskz_loadu_ps(tail_mask, c + i * n + tail_start);
        }
        __m512d diff_j_f64x8 = _mm512_sub_pd(_mm512_cvtps_pd(a_j_f32x8), _mm512_cvtps_pd(b_j_f32x8));
        cdiff_j_f64x8 = _mm512_fmadd_pd(diff_j_f64x8, _mm512_cvtps_pd(c_f32x8), cdiff_j_f64x8);
        j += 8;
        if (j < n) goto nk_mahalanobis_f32_skylake_cycle;
        sum_f64x8 = _mm512_fmadd_pd(diff_i_f64x8, cdiff_j_f64x8, sum_f64x8);
    }

    nk_f64_t quadratic = _mm512_reduce_add_pd(sum_f64x8);
    *result = nk_f64_sqrt_haswell(quadratic > 0 ? quadratic : 0);
}

NK_PUBLIC void nk_bilinear_f32c_skylake(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                        nk_f64c_t *results) {

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors. We use f64 accumulators to prevent catastrophic cancellation.
    __m512i const sign_flip_i64x8 = _mm512_set_epi64(                                   //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000, //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000  //
    );

    // Default case for arbitrary size `n`
    nk_size_t const tail_length = n % 4;
    nk_size_t const tail_start = n - tail_length;
    __mmask8 const tail_mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, tail_length * 2);
    nk_f64_t sum_real = 0;
    nk_f64_t sum_imag = 0;

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_t const a_i_real = (nk_f64_t)a[i].real;
        nk_f64_t const a_i_imag = (nk_f64_t)a[i].imag;
        __m512d cb_j_real_f64x8 = _mm512_setzero_pd();
        __m512d cb_j_imag_f64x8 = _mm512_setzero_pd();
        __m256 b_f32x8, c_f32x8;
        nk_size_t j = 0;

    nk_bilinear_f32c_skylake_cycle:
        if (j + 4 <= n) {
            b_f32x8 = _mm256_loadu_ps((nk_f32_t const *)(b + j));
            c_f32x8 = _mm256_loadu_ps((nk_f32_t const *)(c + i * n + j));
        }
        else {
            b_f32x8 = _mm256_maskz_loadu_ps(tail_mask, (nk_f32_t const *)(b + tail_start));
            c_f32x8 = _mm256_maskz_loadu_ps(tail_mask, (nk_f32_t const *)(c + i * n + tail_start));
        }
        __m512d b_f64x8 = _mm512_cvtps_pd(b_f32x8);
        __m512d c_f64x8 = _mm512_cvtps_pd(c_f32x8);
        // The real part of the product: b.real * c.real - b.imag * c.imag.
        // The subtraction will be performed later with a sign flip.
        cb_j_real_f64x8 = _mm512_fmadd_pd(c_f64x8, b_f64x8, cb_j_real_f64x8);
        // The imaginary part of the product: b.real * c.imag + b.imag * c.real.
        // Swap the imaginary and real parts of `c` before multiplication:
        c_f64x8 = _mm512_permute_pd(c_f64x8, 0x55); //? Same as 0b01010101. Swap adjacent entries within each pair
        cb_j_imag_f64x8 = _mm512_fmadd_pd(c_f64x8, b_f64x8, cb_j_imag_f64x8);
        j += 4;
        if (j < n) goto nk_bilinear_f32c_skylake_cycle;
        // Flip the sign bit in every second scalar before accumulation:
        cb_j_real_f64x8 = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(cb_j_real_f64x8), sign_flip_i64x8));
        // Horizontal sums are the expensive part of the computation:
        nk_f64_t const cb_j_real = _mm512_reduce_add_pd(cb_j_real_f64x8);
        nk_f64_t const cb_j_imag = _mm512_reduce_add_pd(cb_j_imag_f64x8);
        sum_real += a_i_real * cb_j_real - a_i_imag * cb_j_imag;
        sum_imag += a_i_real * cb_j_imag + a_i_imag * cb_j_real;
    }

    // Reduce horizontal sums:
    results->real = sum_real;
    results->imag = sum_imag;
}

NK_PUBLIC void nk_bilinear_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                       nk_f64_t *result) {

    // Default case for arbitrary size `n`
    // Using Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated summation.
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d compensation_f64x8 = _mm512_setzero_pd();
    __mmask8 const tail_mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, tail_length);

    for (nk_size_t i = 0; i != n; ++i) {
        __m512d a_f64x8 = _mm512_set1_pd(a[i]);
        __m512d cb_j_f64x8 = _mm512_setzero_pd();
        __m512d inner_compensation_f64x8 = _mm512_setzero_pd();
        __m512d b_f64x8, c_f64x8;
        nk_size_t j = 0;

    nk_bilinear_f64_skylake_cycle:
        if (j + 8 <= n) {
            b_f64x8 = _mm512_loadu_pd(b + j);
            c_f64x8 = _mm512_loadu_pd(c + i * n + j);
        }
        else {
            b_f64x8 = _mm512_maskz_loadu_pd(tail_mask, b + tail_start);
            c_f64x8 = _mm512_maskz_loadu_pd(tail_mask, c + i * n + tail_start);
        }
        // Inner loop Dot2: accumulate cb_j = sum(b[j] * c[i,j])
        // TwoProd: product = b * c, product_error = fma(b, c, -product)
        {
            __m512d product_f64x8 = _mm512_mul_pd(b_f64x8, c_f64x8);
            __m512d product_error_f64x8 = _mm512_fmsub_pd(b_f64x8, c_f64x8, product_f64x8);
            // TwoSum: t = cb_j + product
            __m512d tentative_sum_f64x8 = _mm512_add_pd(cb_j_f64x8, product_f64x8);
            __m512d virtual_addend_f64x8 = _mm512_sub_pd(tentative_sum_f64x8, cb_j_f64x8);
            __m512d sum_error_f64x8 = _mm512_add_pd(
                _mm512_sub_pd(cb_j_f64x8, _mm512_sub_pd(tentative_sum_f64x8, virtual_addend_f64x8)),
                _mm512_sub_pd(product_f64x8, virtual_addend_f64x8));
            cb_j_f64x8 = tentative_sum_f64x8;
            inner_compensation_f64x8 = _mm512_add_pd(inner_compensation_f64x8,
                                                     _mm512_add_pd(sum_error_f64x8, product_error_f64x8));
        }
        j += 8;
        if (j < n) goto nk_bilinear_f64_skylake_cycle;

        // Combine inner sum with compensation before outer accumulation
        cb_j_f64x8 = _mm512_add_pd(cb_j_f64x8, inner_compensation_f64x8);

        // Outer loop Dot2: accumulate sum += a[i] * cb_j
        // TwoProd: product = a * cb_j, product_error = fma(a, cb_j, -product)
        {
            __m512d product_f64x8 = _mm512_mul_pd(a_f64x8, cb_j_f64x8);
            __m512d product_error_f64x8 = _mm512_fmsub_pd(a_f64x8, cb_j_f64x8, product_f64x8);
            // TwoSum: t = sum + product
            __m512d tentative_sum_f64x8 = _mm512_add_pd(sum_f64x8, product_f64x8);
            __m512d virtual_addend_f64x8 = _mm512_sub_pd(tentative_sum_f64x8, sum_f64x8);
            __m512d sum_error_f64x8 = _mm512_add_pd(
                _mm512_sub_pd(sum_f64x8, _mm512_sub_pd(tentative_sum_f64x8, virtual_addend_f64x8)),
                _mm512_sub_pd(product_f64x8, virtual_addend_f64x8));
            sum_f64x8 = tentative_sum_f64x8;
            compensation_f64x8 = _mm512_add_pd(compensation_f64x8, _mm512_add_pd(sum_error_f64x8, product_error_f64x8));
        }
    }

    // Final: combine sum + compensation before reduce
    *result = _mm512_reduce_add_pd(_mm512_add_pd(sum_f64x8, compensation_f64x8));
}

NK_PUBLIC void nk_mahalanobis_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                          nk_f64_t *result) {
    // Using Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated summation.
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    __mmask8 const tail_mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, tail_length);
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d compensation_f64x8 = _mm512_setzero_pd();

    for (nk_size_t i = 0; i != n; ++i) {
        __m512d diff_i_f64x8 = _mm512_set1_pd(a[i] - b[i]);
        __m512d cdiff_j_f64x8 = _mm512_setzero_pd();
        __m512d inner_compensation_f64x8 = _mm512_setzero_pd();
        __m512d a_j_f64x8, b_j_f64x8, diff_j_f64x8, c_f64x8;
        nk_size_t j = 0;

        // The nested loop is cleaner to implement with a `goto` in this case:
    nk_mahalanobis_f64_skylake_cycle:
        if (j + 8 <= n) {
            a_j_f64x8 = _mm512_loadu_pd(a + j);
            b_j_f64x8 = _mm512_loadu_pd(b + j);
            c_f64x8 = _mm512_loadu_pd(c + i * n + j);
        }
        else {
            a_j_f64x8 = _mm512_maskz_loadu_pd(tail_mask, a + tail_start);
            b_j_f64x8 = _mm512_maskz_loadu_pd(tail_mask, b + tail_start);
            c_f64x8 = _mm512_maskz_loadu_pd(tail_mask, c + i * n + tail_start);
        }
        diff_j_f64x8 = _mm512_sub_pd(a_j_f64x8, b_j_f64x8);

        // Inner loop Dot2: accumulate cdiff_j = sum(diff_j * c[i,j])
        // TwoProd: product = diff_j * c, product_error = fma(diff_j, c, -product)
        {
            __m512d product_f64x8 = _mm512_mul_pd(diff_j_f64x8, c_f64x8);
            __m512d product_error_f64x8 = _mm512_fmsub_pd(diff_j_f64x8, c_f64x8, product_f64x8);
            // TwoSum: t = cdiff_j + product
            __m512d tentative_sum_f64x8 = _mm512_add_pd(cdiff_j_f64x8, product_f64x8);
            __m512d virtual_addend_f64x8 = _mm512_sub_pd(tentative_sum_f64x8, cdiff_j_f64x8);
            __m512d sum_error_f64x8 = _mm512_add_pd(
                _mm512_sub_pd(cdiff_j_f64x8, _mm512_sub_pd(tentative_sum_f64x8, virtual_addend_f64x8)),
                _mm512_sub_pd(product_f64x8, virtual_addend_f64x8));
            cdiff_j_f64x8 = tentative_sum_f64x8;
            inner_compensation_f64x8 = _mm512_add_pd(inner_compensation_f64x8,
                                                     _mm512_add_pd(sum_error_f64x8, product_error_f64x8));
        }
        j += 8;
        if (j < n) goto nk_mahalanobis_f64_skylake_cycle;

        // Combine inner sum with compensation before outer accumulation
        cdiff_j_f64x8 = _mm512_add_pd(cdiff_j_f64x8, inner_compensation_f64x8);

        // Outer loop Dot2: accumulate sum += diff_i * cdiff_j
        // TwoProd: product = diff_i * cdiff_j, product_error = fma(diff_i, cdiff_j, -product)
        {
            __m512d product_f64x8 = _mm512_mul_pd(diff_i_f64x8, cdiff_j_f64x8);
            __m512d product_error_f64x8 = _mm512_fmsub_pd(diff_i_f64x8, cdiff_j_f64x8, product_f64x8);
            // TwoSum: t = sum + product
            __m512d tentative_sum_f64x8 = _mm512_add_pd(sum_f64x8, product_f64x8);
            __m512d virtual_addend_f64x8 = _mm512_sub_pd(tentative_sum_f64x8, sum_f64x8);
            __m512d sum_error_f64x8 = _mm512_add_pd(
                _mm512_sub_pd(sum_f64x8, _mm512_sub_pd(tentative_sum_f64x8, virtual_addend_f64x8)),
                _mm512_sub_pd(product_f64x8, virtual_addend_f64x8));
            sum_f64x8 = tentative_sum_f64x8;
            compensation_f64x8 = _mm512_add_pd(compensation_f64x8, _mm512_add_pd(sum_error_f64x8, product_error_f64x8));
        }
    }

    // Final: combine sum + compensation before reduce
    nk_f64_t quadratic = _mm512_reduce_add_pd(_mm512_add_pd(sum_f64x8, compensation_f64x8));
    *result = nk_f64_sqrt_haswell(quadratic > 0 ? quadratic : 0);
}

NK_PUBLIC void nk_bilinear_f64c_skylake(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                        nk_f64c_t *results) {

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    // Using Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated summation.
    __m512i const sign_flip_i64x8 = _mm512_set_epi64(                                   //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000, //
        0x8000000000000000, 0x0000000000000000, 0x8000000000000000, 0x0000000000000000  //
    );

    // Default case for arbitrary size `n`
    nk_size_t const tail_length = n % 4;
    nk_size_t const tail_start = n - tail_length;
    __mmask8 const tail_mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, tail_length * 2);
    nk_f64_t sum_real = 0;
    nk_f64_t sum_imag = 0;
    nk_f64_t compensation_real = 0;
    nk_f64_t compensation_imag = 0;

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_t const a_i_real = a[i].real;
        nk_f64_t const a_i_imag = a[i].imag;
        __m512d cb_j_real_f64x8 = _mm512_setzero_pd();
        __m512d cb_j_imag_f64x8 = _mm512_setzero_pd();
        __m512d compensation_real_f64x8 = _mm512_setzero_pd();
        __m512d compensation_imag_f64x8 = _mm512_setzero_pd();
        __m512d b_f64x8, c_f64x8;
        nk_size_t j = 0;

    nk_bilinear_f64c_skylake_cycle:
        if (j + 4 <= n) {
            b_f64x8 = _mm512_loadu_pd((nk_f64_t const *)(b + j));
            c_f64x8 = _mm512_loadu_pd((nk_f64_t const *)(c + i * n + j));
        }
        else {
            b_f64x8 = _mm512_maskz_loadu_pd(tail_mask, (nk_f64_t const *)(b + tail_start));
            c_f64x8 = _mm512_maskz_loadu_pd(tail_mask, (nk_f64_t const *)(c + i * n + tail_start));
        }
        // The real part of the product: b.real * c.real - b.imag * c.imag.
        // The subtraction will be performed later with a sign flip.
        // Inner loop Dot2 for real accumulator
        {
            __m512d product_f64x8 = _mm512_mul_pd(c_f64x8, b_f64x8);
            __m512d product_error_f64x8 = _mm512_fmsub_pd(c_f64x8, b_f64x8, product_f64x8);
            __m512d tentative_sum_f64x8 = _mm512_add_pd(cb_j_real_f64x8, product_f64x8);
            __m512d virtual_addend_f64x8 = _mm512_sub_pd(tentative_sum_f64x8, cb_j_real_f64x8);
            __m512d sum_error_f64x8 = _mm512_add_pd(
                _mm512_sub_pd(cb_j_real_f64x8, _mm512_sub_pd(tentative_sum_f64x8, virtual_addend_f64x8)),
                _mm512_sub_pd(product_f64x8, virtual_addend_f64x8));
            cb_j_real_f64x8 = tentative_sum_f64x8;
            compensation_real_f64x8 = _mm512_add_pd(compensation_real_f64x8,
                                                    _mm512_add_pd(sum_error_f64x8, product_error_f64x8));
        }
        // The imaginary part of the product: b.real * c.imag + b.imag * c.real.
        // Swap the imaginary and real parts of `c` before multiplication:
        c_f64x8 = _mm512_permute_pd(c_f64x8, 0x55); //? Same as 0b01010101.
        // Inner loop Dot2 for imaginary accumulator
        {
            __m512d product_f64x8 = _mm512_mul_pd(c_f64x8, b_f64x8);
            __m512d product_error_f64x8 = _mm512_fmsub_pd(c_f64x8, b_f64x8, product_f64x8);
            __m512d tentative_sum_f64x8 = _mm512_add_pd(cb_j_imag_f64x8, product_f64x8);
            __m512d virtual_addend_f64x8 = _mm512_sub_pd(tentative_sum_f64x8, cb_j_imag_f64x8);
            __m512d sum_error_f64x8 = _mm512_add_pd(
                _mm512_sub_pd(cb_j_imag_f64x8, _mm512_sub_pd(tentative_sum_f64x8, virtual_addend_f64x8)),
                _mm512_sub_pd(product_f64x8, virtual_addend_f64x8));
            cb_j_imag_f64x8 = tentative_sum_f64x8;
            compensation_imag_f64x8 = _mm512_add_pd(compensation_imag_f64x8,
                                                    _mm512_add_pd(sum_error_f64x8, product_error_f64x8));
        }
        j += 4;
        if (j < n) goto nk_bilinear_f64c_skylake_cycle;

        // Flip the sign bit in every second scalar before accumulation:
        cb_j_real_f64x8 = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(cb_j_real_f64x8), sign_flip_i64x8));
        compensation_real_f64x8 = _mm512_castsi512_pd(
            _mm512_xor_si512(_mm512_castpd_si512(compensation_real_f64x8), sign_flip_i64x8));

        // Combine inner sums with compensation before horizontal reduce
        cb_j_real_f64x8 = _mm512_add_pd(cb_j_real_f64x8, compensation_real_f64x8);
        cb_j_imag_f64x8 = _mm512_add_pd(cb_j_imag_f64x8, compensation_imag_f64x8);

        // Horizontal sums are the expensive part of the computation:
        nk_f64_t const cb_j_real = _mm512_reduce_add_pd(cb_j_real_f64x8);
        nk_f64_t const cb_j_imag = _mm512_reduce_add_pd(cb_j_imag_f64x8);

        // Outer loop Dot2 for real part: sum_real += a_i_real * cb_j_real - a_i_imag * cb_j_imag
        {
            // First term: a_i_real * cb_j_real
            nk_f64_t product1 = a_i_real * cb_j_real;
            nk_f64_t product_error1 = (a_i_real * cb_j_real) - product1;
            // Second term: -a_i_imag * cb_j_imag
            nk_f64_t product2 = a_i_imag * cb_j_imag;
            nk_f64_t product_error2 = (a_i_imag * cb_j_imag) - product2;
            // TwoSum for first addition: t = sum_real + product1
            nk_f64_t t1 = sum_real + product1;
            nk_f64_t z1 = t1 - sum_real;
            nk_f64_t sum_error1 = (sum_real - (t1 - z1)) + (product1 - z1);
            sum_real = t1;
            compensation_real += sum_error1 + product_error1;
            // TwoSum for subtraction: t = sum_real - product2
            nk_f64_t t2 = sum_real - product2;
            nk_f64_t z2 = t2 - sum_real;
            nk_f64_t sum_error2 = (sum_real - (t2 - z2)) + (-product2 - z2);
            sum_real = t2;
            compensation_real += sum_error2 - product_error2;
        }

        // Outer loop Dot2 for imaginary part: sum_imag += a_i_real * cb_j_imag + a_i_imag * cb_j_real
        {
            // First term: a_i_real * cb_j_imag
            nk_f64_t product1 = a_i_real * cb_j_imag;
            nk_f64_t product_error1 = (a_i_real * cb_j_imag) - product1;
            // Second term: a_i_imag * cb_j_real
            nk_f64_t product2 = a_i_imag * cb_j_real;
            nk_f64_t product_error2 = (a_i_imag * cb_j_real) - product2;
            // TwoSum for first addition: t = sum_imag + product1
            nk_f64_t t1 = sum_imag + product1;
            nk_f64_t z1 = t1 - sum_imag;
            nk_f64_t sum_error1 = (sum_imag - (t1 - z1)) + (product1 - z1);
            sum_imag = t1;
            compensation_imag += sum_error1 + product_error1;
            // TwoSum for second addition: t = sum_imag + product2
            nk_f64_t t2 = sum_imag + product2;
            nk_f64_t z2 = t2 - sum_imag;
            nk_f64_t sum_error2 = (sum_imag - (t2 - z2)) + (product2 - z2);
            sum_imag = t2;
            compensation_imag += sum_error2 + product_error2;
        }
    }

    // Final: combine sum + compensation
    results->real = sum_real + compensation_real;
    results->imag = sum_imag + compensation_imag;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_
#endif // NK_CURVED_SKYLAKE_H
