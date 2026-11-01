/**
 *  @brief SIMD-accelerated Curved Space Similarity for Haswell.
 *  @file include/numkong/curved/haswell.h
 *  @author Ash Vardanian
 *  @date January 14, 2026
 *
 *  @sa include/numkong/curved.h
 *
 *  Implements f16 and bf16 bilinear forms using AVX2 with F16C conversion.
 */
#ifndef NK_CURVED_HASWELL_H
#define NK_CURVED_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL

#include "numkong/types.h"
#include "numkong/reduce/haswell.h"  // `nk_reduce_add_f32x8_haswell_`
#include "numkong/spatial/haswell.h" // `nk_f32_sqrt_haswell`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC void nk_bilinear_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                       nk_f64_t *result) {
    nk_size_t const tail_length = n % 4;
    nk_size_t const tail_start = n - tail_length;
    __m256d sum_f64x4 = _mm256_setzero_pd();

    for (nk_size_t i = 0; i != n; ++i) {
        __m256d a_f64x4 = _mm256_set1_pd((nk_f64_t)a[i]);
        __m256d cb_j_f64x4 = _mm256_setzero_pd();
        for (nk_size_t j = 0; j + 4 <= n; j += 4) {
            __m256d b_f64x4 = _mm256_cvtps_pd(_mm_loadu_ps(b + j));
            __m256d c_f64x4 = _mm256_cvtps_pd(_mm_loadu_ps(c + i * n + j));
            cb_j_f64x4 = _mm256_fmadd_pd(b_f64x4, c_f64x4, cb_j_f64x4);
        }
        sum_f64x4 = _mm256_fmadd_pd(a_f64x4, cb_j_f64x4, sum_f64x4);
    }

    nk_f64_t sum = nk_reduce_add_f64x4_haswell_(sum_f64x4);
    if (tail_length) {
        nk_b128_vec_t b_tail_vec;
        nk_partial_load_b32x4_haswell_(b + tail_start, &b_tail_vec, tail_length);
        __m256d b_tail_f64x4 = _mm256_cvtps_pd(b_tail_vec.xmm_ps);
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f64_t a_i = (nk_f64_t)a[i];
            nk_b128_vec_t c_tail_vec;
            nk_partial_load_b32x4_haswell_(c + i * n + tail_start, &c_tail_vec, tail_length);
            __m256d c_tail_f64x4 = _mm256_cvtps_pd(c_tail_vec.xmm_ps);
            sum += a_i * nk_reduce_add_f64x4_haswell_(_mm256_mul_pd(b_tail_f64x4, c_tail_f64x4));
        }
    }

    *result = sum;
}

NK_PUBLIC void nk_mahalanobis_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                          nk_f64_t *result) {
    nk_size_t const tail_length = n % 4;
    nk_size_t const tail_start = n - tail_length;
    __m256d sum_f64x4 = _mm256_setzero_pd();

    for (nk_size_t i = 0; i != n; ++i) {
        __m256d diff_i_f64x4 = _mm256_set1_pd((nk_f64_t)a[i] - (nk_f64_t)b[i]);
        __m256d cdiff_j_f64x4 = _mm256_setzero_pd();
        for (nk_size_t j = 0; j + 4 <= n; j += 4) {
            __m256d diff_j_f64x4 = _mm256_sub_pd( //
                _mm256_cvtps_pd(_mm_loadu_ps(a + j)), _mm256_cvtps_pd(_mm_loadu_ps(b + j)));
            __m256d c_f64x4 = _mm256_cvtps_pd(_mm_loadu_ps(c + i * n + j));
            cdiff_j_f64x4 = _mm256_fmadd_pd(diff_j_f64x4, c_f64x4, cdiff_j_f64x4);
        }
        sum_f64x4 = _mm256_fmadd_pd(diff_i_f64x4, cdiff_j_f64x4, sum_f64x4);
    }

    nk_f64_t sum = nk_reduce_add_f64x4_haswell_(sum_f64x4);
    if (tail_length) {
        nk_b128_vec_t a_tail_vec, b_tail_vec;
        nk_partial_load_b32x4_haswell_(a + tail_start, &a_tail_vec, tail_length);
        nk_partial_load_b32x4_haswell_(b + tail_start, &b_tail_vec, tail_length);
        __m256d diff_tail_f64x4 = _mm256_sub_pd(_mm256_cvtps_pd(a_tail_vec.xmm_ps), _mm256_cvtps_pd(b_tail_vec.xmm_ps));
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f64_t diff_i = (nk_f64_t)a[i] - (nk_f64_t)b[i];
            nk_b128_vec_t c_tail_vec;
            nk_partial_load_b32x4_haswell_(c + i * n + tail_start, &c_tail_vec, tail_length);
            __m256d c_tail_f64x4 = _mm256_cvtps_pd(c_tail_vec.xmm_ps);
            sum += diff_i * nk_reduce_add_f64x4_haswell_(_mm256_mul_pd(diff_tail_f64x4, c_tail_f64x4));
        }
    }

    *result = nk_f64_sqrt_haswell(sum > 0 ? sum : 0);
}

NK_PUBLIC void nk_bilinear_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                       nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    for (nk_size_t i = 0; i != n; ++i) {
        __m256 a_f32x8 = _mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(a + i)));
        __m256 cb_j_f32x8 = _mm256_setzero_ps();
        for (nk_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 b_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(b + j)));
            __m256 c_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(c + i * n + j)));
            cb_j_f32x8 = _mm256_fmadd_ps(b_f32x8, c_f32x8, cb_j_f32x8);
        }
        sum_f32x8 = _mm256_fmadd_ps(a_f32x8, cb_j_f32x8, sum_f32x8);
    }

    // Handle the tail of every row
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32_t a_i = _mm256_cvtss_f32(_mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(a + i))));
            nk_b256_vec_t b_vec;
            nk_partial_load_f16x8_to_f32x8_haswell_(b + tail_start, &b_vec, tail_length);
            __m256 b_f32x8 = b_vec.ymm_ps;
            nk_b256_vec_t c_vec;
            nk_partial_load_f16x8_to_f32x8_haswell_(c + i * n + tail_start, &c_vec, tail_length);
            __m256 c_f32x8 = c_vec.ymm_ps;
            nk_f32_t cb_j = nk_reduce_add_f32x8_haswell_(_mm256_mul_ps(b_f32x8, c_f32x8));
            sum += a_i * cb_j;
        }
    }

    *result = sum;
}

NK_PUBLIC void nk_mahalanobis_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                          nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    for (nk_size_t i = 0; i != n; ++i) {
        __m256 diff_i_f32x8 = _mm256_sub_ps(                          //
            _mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(a + i))), //
            _mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(b + i))));
        __m256 cdiff_j_f32x8 = _mm256_setzero_ps();
        for (nk_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 diff_j_f32x8 = _mm256_sub_ps( //
                _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(a + j))),
                _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(b + j))));
            __m256 c_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(c + i * n + j)));
            cdiff_j_f32x8 = _mm256_fmadd_ps(diff_j_f32x8, c_f32x8, cdiff_j_f32x8);
        }
        sum_f32x8 = _mm256_fmadd_ps(diff_i_f32x8, cdiff_j_f32x8, sum_f32x8);
    }

    // Handle the tail of every row
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32_t diff_i = _mm256_cvtss_f32(_mm256_sub_ps(             //
                _mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(a + i))), //
                _mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(b + i)))));
            nk_b256_vec_t a_tail_vec, b_tail_vec;
            nk_partial_load_f16x8_to_f32x8_haswell_(a + tail_start, &a_tail_vec, tail_length);
            nk_partial_load_f16x8_to_f32x8_haswell_(b + tail_start, &b_tail_vec, tail_length);
            __m256 diff_j_f32x8 = _mm256_sub_ps(a_tail_vec.ymm_ps, b_tail_vec.ymm_ps);
            nk_b256_vec_t c_vec;
            nk_partial_load_f16x8_to_f32x8_haswell_(c + i * n + tail_start, &c_vec, tail_length);
            __m256 c_f32x8 = c_vec.ymm_ps;
            nk_f32_t cdiff_j = nk_reduce_add_f32x8_haswell_(_mm256_mul_ps(diff_j_f32x8, c_f32x8));
            sum += diff_i * cdiff_j;
        }
    }

    *result = nk_f32_sqrt_haswell(sum > 0 ? sum : 0);
}

NK_PUBLIC void nk_bilinear_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                        nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    for (nk_size_t i = 0; i != n; ++i) {
        // The `nk_bf16_to_f32_serial` is cheaper than `nk_bf16x8_to_f32x8_haswell_`
        nk_f32_t a_f32;
        nk_bf16_to_f32_serial(a + i, &a_f32);
        __m256 a_f32x8 = _mm256_set1_ps(a_f32);
        __m256 cb_j_f32x8 = _mm256_setzero_ps();
        for (nk_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 b_f32x8 = nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(b + j)));
            __m256 c_f32x8 = nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(c + i * n + j)));
            cb_j_f32x8 = _mm256_fmadd_ps(b_f32x8, c_f32x8, cb_j_f32x8);
        }
        sum_f32x8 = _mm256_fmadd_ps(a_f32x8, cb_j_f32x8, sum_f32x8);
    }

    // Handle the tail of every row
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32_t a_i;
            nk_bf16_to_f32_serial(a + i, &a_i);
            nk_b256_vec_t b_vec;
            nk_partial_load_bf16x8_to_f32x8_haswell_(b + tail_start, &b_vec, tail_length);
            __m256 b_f32x8 = b_vec.ymm_ps;
            nk_b256_vec_t c_vec;
            nk_partial_load_bf16x8_to_f32x8_haswell_(c + i * n + tail_start, &c_vec, tail_length);
            __m256 c_f32x8 = c_vec.ymm_ps;
            nk_f32_t cb_j = nk_reduce_add_f32x8_haswell_(_mm256_mul_ps(b_f32x8, c_f32x8));
            sum += a_i * cb_j;
        }
    }

    *result = sum;
}

NK_PUBLIC void nk_mahalanobis_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                           nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t a_i, b_i;
        nk_bf16_to_f32_serial(a + i, &a_i);
        nk_bf16_to_f32_serial(b + i, &b_i);
        __m256 diff_i_f32x8 = _mm256_sub_ps( //
            _mm256_set1_ps(a_i),             //
            _mm256_set1_ps(b_i));
        __m256 cdiff_j_f32x8 = _mm256_setzero_ps();
        for (nk_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 diff_j_f32x8 = _mm256_sub_ps(                                        //
                nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(a + j))), //
                nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(b + j))));
            __m256 c_f32x8 = nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(c + i * n + j)));
            cdiff_j_f32x8 = _mm256_fmadd_ps(diff_j_f32x8, c_f32x8, cdiff_j_f32x8);
        }
        sum_f32x8 = _mm256_fmadd_ps(diff_i_f32x8, cdiff_j_f32x8, sum_f32x8);
    }

    // Handle the tail of every row
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32_t a_i, b_i;
            nk_bf16_to_f32_serial(a + i, &a_i);
            nk_bf16_to_f32_serial(b + i, &b_i);
            nk_f32_t diff_i = a_i - b_i;
            nk_b256_vec_t a_tail_vec, b_tail_vec;
            nk_partial_load_bf16x8_to_f32x8_haswell_(a + tail_start, &a_tail_vec, tail_length);
            nk_partial_load_bf16x8_to_f32x8_haswell_(b + tail_start, &b_tail_vec, tail_length);
            __m256 diff_j_f32x8 = _mm256_sub_ps(a_tail_vec.ymm_ps, b_tail_vec.ymm_ps);
            nk_b256_vec_t c_vec;
            nk_partial_load_bf16x8_to_f32x8_haswell_(c + i * n + tail_start, &c_vec, tail_length);
            __m256 c_f32x8 = c_vec.ymm_ps;
            nk_f32_t cdiff_j = nk_reduce_add_f32x8_haswell_(_mm256_mul_ps(diff_j_f32x8, c_f32x8));
            sum += diff_i * cdiff_j;
        }
    }

    *result = nk_f32_sqrt_haswell(sum > 0 ? sum : 0);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_
#endif // NK_CURVED_HASWELL_H
