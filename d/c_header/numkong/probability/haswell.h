/**
 *  @brief Haswell-accelerated Probability Distribution Similarity Measures.
 *  @file include/numkong/probability/haswell.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/probability.h
 */
#ifndef NK_PROBABILITY_HASWELL_H
#define NK_PROBABILITY_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL

#include "numkong/types.h"
#include "numkong/reduce/haswell.h"  // `nk_reduce_add_f32x8_haswell_`, `nk_reduce_add_f64x4_haswell_`
#include "numkong/spatial/haswell.h" // `nk_f32_sqrt_haswell`, `nk_f64_sqrt_haswell`
#include "numkong/cast/haswell.h"    // `nk_partial_load_f16x8_to_f32x8_haswell_`, `nk_partial_load_b64x4_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

NK_INTERNAL __m256 nk_log2_f32x8_haswell_(__m256 x) {
    // Extracting the exponent
    __m256i bits_i32x8 = _mm256_castps_si256(x);
    __m256i exponent_i32x8 = _mm256_srli_epi32(_mm256_and_si256(bits_i32x8, _mm256_set1_epi32(0x7F800000)), 23);
    exponent_i32x8 = _mm256_sub_epi32(exponent_i32x8, _mm256_set1_epi32(127)); // removing the bias
    __m256 exponent_f32x8 = _mm256_cvtepi32_ps(exponent_i32x8);

    // Extracting the mantissa ∈ [1, 2)
    __m256 mantissa_f32x8 = _mm256_castsi256_ps(
        _mm256_or_si256(_mm256_and_si256(bits_i32x8, _mm256_set1_epi32(0x007FFFFF)), _mm256_set1_epi32(0x3F800000)));

    // Compute log2(m) using the s-series: s = (m-1)/(m+1), s ∈ [0, 1/3] for m ∈ [1, 2)
    // log2(m) = (2/ln2) × s × (1 + s²/3 + s⁴/5 + s⁶/7 + s⁸/9)
    __m256 one_f32x8 = _mm256_set1_ps(1.0f);
    __m256 s_f32x8 = _mm256_div_ps(_mm256_sub_ps(mantissa_f32x8, one_f32x8), _mm256_add_ps(mantissa_f32x8, one_f32x8));
    __m256 s2_f32x8 = _mm256_mul_ps(s_f32x8, s_f32x8);
    __m256 series_f32x8 = _mm256_set1_ps(0.111111111f);                                   // 1/9
    series_f32x8 = _mm256_fmadd_ps(series_f32x8, s2_f32x8, _mm256_set1_ps(0.142857143f)); // 1/7
    series_f32x8 = _mm256_fmadd_ps(series_f32x8, s2_f32x8, _mm256_set1_ps(0.2f));         // 1/5
    series_f32x8 = _mm256_fmadd_ps(series_f32x8, s2_f32x8, _mm256_set1_ps(0.333333333f)); // 1/3
    series_f32x8 = _mm256_fmadd_ps(series_f32x8, s2_f32x8, one_f32x8);                    // 1
    __m256 log2m_f32x8 = _mm256_mul_ps(_mm256_set1_ps(2.885390081777927f), _mm256_mul_ps(s_f32x8, series_f32x8));
    return _mm256_add_ps(log2m_f32x8, exponent_f32x8);
}

NK_INTERNAL __m256d nk_log2_f64x4_haswell_(__m256d x) {
    // Extract exponent via integer shift: (bits >> 52) - 1023
    __m256i bits_i64x4 = _mm256_castpd_si256(x);
    __m256i exponent_i64x4 = _mm256_srli_epi64(bits_i64x4, 52);
    // AVX2 has no _mm256_cvtepi64_pd, so extract lanes and convert
    nk_f64_t exp0 = (nk_f64_t)((nk_i64_t)_mm256_extract_epi64(exponent_i64x4, 0) - 1023);
    nk_f64_t exp1 = (nk_f64_t)((nk_i64_t)_mm256_extract_epi64(exponent_i64x4, 1) - 1023);
    nk_f64_t exp2 = (nk_f64_t)((nk_i64_t)_mm256_extract_epi64(exponent_i64x4, 2) - 1023);
    nk_f64_t exp3 = (nk_f64_t)((nk_i64_t)_mm256_extract_epi64(exponent_i64x4, 3) - 1023);
    __m256d exponent_f64x4 = _mm256_set_pd(exp3, exp2, exp1, exp0);

    // Extract mantissa: clear exponent bits, set exponent to 1023 (= 1.0 bias)
    __m256i mantissa_mask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFFLL);
    __m256i bias = _mm256_set1_epi64x(0x3FF0000000000000LL);
    __m256d mantissa_f64x4 = _mm256_castsi256_pd(_mm256_or_si256(_mm256_and_si256(bits_i64x4, mantissa_mask), bias));

    // s-series: s = (m-1)/(m+1), log2(m) = 2*s*P(s²) * log2(e)
    __m256d one_f64x4 = _mm256_set1_pd(1.0);
    __m256d s_f64x4 = _mm256_div_pd(_mm256_sub_pd(mantissa_f64x4, one_f64x4), _mm256_add_pd(mantissa_f64x4, one_f64x4));
    __m256d s2_f64x4 = _mm256_mul_pd(s_f64x4, s_f64x4);

    // 14-term Horner: P(s²) = 1 + s²/3 + s⁴/5 + ... + s²⁶/27
    __m256d poly_f64x4 = _mm256_set1_pd(1.0 / 27.0);
    poly_f64x4 = _mm256_fmadd_pd(s2_f64x4, poly_f64x4, _mm256_set1_pd(1.0 / 25.0));
    poly_f64x4 = _mm256_fmadd_pd(s2_f64x4, poly_f64x4, _mm256_set1_pd(1.0 / 23.0));
    poly_f64x4 = _mm256_fmadd_pd(s2_f64x4, poly_f64x4, _mm256_set1_pd(1.0 / 21.0));
    poly_f64x4 = _mm256_fmadd_pd(s2_f64x4, poly_f64x4, _mm256_set1_pd(1.0 / 19.0));
    poly_f64x4 = _mm256_fmadd_pd(s2_f64x4, poly_f64x4, _mm256_set1_pd(1.0 / 17.0));
    poly_f64x4 = _mm256_fmadd_pd(s2_f64x4, poly_f64x4, _mm256_set1_pd(1.0 / 15.0));
    poly_f64x4 = _mm256_fmadd_pd(s2_f64x4, poly_f64x4, _mm256_set1_pd(1.0 / 13.0));
    poly_f64x4 = _mm256_fmadd_pd(s2_f64x4, poly_f64x4, _mm256_set1_pd(1.0 / 11.0));
    poly_f64x4 = _mm256_fmadd_pd(s2_f64x4, poly_f64x4, _mm256_set1_pd(1.0 / 9.0));
    poly_f64x4 = _mm256_fmadd_pd(s2_f64x4, poly_f64x4, _mm256_set1_pd(1.0 / 7.0));
    poly_f64x4 = _mm256_fmadd_pd(s2_f64x4, poly_f64x4, _mm256_set1_pd(1.0 / 5.0));
    poly_f64x4 = _mm256_fmadd_pd(s2_f64x4, poly_f64x4, _mm256_set1_pd(1.0 / 3.0));
    poly_f64x4 = _mm256_fmadd_pd(s2_f64x4, poly_f64x4, _mm256_set1_pd(1.0));

    __m256d two_f64x4 = _mm256_set1_pd(2.0);
    __m256d ln_m_f64x4 = _mm256_mul_pd(_mm256_mul_pd(two_f64x4, s_f64x4), poly_f64x4);
    __m256d log2e_f64x4 = _mm256_set1_pd(1.4426950408889634);
    __m256d log2_m_f64x4 = _mm256_mul_pd(ln_m_f64x4, log2e_f64x4);

    return _mm256_add_pd(exponent_f64x4, log2_m_f64x4);
}

NK_PUBLIC void nk_kld_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    __m256 epsilon_f32x8 = _mm256_set1_ps(epsilon);
    __m256 a_f32x8, b_f32x8;

nk_kld_f16_haswell_cycle:
    if (n < 8) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_f16x8_to_f32x8_haswell_(a, &a_vec, n);
        nk_partial_load_f16x8_to_f32x8_haswell_(b, &b_vec, n);
        a_f32x8 = a_vec.ymm_ps;
        b_f32x8 = b_vec.ymm_ps;
        n = 0;
    }
    else {
        a_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)a));
        b_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    __m256 ratio_f32x8 = _mm256_div_ps(_mm256_add_ps(a_f32x8, epsilon_f32x8), _mm256_add_ps(b_f32x8, epsilon_f32x8));
    __m256 log_ratio_f32x8 = nk_log2_f32x8_haswell_(ratio_f32x8);
    __m256 contribution_f32x8 = _mm256_mul_ps(a_f32x8, log_ratio_f32x8);
    sum_f32x8 = _mm256_add_ps(sum_f32x8, contribution_f32x8);
    if (n) goto nk_kld_f16_haswell_cycle;

    nk_f32_t log2_normalizer = 0.6931471805599453f;
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    sum *= log2_normalizer;
    *result = sum;
}

NK_PUBLIC void nk_jsd_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    __m256 epsilon_f32x8 = _mm256_set1_ps(epsilon);
    __m256 sum_f32x8 = _mm256_setzero_ps();
    __m256 a_f32x8, b_f32x8;

nk_jsd_f16_haswell_cycle:
    if (n < 8) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_f16x8_to_f32x8_haswell_(a, &a_vec, n);
        nk_partial_load_f16x8_to_f32x8_haswell_(b, &b_vec, n);
        a_f32x8 = a_vec.ymm_ps;
        b_f32x8 = b_vec.ymm_ps;
        n = 0;
    }
    else {
        a_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)a));
        b_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    __m256 mean_f32x8 = _mm256_mul_ps(_mm256_add_ps(a_f32x8, b_f32x8), _mm256_set1_ps(0.5f)); // M = (P + Q) / 2
    __m256 ratio_a_f32x8 = _mm256_div_ps(_mm256_add_ps(a_f32x8, epsilon_f32x8),
                                         _mm256_add_ps(mean_f32x8, epsilon_f32x8));
    __m256 ratio_b_f32x8 = _mm256_div_ps(_mm256_add_ps(b_f32x8, epsilon_f32x8),
                                         _mm256_add_ps(mean_f32x8, epsilon_f32x8));
    __m256 log_ratio_a_f32x8 = nk_log2_f32x8_haswell_(ratio_a_f32x8);
    __m256 log_ratio_b_f32x8 = nk_log2_f32x8_haswell_(ratio_b_f32x8);
    __m256 contribution_a_f32x8 = _mm256_mul_ps(a_f32x8, log_ratio_a_f32x8);
    __m256 contribution_b_f32x8 = _mm256_mul_ps(b_f32x8, log_ratio_b_f32x8);
    sum_f32x8 = _mm256_add_ps(sum_f32x8, contribution_a_f32x8);
    sum_f32x8 = _mm256_add_ps(sum_f32x8, contribution_b_f32x8);
    if (n) goto nk_jsd_f16_haswell_cycle;

    nk_f32_t log2_normalizer = 0.6931471805599453f;
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    sum *= log2_normalizer / 2;
    *result = sum > 0 ? nk_f32_sqrt_haswell(sum) : 0;
}

NK_PUBLIC void nk_kld_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_f64_t epsilon = NK_F64_DIVISION_EPSILON;
    __m256d epsilon_f64x4 = _mm256_set1_pd(epsilon);
    __m256d sum_f64x4 = _mm256_setzero_pd();
    __m256d compensation_f64x4 = _mm256_setzero_pd();
    __m256d a_f64x4, b_f64x4;

nk_kld_f64_haswell_cycle:
    if (n < 4) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b64x4_haswell_(a, &a_vec, n);
        nk_partial_load_b64x4_haswell_(b, &b_vec, n);
        a_f64x4 = a_vec.ymm_pd;
        b_f64x4 = b_vec.ymm_pd;
        n = 0;
    }
    else {
        a_f64x4 = _mm256_loadu_pd(a);
        b_f64x4 = _mm256_loadu_pd(b);
        n -= 4, a += 4, b += 4;
    }
    __m256d ratio_f64x4 = _mm256_div_pd(_mm256_add_pd(a_f64x4, epsilon_f64x4), _mm256_add_pd(b_f64x4, epsilon_f64x4));
    __m256d log_ratio_f64x4 = nk_log2_f64x4_haswell_(ratio_f64x4);
    __m256d contribution_f64x4 = _mm256_mul_pd(a_f64x4, log_ratio_f64x4);
    // Kahan compensated summation
    __m256d compensated_f64x4 = _mm256_sub_pd(contribution_f64x4, compensation_f64x4);
    __m256d tentative_f64x4 = _mm256_add_pd(sum_f64x4, compensated_f64x4);
    compensation_f64x4 = _mm256_sub_pd(_mm256_sub_pd(tentative_f64x4, sum_f64x4), compensated_f64x4);
    sum_f64x4 = tentative_f64x4;
    if (n) goto nk_kld_f64_haswell_cycle;

    nk_f64_t log2_normalizer = 0.6931471805599453;
    *result = nk_reduce_add_f64x4_haswell_(sum_f64x4) * log2_normalizer;
}

NK_PUBLIC void nk_jsd_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_f64_t epsilon = NK_F64_DIVISION_EPSILON;
    __m256d epsilon_f64x4 = _mm256_set1_pd(epsilon);
    __m256d sum_f64x4 = _mm256_setzero_pd();
    __m256d compensation_f64x4 = _mm256_setzero_pd();
    __m256d a_f64x4, b_f64x4;

nk_jsd_f64_haswell_cycle:
    if (n < 4) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b64x4_haswell_(a, &a_vec, n);
        nk_partial_load_b64x4_haswell_(b, &b_vec, n);
        a_f64x4 = a_vec.ymm_pd;
        b_f64x4 = b_vec.ymm_pd;
        n = 0;
    }
    else {
        a_f64x4 = _mm256_loadu_pd(a);
        b_f64x4 = _mm256_loadu_pd(b);
        n -= 4, a += 4, b += 4;
    }
    __m256d mean_f64x4 = _mm256_mul_pd(_mm256_add_pd(a_f64x4, b_f64x4), _mm256_set1_pd(0.5));
    __m256d ratio_a_f64x4 = _mm256_div_pd(_mm256_add_pd(a_f64x4, epsilon_f64x4),
                                          _mm256_add_pd(mean_f64x4, epsilon_f64x4));
    __m256d ratio_b_f64x4 = _mm256_div_pd(_mm256_add_pd(b_f64x4, epsilon_f64x4),
                                          _mm256_add_pd(mean_f64x4, epsilon_f64x4));
    __m256d log_ratio_a_f64x4 = nk_log2_f64x4_haswell_(ratio_a_f64x4);
    __m256d log_ratio_b_f64x4 = nk_log2_f64x4_haswell_(ratio_b_f64x4);
    __m256d contribution_a_f64x4 = _mm256_mul_pd(a_f64x4, log_ratio_a_f64x4);
    __m256d contribution_b_f64x4 = _mm256_mul_pd(b_f64x4, log_ratio_b_f64x4);
    // Kahan compensated summation for contribution a
    __m256d compensated_a_f64x4 = _mm256_sub_pd(contribution_a_f64x4, compensation_f64x4);
    __m256d tentative_a_f64x4 = _mm256_add_pd(sum_f64x4, compensated_a_f64x4);
    compensation_f64x4 = _mm256_sub_pd(_mm256_sub_pd(tentative_a_f64x4, sum_f64x4), compensated_a_f64x4);
    sum_f64x4 = tentative_a_f64x4;
    // Kahan compensated summation for contribution b
    __m256d compensated_b_f64x4 = _mm256_sub_pd(contribution_b_f64x4, compensation_f64x4);
    __m256d tentative_b_f64x4 = _mm256_add_pd(sum_f64x4, compensated_b_f64x4);
    compensation_f64x4 = _mm256_sub_pd(_mm256_sub_pd(tentative_b_f64x4, sum_f64x4), compensated_b_f64x4);
    sum_f64x4 = tentative_b_f64x4;
    if (n) goto nk_jsd_f64_haswell_cycle;

    nk_f64_t log2_normalizer = 0.6931471805599453;
    nk_f64_t sum = nk_reduce_add_f64x4_haswell_(sum_f64x4);
    sum *= log2_normalizer / 2;
    *result = sum > 0 ? nk_f64_sqrt_haswell(sum) : 0;
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
#endif // NK_PROBABILITY_HASWELL_H
