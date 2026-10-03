/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for Skylake.
 *  @file include/numkong/spatial/skylake.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_skylake_instructions Key AVX-512 Spatial Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      _mm512_fmadd_ps             VFMADD132PS (ZMM, ZMM, ZMM)     4cy         0.5/cy      p05
 *      _mm512_sub_ps               VSUBPS (ZMM, ZMM, ZMM)          4cy         0.5/cy      p05
 *      _mm512_rsqrt14_ps           VRSQRT14PS (ZMM, ZMM)           4cy         1/cy        p0
 *      _mm512_sqrt_ps              VSQRTPS (ZMM, ZMM)              12cy        3cy         p0
 *      _mm512_reduce_add_ps        (sequence)                      ~8-10cy     -           -
 *
 *  Distance computations benefit from Skylake-X's dual FMA units achieving 0.5cy throughput for
 *  fused multiply-add operations. VRSQRT14PS provides ~14-bit precision reciprocal square root;
 *  with Newton-Raphson refinement, this exceeds f32's 23-bit mantissa requirements.
 */
#ifndef NK_SPATIAL_SKYLAKE_H
#define NK_SPATIAL_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE

#include "numkong/types.h"
#include "numkong/reduce/skylake.h" // `nk_reduce_add_f32x16_skylake_`
#include "numkong/dot/skylake.h"    // `nk_dot_f64x8_state_skylake_t`

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

/** @brief Reciprocal square root of 16 floats with Newton-Raphson refinement (~28-bit precision). */
NK_INTERNAL __m512 nk_rsqrt_f32x16_skylake_(__m512 x) {
    __m512 rsqrt = _mm512_rsqrt14_ps(x);
    __m512 nr = _mm512_mul_ps(_mm512_mul_ps(x, rsqrt), rsqrt);
    nr = _mm512_sub_ps(_mm512_set1_ps(3.0f), nr);
    return _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.5f), rsqrt), nr);
}

/** @brief Reciprocal square root of 8 doubles with Newton-Raphson refinement (~28-bit precision). */
NK_INTERNAL __m512d nk_rsqrt_f64x8_skylake_(__m512d x) {
    __m512d rsqrt = _mm512_rsqrt14_pd(x);
    __m512d nr = _mm512_mul_pd(_mm512_mul_pd(x, rsqrt), rsqrt);
    nr = _mm512_sub_pd(_mm512_set1_pd(3.0), nr);
    return _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(0.5), rsqrt), nr);
}

#pragma region - Traditional Floats

NK_PUBLIC void nk_sqeuclidean_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    // Upcast to f64 for higher precision accumulation
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m256 a_f32x8, b_f32x8;

nk_sqeuclidean_f32_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x8 = _mm256_maskz_loadu_ps(mask, a);
        b_f32x8 = _mm256_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_f32x8 = _mm256_loadu_ps(a);
        b_f32x8 = _mm256_loadu_ps(b);
        a += 8, b += 8, n -= 8;
    }
    __m512d a_f64x8 = _mm512_cvtps_pd(a_f32x8);
    __m512d b_f64x8 = _mm512_cvtps_pd(b_f32x8);
    __m512d diff_f64x8 = _mm512_sub_pd(a_f64x8, b_f64x8);
    sum_f64x8 = _mm512_fmadd_pd(diff_f64x8, diff_f64x8, sum_f64x8);
    if (n) goto nk_sqeuclidean_f32_skylake_cycle;

    *result = _mm512_reduce_add_pd(sum_f64x8);
}

NK_PUBLIC void nk_euclidean_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_sqeuclidean_f32_skylake(a, b, n, result);
    *result = nk_f64_sqrt_haswell(*result);
}

NK_INTERNAL nk_f64_t nk_angular_normalize_f64_skylake_(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {

    // If both vectors have magnitude 0, the distance is 0.
    if (a2 == 0 && b2 == 0) return 0;
    // If any one of the vectors is 0, the square root of the product is 0,
    // the division is illformed, and the result is 1.
    else if (ab == 0) return 1;

    // Design note: We use exact `_mm_sqrt_pd` instead of `_mm_rsqrt14_pd` approximation.
    // The AVX-512 `_mm_rsqrt14_pd` has max relative error of 2⁻¹⁴ (~14 bits precision).
    // Even with Newton-Raphson refinement (doubles precision to ~28 bits), this is
    // insufficient for f64's 52-bit mantissa, causing ULP errors in the tens of millions.
    // The `_mm_sqrt_pd` instruction provides full f64 precision.
    //
    // Precision comparison for 1536-dimensional vectors:
    //      DType     rsqrt14+NR Error     Exact sqrt Error
    //      float64   1.35e-11 ± 1.85e-11  ~0 (2 ULP max)
    //
    // https://web.archive.org/web/20210208132927/http://assemblyrequired.crashworks.org/timing-square-root/
    __m128d squares_f64x2 = _mm_set_pd(a2, b2);
    __m128d sqrts_f64x2 = _mm_sqrt_pd(squares_f64x2);
    nk_f64_t a_sqrt = _mm_cvtsd_f64(_mm_unpackhi_pd(sqrts_f64x2, sqrts_f64x2));
    nk_f64_t b_sqrt = _mm_cvtsd_f64(sqrts_f64x2);
    nk_f64_t result = 1 - ab / (a_sqrt * b_sqrt);
    return result > 0 ? result : 0;
}

NK_PUBLIC void nk_angular_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    // Upcast to f64 for higher precision accumulation
    __m512d dot_f64x8 = _mm512_setzero_pd();
    __m512d a_norm_sq_f64x8 = _mm512_setzero_pd();
    __m512d b_norm_sq_f64x8 = _mm512_setzero_pd();
    __m256 a_f32x8, b_f32x8;

nk_angular_f32_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x8 = _mm256_maskz_loadu_ps(mask, a);
        b_f32x8 = _mm256_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_f32x8 = _mm256_loadu_ps(a);
        b_f32x8 = _mm256_loadu_ps(b);
        a += 8, b += 8, n -= 8;
    }
    __m512d a_f64x8 = _mm512_cvtps_pd(a_f32x8);
    __m512d b_f64x8 = _mm512_cvtps_pd(b_f32x8);
    dot_f64x8 = _mm512_fmadd_pd(a_f64x8, b_f64x8, dot_f64x8);
    a_norm_sq_f64x8 = _mm512_fmadd_pd(a_f64x8, a_f64x8, a_norm_sq_f64x8);
    b_norm_sq_f64x8 = _mm512_fmadd_pd(b_f64x8, b_f64x8, b_norm_sq_f64x8);
    if (n) goto nk_angular_f32_skylake_cycle;

    nk_f64_t dot_f64 = _mm512_reduce_add_pd(dot_f64x8);
    nk_f64_t a_norm_sq_f64 = _mm512_reduce_add_pd(a_norm_sq_f64x8);
    nk_f64_t b_norm_sq_f64 = _mm512_reduce_add_pd(b_norm_sq_f64x8);
    *result = nk_angular_normalize_f64_skylake_(dot_f64, a_norm_sq_f64, b_norm_sq_f64);
}

NK_PUBLIC void nk_sqeuclidean_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d a_f64x8, b_f64x8;

nk_sqeuclidean_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        b_f64x8 = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    __m512d diff_f64x8 = _mm512_sub_pd(a_f64x8, b_f64x8);
    sum_f64x8 = _mm512_fmadd_pd(diff_f64x8, diff_f64x8, sum_f64x8);
    if (n) goto nk_sqeuclidean_f64_skylake_cycle;

    *result = _mm512_reduce_add_pd(sum_f64x8);
}

NK_PUBLIC void nk_euclidean_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_sqeuclidean_f64_skylake(a, b, n, result);
    *result = nk_f64_sqrt_haswell(*result);
}

NK_PUBLIC void nk_angular_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    // Dot2 (Ogita-Rump-Oishi 2005) for cross-product a × b only - it may have cancellation.
    // Self-products ‖a‖² and ‖b‖² use simple FMA - all terms are non-negative, no cancellation.
    __m512d dot_sum_f64x8 = _mm512_setzero_pd();
    __m512d dot_compensation_f64x8 = _mm512_setzero_pd();
    __m512d a_norm_sq_f64x8 = _mm512_setzero_pd();
    __m512d b_norm_sq_f64x8 = _mm512_setzero_pd();
    __m512d a_f64x8, b_f64x8;

nk_angular_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        b_f64x8 = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    // TwoProd for cross-product: product = a * b, error = fma(a, b, -product)
    __m512d x_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    __m512d product_error_f64x8 = _mm512_fmsub_pd(a_f64x8, b_f64x8, x_f64x8);
    // Neumaier TwoSum: t = sum + x, with masked error recovery
    __m512d tentative_sum_f64x8 = _mm512_add_pd(dot_sum_f64x8, x_f64x8);
    __m512d abs_sum_f64x8 = _mm512_abs_pd(dot_sum_f64x8);
    __m512d abs_x_f64x8 = _mm512_abs_pd(x_f64x8);
    __mmask8 sum_ge_x = _mm512_cmp_pd_mask(abs_sum_f64x8, abs_x_f64x8, _CMP_GE_OQ);
    // z = t - larger, error = smaller - z
    __m512d virtual_addend_f64x8 = _mm512_sub_pd(tentative_sum_f64x8, x_f64x8);
    virtual_addend_f64x8 = _mm512_mask_sub_pd(virtual_addend_f64x8, sum_ge_x, tentative_sum_f64x8, dot_sum_f64x8);
    __m512d sum_error_f64x8 = _mm512_sub_pd(dot_sum_f64x8, virtual_addend_f64x8);
    sum_error_f64x8 = _mm512_mask_sub_pd(sum_error_f64x8, sum_ge_x, x_f64x8, virtual_addend_f64x8);
    dot_sum_f64x8 = tentative_sum_f64x8;
    dot_compensation_f64x8 = _mm512_add_pd(dot_compensation_f64x8, _mm512_add_pd(sum_error_f64x8, product_error_f64x8));
    // Simple FMA for self-products (no cancellation possible)
    a_norm_sq_f64x8 = _mm512_fmadd_pd(a_f64x8, a_f64x8, a_norm_sq_f64x8);
    b_norm_sq_f64x8 = _mm512_fmadd_pd(b_f64x8, b_f64x8, b_norm_sq_f64x8);
    if (n) goto nk_angular_f64_skylake_cycle;

    nk_f64_t dot_product_f64 = nk_dot_stable_sum_f64x8_skylake_(dot_sum_f64x8, dot_compensation_f64x8);
    nk_f64_t a_norm_sq_f64 = _mm512_reduce_add_pd(a_norm_sq_f64x8);
    nk_f64_t b_norm_sq_f64 = _mm512_reduce_add_pd(b_norm_sq_f64x8);
    *result = nk_angular_normalize_f64_skylake_(dot_product_f64, a_norm_sq_f64, b_norm_sq_f64);
}

/** @brief Angular from_dot for native f64: 1 − dot / √(query_sumsq × target_sumsq) for 4 pairs. */
NK_INTERNAL void nk_angular_f64x4_from_dot_skylake_(nk_b256_vec_t dots, nk_f64_t query_sumsq,
                                                    nk_b256_vec_t target_sumsqs, nk_b256_vec_t *results) {
    __m256d dots_f64x4 = dots.ymm_pd;
    __m256d query_sumsq_f64x4 = _mm256_set1_pd(query_sumsq);
    __m256d products_f64x4 = _mm256_mul_pd(query_sumsq_f64x4, target_sumsqs.ymm_pd);
    __m256d sqrt_products_f64x4 = _mm256_sqrt_pd(products_f64x4);
    __m256d normalized_f64x4 = _mm256_div_pd(dots_f64x4, sqrt_products_f64x4);
    __m256d ones_f64x4 = _mm256_set1_pd(1.0);
    __m256d angular_f64x4 = _mm256_sub_pd(ones_f64x4, normalized_f64x4);
    results->ymm_pd = _mm256_max_pd(angular_f64x4, _mm256_setzero_pd());
}

/** @brief Euclidean from_dot for native f64: √(query_sumsq + target_sumsq − 2 × dot) for 4 pairs. */
NK_INTERNAL void nk_euclidean_f64x4_from_dot_skylake_(nk_b256_vec_t dots, nk_f64_t query_sumsq,
                                                      nk_b256_vec_t target_sumsqs, nk_b256_vec_t *results) {
    __m256d dots_f64x4 = dots.ymm_pd;
    __m256d query_sumsq_f64x4 = _mm256_set1_pd(query_sumsq);
    __m256d two_f64x4 = _mm256_set1_pd(2.0);
    __m256d sum_sq_f64x4 = _mm256_add_pd(query_sumsq_f64x4, target_sumsqs.ymm_pd);
    __m256d dist_sq_f64x4 = _mm256_fnmadd_pd(two_f64x4, dots_f64x4, sum_sq_f64x4);
    __m256d zeros_f64x4 = _mm256_setzero_pd();
    __m256d clamped_f64x4 = _mm256_max_pd(dist_sq_f64x4, zeros_f64x4);
    results->ymm_pd = _mm256_sqrt_pd(clamped_f64x4);
}

/** @brief Angular from_dot: f32 dots upcast to f64 for precision. Output via nk_b128_vec_t (f32). */
NK_INTERNAL void nk_angular_through_f64_from_dot_skylake_(nk_b128_vec_t dots, nk_f32_t query_sumsq,
                                                          nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    __m128 dots_f32x4 = dots.xmm_ps;
    __m256d dots_f64x4 = _mm256_cvtps_pd(dots_f32x4);
    __m256d query_sumsq_f64x4 = _mm256_set1_pd((nk_f64_t)query_sumsq);
    __m256d target_sumsqs_f64x4 = _mm256_cvtps_pd(target_sumsqs.xmm_ps);
    __m256d products_f64x4 = _mm256_mul_pd(query_sumsq_f64x4, target_sumsqs_f64x4);
    __m256d sqrt_products_f64x4 = _mm256_sqrt_pd(products_f64x4);
    __m256d normalized_f64x4 = _mm256_div_pd(dots_f64x4, sqrt_products_f64x4);
    __m256d ones_f64x4 = _mm256_set1_pd(1.0);
    __m256d angular_f64x4 = _mm256_sub_pd(ones_f64x4, normalized_f64x4);
    angular_f64x4 = _mm256_max_pd(angular_f64x4, _mm256_setzero_pd());
    results->xmm_ps = _mm256_cvtpd_ps(angular_f64x4);
}

/** @brief Euclidean from_dot: f32 dots upcast to f64 for precision. Output via nk_b128_vec_t (f32). */
NK_INTERNAL void nk_euclidean_through_f64_from_dot_skylake_(nk_b128_vec_t dots, nk_f32_t query_sumsq,
                                                            nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    __m128 dots_f32x4 = dots.xmm_ps;
    __m256d dots_f64x4 = _mm256_cvtps_pd(dots_f32x4);
    __m256d query_sumsq_f64x4 = _mm256_set1_pd((nk_f64_t)query_sumsq);
    __m256d target_sumsqs_f64x4 = _mm256_cvtps_pd(target_sumsqs.xmm_ps);
    __m256d two_f64x4 = _mm256_set1_pd(2.0);
    __m256d sum_sq_f64x4 = _mm256_add_pd(query_sumsq_f64x4, target_sumsqs_f64x4);
    __m256d dist_sq_f64x4 = _mm256_fnmadd_pd(two_f64x4, dots_f64x4, sum_sq_f64x4);
    __m256d zeros_f64x4 = _mm256_setzero_pd();
    __m256d clamped_f64x4 = _mm256_max_pd(dist_sq_f64x4, zeros_f64x4);
    __m256d dist_f64x4 = _mm256_sqrt_pd(clamped_f64x4);
    results->xmm_ps = _mm256_cvtpd_ps(dist_f64x4);
}

#pragma endregion - Traditional Floats
#pragma region - Smaller Floats

NK_PUBLIC void nk_sqeuclidean_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m256i a_f16x16, b_f16x16;

nk_sqeuclidean_f16_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_f16x16 = _mm256_maskz_loadu_epi16(mask, a);
        b_f16x16 = _mm256_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_f16x16 = _mm256_loadu_si256((__m256i const *)a);
        b_f16x16 = _mm256_loadu_si256((__m256i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = _mm512_cvtph_ps(a_f16x16);
    __m512 b_f32x16 = _mm512_cvtph_ps(b_f16x16);
    __m512 diff_f32x16 = _mm512_sub_ps(a_f32x16, b_f32x16);
    sum_f32x16 = _mm512_fmadd_ps(diff_f32x16, diff_f32x16, sum_f32x16);
    if (n) goto nk_sqeuclidean_f16_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_euclidean_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_f16_skylake(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_PUBLIC void nk_angular_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 dot_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m256i a_f16x16, b_f16x16;

nk_angular_f16_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_f16x16 = _mm256_maskz_loadu_epi16(mask, a);
        b_f16x16 = _mm256_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_f16x16 = _mm256_loadu_si256((__m256i const *)a);
        b_f16x16 = _mm256_loadu_si256((__m256i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = _mm512_cvtph_ps(a_f16x16);
    __m512 b_f32x16 = _mm512_cvtph_ps(b_f16x16);
    dot_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, dot_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, a_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, b_norm_sq_f32x16);
    if (n) goto nk_angular_f16_skylake_cycle;

    nk_f32_t dot_f32 = nk_reduce_add_f32x16_skylake_(dot_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(a_norm_sq_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(b_norm_sq_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_PUBLIC void nk_sqeuclidean_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m128i a_e4m3x16, b_e4m3x16;

nk_sqeuclidean_e4m3_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_e4m3x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e4m3x16 = _mm_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e4m3x16 = _mm_loadu_si128((__m128i const *)a);
        b_e4m3x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = nk_e4m3x16_to_f32x16_skylake_(a_e4m3x16);
    __m512 b_f32x16 = nk_e4m3x16_to_f32x16_skylake_(b_e4m3x16);
    __m512 diff_f32x16 = _mm512_sub_ps(a_f32x16, b_f32x16);
    sum_f32x16 = _mm512_fmadd_ps(diff_f32x16, diff_f32x16, sum_f32x16);
    if (n) goto nk_sqeuclidean_e4m3_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_euclidean_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e4m3_skylake(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_PUBLIC void nk_angular_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 dot_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m128i a_e4m3x16, b_e4m3x16;

nk_angular_e4m3_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_e4m3x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e4m3x16 = _mm_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e4m3x16 = _mm_loadu_si128((__m128i const *)a);
        b_e4m3x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = nk_e4m3x16_to_f32x16_skylake_(a_e4m3x16);
    __m512 b_f32x16 = nk_e4m3x16_to_f32x16_skylake_(b_e4m3x16);
    dot_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, dot_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, a_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, b_norm_sq_f32x16);
    if (n) goto nk_angular_e4m3_skylake_cycle;

    nk_f32_t dot_f32 = nk_reduce_add_f32x16_skylake_(dot_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(a_norm_sq_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(b_norm_sq_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_PUBLIC void nk_sqeuclidean_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m128i a_e5m2x16, b_e5m2x16;

nk_sqeuclidean_e5m2_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_e5m2x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e5m2x16 = _mm_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e5m2x16 = _mm_loadu_si128((__m128i const *)a);
        b_e5m2x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = nk_e5m2x16_to_f32x16_skylake_(a_e5m2x16);
    __m512 b_f32x16 = nk_e5m2x16_to_f32x16_skylake_(b_e5m2x16);
    __m512 diff_f32x16 = _mm512_sub_ps(a_f32x16, b_f32x16);
    sum_f32x16 = _mm512_fmadd_ps(diff_f32x16, diff_f32x16, sum_f32x16);
    if (n) goto nk_sqeuclidean_e5m2_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_euclidean_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e5m2_skylake(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_PUBLIC void nk_angular_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 dot_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m128i a_e5m2x16, b_e5m2x16;

nk_angular_e5m2_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_e5m2x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e5m2x16 = _mm_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e5m2x16 = _mm_loadu_si128((__m128i const *)a);
        b_e5m2x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = nk_e5m2x16_to_f32x16_skylake_(a_e5m2x16);
    __m512 b_f32x16 = nk_e5m2x16_to_f32x16_skylake_(b_e5m2x16);
    dot_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, dot_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, a_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, b_norm_sq_f32x16);
    if (n) goto nk_angular_e5m2_skylake_cycle;

    nk_f32_t dot_f32 = nk_reduce_add_f32x16_skylake_(dot_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(a_norm_sq_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(b_norm_sq_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_PUBLIC void nk_sqeuclidean_e2m3_skylake(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m128i a_e2m3x16, b_e2m3x16;

nk_sqeuclidean_e2m3_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_e2m3x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e2m3x16 = _mm_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e2m3x16 = _mm_loadu_si128((__m128i const *)a);
        b_e2m3x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = nk_e2m3x16_to_f32x16_skylake_(a_e2m3x16);
    __m512 b_f32x16 = nk_e2m3x16_to_f32x16_skylake_(b_e2m3x16);
    __m512 diff_f32x16 = _mm512_sub_ps(a_f32x16, b_f32x16);
    sum_f32x16 = _mm512_fmadd_ps(diff_f32x16, diff_f32x16, sum_f32x16);
    if (n) goto nk_sqeuclidean_e2m3_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_euclidean_e2m3_skylake(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e2m3_skylake(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_PUBLIC void nk_angular_e2m3_skylake(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 dot_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m128i a_e2m3x16, b_e2m3x16;

nk_angular_e2m3_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_e2m3x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e2m3x16 = _mm_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e2m3x16 = _mm_loadu_si128((__m128i const *)a);
        b_e2m3x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = nk_e2m3x16_to_f32x16_skylake_(a_e2m3x16);
    __m512 b_f32x16 = nk_e2m3x16_to_f32x16_skylake_(b_e2m3x16);
    dot_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, dot_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, a_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, b_norm_sq_f32x16);
    if (n) goto nk_angular_e2m3_skylake_cycle;

    nk_f32_t dot_f32 = nk_reduce_add_f32x16_skylake_(dot_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(a_norm_sq_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(b_norm_sq_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_PUBLIC void nk_sqeuclidean_e3m2_skylake(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m128i a_e3m2x16, b_e3m2x16;

nk_sqeuclidean_e3m2_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_e3m2x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e3m2x16 = _mm_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e3m2x16 = _mm_loadu_si128((__m128i const *)a);
        b_e3m2x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = nk_e3m2x16_to_f32x16_skylake_(a_e3m2x16);
    __m512 b_f32x16 = nk_e3m2x16_to_f32x16_skylake_(b_e3m2x16);
    __m512 diff_f32x16 = _mm512_sub_ps(a_f32x16, b_f32x16);
    sum_f32x16 = _mm512_fmadd_ps(diff_f32x16, diff_f32x16, sum_f32x16);
    if (n) goto nk_sqeuclidean_e3m2_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_euclidean_e3m2_skylake(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e3m2_skylake(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_PUBLIC void nk_angular_e3m2_skylake(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 dot_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m128i a_e3m2x16, b_e3m2x16;

nk_angular_e3m2_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_e3m2x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e3m2x16 = _mm_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e3m2x16 = _mm_loadu_si128((__m128i const *)a);
        b_e3m2x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = nk_e3m2x16_to_f32x16_skylake_(a_e3m2x16);
    __m512 b_f32x16 = nk_e3m2x16_to_f32x16_skylake_(b_e3m2x16);
    dot_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, dot_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, a_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, b_norm_sq_f32x16);
    if (n) goto nk_angular_e3m2_skylake_cycle;

    nk_f32_t dot_f32 = nk_reduce_add_f32x16_skylake_(dot_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(a_norm_sq_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(b_norm_sq_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_f32, a_norm_sq_f32, b_norm_sq_f32);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma endregion - Smaller Floats
#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_
#endif // NK_SPATIAL_SKYLAKE_H
