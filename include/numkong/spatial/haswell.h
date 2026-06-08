/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for Haswell.
 *  @file include/numkong/spatial/haswell.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_haswell_instructions Key AVX2 Spatial Instructions
 *
 *      Intrinsic        Instruction             Haswell    Genoa
 *      _mm256_fmadd_ps  VFMADD (YMM, YMM, YMM)  5cy @ p01  4cy @ p01
 *      _mm256_mul_ps    VMULPS (YMM, YMM, YMM)  5cy @ p01  3cy @ p01
 *      _mm256_add_ps    VADDPS (YMM, YMM, YMM)  3cy @ p01  3cy @ p23
 *      _mm256_sub_ps    VSUBPS (YMM, YMM, YMM)  3cy @ p01  3cy @ p23
 *      _mm_rsqrt_ps     VRSQRTPS (XMM, XMM)     5cy @ p0   4cy @ p01
 *      _mm_sqrt_ps      VSQRTPS (XMM, XMM)      11cy @ p0  15cy @ p01
 *      _mm256_sqrt_ps   VSQRTPS (YMM, YMM)      19cy @ p0  15cy @ p01
 *
 *  For angular distance normalization, `_mm_rsqrt_ps` provides ~12-bit precision (1.5 x 2⁻¹² error).
 *  Newton-Raphson refinement doubles precision to ~22-24 bits, sufficient for f32. For f64 we use
 *  the exact `_mm_sqrt_pd` instruction since fast rsqrt approximations lack f64 precision.
 */
#ifndef NK_SPATIAL_HASWELL_H
#define NK_SPATIAL_HASWELL_H

#if NK_TARGET_X8664_
#if NK_TARGET_HASWELL

#include "numkong/types.h"
#include "numkong/scalar/haswell.h" // `nk_f32_sqrt_haswell`
#include "numkong/dot/haswell.h"    // `nk_dot_f32x4_state_haswell_t`
#include "numkong/reduce/haswell.h" // `nk_reduce_add_f32x8_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

/** @brief Reciprocal square root of 4 floats with Newton-Raphson refinement. */
NK_INTERNAL __m128 nk_rsqrt_f32x4_haswell_(__m128 x) {
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(x);
    __m128 nr_f32x4 = _mm_mul_ps(_mm_mul_ps(x, rsqrt_f32x4), rsqrt_f32x4);
    nr_f32x4 = _mm_sub_ps(_mm_set1_ps(3.0f), nr_f32x4);
    return _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(0.5f), rsqrt_f32x4), nr_f32x4);
}

/** @brief Safe square root of 4 floats with zero-clamping for numerical stability. */
NK_INTERNAL __m128 nk_sqrt_f32x4_haswell_(__m128 x) { return _mm_sqrt_ps(_mm_max_ps(x, _mm_setzero_ps())); }

/** @brief Angular from_dot: computes 1 − dot × rsqrt(query_sumsq × target_sumsq) for 4 pairs. */
NK_INTERNAL void nk_angular_through_f32_from_dot_haswell_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
                                                          nk_b128_vec_t const *target_sumsqs_vec,
                                                          nk_b128_vec_t *result_vec) {
    __m128 dots_f32x4 = dots_vec->xmm_ps;
    __m128 query_sumsq_f32x4 = _mm_set1_ps(query_sumsq);
    __m128 products_f32x4 = _mm_mul_ps(query_sumsq_f32x4, target_sumsqs_vec->xmm_ps);
    __m128 rsqrt_f32x4 = nk_rsqrt_f32x4_haswell_(products_f32x4);
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);
    __m128 angular_f32x4 = _mm_sub_ps(_mm_set1_ps(1.0f), normalized_f32x4);
    result_vec->xmm_ps = _mm_max_ps(angular_f32x4, _mm_setzero_ps());
}

/** @brief Euclidean from_dot: computes √(query_sumsq + target_sumsq − 2 × dot) for 4 pairs. */
NK_INTERNAL void nk_euclidean_through_f32_from_dot_haswell_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
                                                            nk_b128_vec_t const *target_sumsqs_vec,
                                                            nk_b128_vec_t *result_vec) {
    __m128 dots_f32x4 = dots_vec->xmm_ps;
    __m128 query_sumsq_f32x4 = _mm_set1_ps(query_sumsq);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_sumsq_f32x4, target_sumsqs_vec->xmm_ps);
    __m128 dist_sq_f32x4 = _mm_fnmadd_ps(_mm_set1_ps(2.0f), dots_f32x4, sum_sq_f32x4);
    result_vec->xmm_ps = nk_sqrt_f32x4_haswell_(dist_sq_f32x4);
}

/** @brief Angular from_dot for native f64: 1 − dot / √(query_sumsq × target_sumsq) for 4 pairs. */
NK_INTERNAL void nk_angular_through_f64_from_dot_haswell_(nk_b256_vec_t const *dots_vec, nk_f64_t query_sumsq,
                                                          nk_b256_vec_t const *target_sumsqs_vec,
                                                          nk_b256_vec_t *result_vec) {
    __m256d dots_f64x4 = dots_vec->ymm_pd;
    __m256d query_sumsq_f64x4 = _mm256_set1_pd(query_sumsq);
    __m256d products_f64x4 = _mm256_mul_pd(query_sumsq_f64x4, target_sumsqs_vec->ymm_pd);
    __m256d sqrt_products_f64x4 = _mm256_sqrt_pd(products_f64x4);
    __m256d normalized_f64x4 = _mm256_div_pd(dots_f64x4, sqrt_products_f64x4);
    __m256d angular_f64x4 = _mm256_sub_pd(_mm256_set1_pd(1.0), normalized_f64x4);
    result_vec->ymm_pd = _mm256_max_pd(angular_f64x4, _mm256_setzero_pd());
}

/** @brief Euclidean from_dot for native f64: √(query_sumsq + target_sumsq − 2 × dot) for 4 pairs. */
NK_INTERNAL void nk_euclidean_through_f64_from_dot_haswell_(nk_b256_vec_t const *dots_vec, nk_f64_t query_sumsq,
                                                            nk_b256_vec_t const *target_sumsqs_vec,
                                                            nk_b256_vec_t *result_vec) {
    __m256d dots_f64x4 = dots_vec->ymm_pd;
    __m256d query_sumsq_f64x4 = _mm256_set1_pd(query_sumsq);
    __m256d sum_sq_f64x4 = _mm256_add_pd(query_sumsq_f64x4, target_sumsqs_vec->ymm_pd);
    __m256d dist_sq_f64x4 = _mm256_fnmadd_pd(_mm256_set1_pd(2.0), dots_f64x4, sum_sq_f64x4);
    result_vec->ymm_pd = _mm256_sqrt_pd(_mm256_max_pd(dist_sq_f64x4, _mm256_setzero_pd()));
}

/** @brief Angular from_dot for i32 accumulators: cast to f32, rsqrt+NR, clamp. 4 pairs. */
NK_INTERNAL void nk_angular_through_i32_from_dot_haswell_(nk_b128_vec_t const *dots_vec, nk_i32_t query_sumsq,
                                                          nk_b128_vec_t const *target_sumsqs_vec,
                                                          nk_b128_vec_t *result_vec) {
    __m128 dots_f32x4 = _mm_cvtepi32_ps(dots_vec->xmm);
    __m128 query_sumsq_f32x4 = _mm_set1_ps((nk_f32_t)query_sumsq);
    __m128 products_f32x4 = _mm_mul_ps(query_sumsq_f32x4, _mm_cvtepi32_ps(target_sumsqs_vec->xmm));
    __m128 rsqrt_f32x4 = nk_rsqrt_f32x4_haswell_(products_f32x4);
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);
    __m128 angular_f32x4 = _mm_sub_ps(_mm_set1_ps(1.0f), normalized_f32x4);
    result_vec->xmm_ps = _mm_max_ps(angular_f32x4, _mm_setzero_ps());
}

/** @brief Euclidean from_dot for i32 accumulators: cast to f32, then √(a² + b² − 2ab). 4 pairs. */
NK_INTERNAL void nk_euclidean_through_i32_from_dot_haswell_(nk_b128_vec_t const *dots_vec, nk_i32_t query_sumsq,
                                                            nk_b128_vec_t const *target_sumsqs_vec,
                                                            nk_b128_vec_t *result_vec) {
    __m128 dots_f32x4 = _mm_cvtepi32_ps(dots_vec->xmm);
    __m128 query_sumsq_f32x4 = _mm_set1_ps((nk_f32_t)query_sumsq);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_sumsq_f32x4, _mm_cvtepi32_ps(target_sumsqs_vec->xmm));
    __m128 dist_sq_f32x4 = _mm_fnmadd_ps(_mm_set1_ps(2.0f), dots_f32x4, sum_sq_f32x4);
    result_vec->xmm_ps = nk_sqrt_f32x4_haswell_(dist_sq_f32x4);
}

/** @brief Angular from_dot for u32 accumulators: cast to f32, rsqrt+NR, clamp. 4 pairs. */
NK_INTERNAL void nk_angular_through_u32_from_dot_haswell_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                          nk_b128_vec_t const *target_sumsqs_vec,
                                                          nk_b128_vec_t *result_vec) {
    __m128 dots_f32x4 = _mm_cvtepi32_ps(dots_vec->xmm);
    __m128 query_sumsq_f32x4 = _mm_set1_ps((nk_f32_t)query_sumsq);
    __m128 products_f32x4 = _mm_mul_ps(query_sumsq_f32x4, _mm_cvtepi32_ps(target_sumsqs_vec->xmm));
    __m128 rsqrt_f32x4 = nk_rsqrt_f32x4_haswell_(products_f32x4);
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);
    __m128 angular_f32x4 = _mm_sub_ps(_mm_set1_ps(1.0f), normalized_f32x4);
    result_vec->xmm_ps = _mm_max_ps(angular_f32x4, _mm_setzero_ps());
}

/** @brief Euclidean from_dot for u32 accumulators: cast to f32, then √(a² + b² − 2ab). 4 pairs. */
NK_INTERNAL void nk_euclidean_through_u32_from_dot_haswell_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                            nk_b128_vec_t const *target_sumsqs_vec,
                                                            nk_b128_vec_t *result_vec) {
    __m128 dots_f32x4 = _mm_cvtepi32_ps(dots_vec->xmm);
    __m128 query_sumsq_f32x4 = _mm_set1_ps((nk_f32_t)query_sumsq);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_sumsq_f32x4, _mm_cvtepi32_ps(target_sumsqs_vec->xmm));
    __m128 dist_sq_f32x4 = _mm_fnmadd_ps(_mm_set1_ps(2.0f), dots_f32x4, sum_sq_f32x4);
    result_vec->xmm_ps = nk_sqrt_f32x4_haswell_(dist_sq_f32x4);
}

NK_INTERNAL nk_f64_t nk_angular_normalize_f64_haswell_(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {

    // If both vectors have magnitude 0, the distance is 0.
    if (a2 == 0 && b2 == 0) return 0;
    // If any one of the vectors is 0, the square root of the product is 0,
    // the division is illformed, and the result is 1.
    else if (ab == 0) return 1;

    // Design note: We use exact `_mm_sqrt_pd` instead of fast rsqrt approximation.
    // The f32 `_mm_rsqrt_ps` has max relative error of 1.5 × 2⁻¹² (~11 bits precision).
    // Even with Newton-Raphson refinement (doubles precision to ~22-24 bits), this is
    // insufficient for f64's 52-bit mantissa, causing ULP errors in the hundreds of millions.
    // The `_mm_sqrt_pd` instruction has ~13 cycle latency but provides full f64 precision.
    // https://web.archive.org/web/20210208132927/http://assemblyrequired.crashworks.org/timing-square-root/
    __m128d squares_f64x2 = _mm_set_pd(a2, b2);
    __m128d sqrts_f64x2 = _mm_sqrt_pd(squares_f64x2);
    nk_f64_t a_sqrt = _mm_cvtsd_f64(_mm_unpackhi_pd(sqrts_f64x2, sqrts_f64x2));
    nk_f64_t b_sqrt = _mm_cvtsd_f64(sqrts_f64x2);
    nk_f64_t result = 1 - ab / (a_sqrt * b_sqrt);
    return result > 0 ? result : 0;
}

NK_INTERNAL nk_f32_t nk_angular_normalize_f32_haswell_(nk_f32_t ab, nk_f32_t a2, nk_f32_t b2) {

    // If both vectors have magnitude 0, the distance is 0.
    if (a2 == 0.0f && b2 == 0.0f) return 0.0f;
    // If any one of the vectors is 0, the square root of the product is 0,
    // the division is illformed, and the result is 1.
    else if (ab == 0.0f) return 1.0f;

    // Load the squares into an __m128 register for single-precision floating-point operations
    __m128 squares_f32x4 = _mm_set_ps(a2, b2, a2, b2); // We replicate to make use of full register

    // Compute the reciprocal square root of the squares using `_mm_rsqrt_ps` (single-precision)
    __m128 rsqrts_f32x4 = _mm_rsqrt_ps(squares_f32x4);

    // Perform one iteration of Newton-Raphson refinement to improve the precision of rsqrt:
    // Formula: y' = y × (1.5 - 0.5 × x × y × y)
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_halves_f32x4 = _mm_set1_ps(1.5f);
    rsqrts_f32x4 = _mm_mul_ps(
        rsqrts_f32x4,
        _mm_sub_ps(three_halves_f32x4,
                   _mm_mul_ps(half_f32x4, _mm_mul_ps(squares_f32x4, _mm_mul_ps(rsqrts_f32x4, rsqrts_f32x4)))));

    // Extract the reciprocal square roots of a2 and b2 from the __m128 register
    nk_f32_t a2_reciprocal = _mm_cvtss_f32(_mm_shuffle_ps(rsqrts_f32x4, rsqrts_f32x4, _MM_SHUFFLE(0, 0, 0, 1)));
    nk_f32_t b2_reciprocal = _mm_cvtss_f32(rsqrts_f32x4);

    // Calculate the angular distance: 1 - dot_product × a2_reciprocal × b2_reciprocal
    nk_f32_t result = 1.0f - ab * a2_reciprocal * b2_reciprocal;
    return result > 0 ? result : 0;
}

#pragma region F16 and BF16 Floats

NK_PUBLIC void nk_sqeuclidean_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 distance_sq_f32x8 = _mm256_setzero_ps();

nk_sqeuclidean_f16_haswell_cycle:
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
    __m256 diff_f32x8 = _mm256_sub_ps(a_f32x8, b_f32x8);
    distance_sq_f32x8 = _mm256_fmadd_ps(diff_f32x8, diff_f32x8, distance_sq_f32x8);
    if (n) goto nk_sqeuclidean_f16_haswell_cycle;

    *result = nk_reduce_add_f32x8_haswell_(distance_sq_f32x8);
}

NK_PUBLIC void nk_euclidean_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_f16_haswell(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_PUBLIC void nk_angular_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 dot_product_f32x8 = _mm256_setzero_ps(), a_norm_sq_f32x8 = _mm256_setzero_ps(),
           b_norm_sq_f32x8 = _mm256_setzero_ps();

nk_angular_f16_haswell_cycle:
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
    dot_product_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, dot_product_f32x8);
    a_norm_sq_f32x8 = _mm256_fmadd_ps(a_f32x8, a_f32x8, a_norm_sq_f32x8);
    b_norm_sq_f32x8 = _mm256_fmadd_ps(b_f32x8, b_f32x8, b_norm_sq_f32x8);
    if (n) goto nk_angular_f16_haswell_cycle;

    nk_f32_t dot_product_f32 = nk_reduce_add_f32x8_haswell_(dot_product_f32x8);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(a_norm_sq_f32x8);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(b_norm_sq_f32x8);
    *result = nk_angular_normalize_f32_haswell_(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_PUBLIC void nk_sqeuclidean_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256i a_bf16_i16x16, b_bf16_i16x16;
    __m256 distance_sq_f32x8 = _mm256_setzero_ps();
    __m256i mask_high_u32x8 = _mm256_set1_epi32((int)0xFFFF0000);

nk_sqeuclidean_bf16_haswell_cycle:
    if (n < 16) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b16x16_serial_(a, &a_vec, n);
        nk_partial_load_b16x16_serial_(b, &b_vec, n);
        a_bf16_i16x16 = a_vec.ymm;
        b_bf16_i16x16 = b_vec.ymm;
        n = 0;
    }
    else {
        a_bf16_i16x16 = _mm256_loadu_si256((__m256i const *)a);
        b_bf16_i16x16 = _mm256_loadu_si256((__m256i const *)b);
        n -= 16, a += 16, b += 16;
    }
    __m256 a_even_f32x8 = _mm256_castsi256_ps(_mm256_slli_epi32(a_bf16_i16x16, 16));
    __m256 b_even_f32x8 = _mm256_castsi256_ps(_mm256_slli_epi32(b_bf16_i16x16, 16));
    __m256 diff_even_f32x8 = _mm256_sub_ps(a_even_f32x8, b_even_f32x8);
    distance_sq_f32x8 = _mm256_fmadd_ps(diff_even_f32x8, diff_even_f32x8, distance_sq_f32x8);
    __m256 a_odd_f32x8 = _mm256_castsi256_ps(_mm256_and_si256(a_bf16_i16x16, mask_high_u32x8));
    __m256 b_odd_f32x8 = _mm256_castsi256_ps(_mm256_and_si256(b_bf16_i16x16, mask_high_u32x8));
    __m256 diff_odd_f32x8 = _mm256_sub_ps(a_odd_f32x8, b_odd_f32x8);
    distance_sq_f32x8 = _mm256_fmadd_ps(diff_odd_f32x8, diff_odd_f32x8, distance_sq_f32x8);
    if (n) goto nk_sqeuclidean_bf16_haswell_cycle;

    *result = nk_reduce_add_f32x8_haswell_(distance_sq_f32x8);
}

NK_PUBLIC void nk_euclidean_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_bf16_haswell(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_PUBLIC void nk_angular_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256i a_bf16_i16x16, b_bf16_i16x16;
    __m256 dot_product_f32x8 = _mm256_setzero_ps(), a_norm_sq_f32x8 = _mm256_setzero_ps(),
           b_norm_sq_f32x8 = _mm256_setzero_ps();
    __m256i mask_high_u32x8 = _mm256_set1_epi32((int)0xFFFF0000);

nk_angular_bf16_haswell_cycle:
    if (n < 16) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b16x16_serial_(a, &a_vec, n);
        nk_partial_load_b16x16_serial_(b, &b_vec, n);
        a_bf16_i16x16 = a_vec.ymm;
        b_bf16_i16x16 = b_vec.ymm;
        n = 0;
    }
    else {
        a_bf16_i16x16 = _mm256_loadu_si256((__m256i const *)a);
        b_bf16_i16x16 = _mm256_loadu_si256((__m256i const *)b);
        n -= 16, a += 16, b += 16;
    }
    __m256 a_even_f32x8 = _mm256_castsi256_ps(_mm256_slli_epi32(a_bf16_i16x16, 16));
    __m256 b_even_f32x8 = _mm256_castsi256_ps(_mm256_slli_epi32(b_bf16_i16x16, 16));
    dot_product_f32x8 = _mm256_fmadd_ps(a_even_f32x8, b_even_f32x8, dot_product_f32x8);
    a_norm_sq_f32x8 = _mm256_fmadd_ps(a_even_f32x8, a_even_f32x8, a_norm_sq_f32x8);
    b_norm_sq_f32x8 = _mm256_fmadd_ps(b_even_f32x8, b_even_f32x8, b_norm_sq_f32x8);
    __m256 a_odd_f32x8 = _mm256_castsi256_ps(_mm256_and_si256(a_bf16_i16x16, mask_high_u32x8));
    __m256 b_odd_f32x8 = _mm256_castsi256_ps(_mm256_and_si256(b_bf16_i16x16, mask_high_u32x8));
    dot_product_f32x8 = _mm256_fmadd_ps(a_odd_f32x8, b_odd_f32x8, dot_product_f32x8);
    a_norm_sq_f32x8 = _mm256_fmadd_ps(a_odd_f32x8, a_odd_f32x8, a_norm_sq_f32x8);
    b_norm_sq_f32x8 = _mm256_fmadd_ps(b_odd_f32x8, b_odd_f32x8, b_norm_sq_f32x8);
    if (n) goto nk_angular_bf16_haswell_cycle;

    nk_f32_t dot_product_f32 = nk_reduce_add_f32x8_haswell_(dot_product_f32x8);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(a_norm_sq_f32x8);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(b_norm_sq_f32x8);
    *result = nk_angular_normalize_f32_haswell_(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

#pragma endregion F16 and BF16 Floats
#pragma region I8 and U8 Integers

NK_PUBLIC void nk_sqeuclidean_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result) {
    // Optimized i8 L2-squared using saturating subtract + VPMADDWD
    //
    // Approach:
    //   - XOR with 0x80 to reinterpret signed i8 as unsigned u8
    //   - Compute |a-b| using unsigned saturating subtraction: diff = (a ⊖ b) | (b ⊖ a)
    //   - Zero-extend u8→u16 using unpacking (1cy latency @ p5)
    //   - Square using vpmaddwd on u16 values (32 elements/iteration)
    //
    // The XOR bias is needed because subs_epu8 (unsigned) saturates to 0 when the result
    // would be negative, so OR-ing both directions gives the true |a-b|.
    // A naive subs_epi8 (signed) would saturate to -128, corrupting the OR trick.
    //
    // Correctness: For squared distance, |a-b|² = (a-b)², so unsigned absolute differences are valid.
    //              The XOR preserves distances: |a-b| = |(a^0x80) - (b^0x80)|.
    //
    __m256i distance_sq_low_i32x8 = _mm256_setzero_si256();
    __m256i distance_sq_high_i32x8 = _mm256_setzero_si256();
    __m256i const zeros_i8x32 = _mm256_setzero_si256();
    __m256i const bias_i8x32 = _mm256_set1_epi8((char)0x80);
    __m256i diff_low_i16x16, diff_high_i16x16;
    __m256i a_i8x32, b_i8x32, diff_u8x32;

    // Process 32 elements per iteration with 256-bit loads
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        a_i8x32 = _mm256_loadu_si256((__m256i const *)(a + i));
        b_i8x32 = _mm256_loadu_si256((__m256i const *)(b + i));

        // Reinterpret signed i8 as unsigned u8 by flipping the sign bit
        a_i8x32 = _mm256_xor_si256(a_i8x32, bias_i8x32);
        b_i8x32 = _mm256_xor_si256(b_i8x32, bias_i8x32);

        // Compute |a-b| using unsigned saturating subtraction
        // subs_epu8 saturates to 0 if result would be negative
        // OR-ing both directions gives absolute difference as unsigned
        diff_u8x32 = _mm256_or_si256(_mm256_subs_epu8(a_i8x32, b_i8x32), _mm256_subs_epu8(b_i8x32, a_i8x32));

        // Zero-extend to i16 using unpack (1cy @ p5, much faster than cvtepi8_epi16)
        diff_low_i16x16 = _mm256_unpacklo_epi8(diff_u8x32, zeros_i8x32);
        diff_high_i16x16 = _mm256_unpackhi_epi8(diff_u8x32, zeros_i8x32);

        // Multiply and accumulate at i16 level, accumulate at i32 level
        distance_sq_low_i32x8 = _mm256_add_epi32(distance_sq_low_i32x8,
                                                 _mm256_madd_epi16(diff_low_i16x16, diff_low_i16x16));
        distance_sq_high_i32x8 = _mm256_add_epi32(distance_sq_high_i32x8,
                                                  _mm256_madd_epi16(diff_high_i16x16, diff_high_i16x16));
    }

    // Reduce to scalar
    nk_i32_t distance_sq_i32 = nk_reduce_add_i32x8_haswell_(
        _mm256_add_epi32(distance_sq_low_i32x8, distance_sq_high_i32x8));

    // Take care of the tail:
    for (; i < n; ++i) {
        nk_i32_t diff_i32 = (nk_i32_t)(a[i]) - b[i];
        distance_sq_i32 += diff_i32 * diff_i32;
    }

    *result = (nk_u32_t)distance_sq_i32;
}

NK_PUBLIC void nk_euclidean_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq_u32;
    nk_sqeuclidean_i8_haswell(a, b, n, &distance_sq_u32);
    *result = nk_f32_sqrt_haswell((nk_f32_t)distance_sq_u32);
}

NK_PUBLIC void nk_angular_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {

    __m256i dot_product_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_i32x8 = _mm256_setzero_si256();

    // AVX2 has no instructions for 8-bit signed integer dot products,
    // but it has a weird instruction for mixed signed-unsigned 8-bit dot product.
    // So we need to normalize the first vector to its absolute value,
    // and shift the product sign into the second vector.
    //
    //      __m256i a_i8_abs_vec = _mm256_abs_epi8(a_i8_vec);
    //      __m256i b_i8_flipped_vec = _mm256_sign_epi8(b_i8_vec, a_i8_vec);
    //      __m256i ab_i16_vec = _mm256_maddubs_epi16(a_i8_abs_vec, b_i8_flipped_vec);
    //
    // The problem with this approach, however, is the `-128` value in the second vector.
    // Flipping its sign will do nothing, and the result will be incorrect.
    // This can easily lead to noticeable numerical errors in the final result.
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i a_i8x16 = _mm_loadu_si128((__m128i const *)(a + i));
        __m128i b_i8x16 = _mm_loadu_si128((__m128i const *)(b + i));

        // Sign extend i8 → i16 directly (128-bit → 256-bit, no port 5 pressure)
        __m256i a_i16x16 = _mm256_cvtepi8_epi16(a_i8x16);
        __m256i b_i16x16 = _mm256_cvtepi8_epi16(b_i8x16);

        // Multiply and accumulate as i16 pairs, accumulate products as i32:
        dot_product_i32x8 = _mm256_add_epi32(dot_product_i32x8, _mm256_madd_epi16(a_i16x16, b_i16x16));
        a_norm_sq_i32x8 = _mm256_add_epi32(a_norm_sq_i32x8, _mm256_madd_epi16(a_i16x16, a_i16x16));
        b_norm_sq_i32x8 = _mm256_add_epi32(b_norm_sq_i32x8, _mm256_madd_epi16(b_i16x16, b_i16x16));
    }

    // Reduce to scalar
    nk_i32_t dot_product_i32 = nk_reduce_add_i32x8_haswell_(dot_product_i32x8);
    nk_i32_t a_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(a_norm_sq_i32x8);
    nk_i32_t b_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(b_norm_sq_i32x8);

    // Take care of the tail:
    for (; i < n; ++i) {
        nk_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = nk_angular_normalize_f32_haswell_((nk_f32_t)dot_product_i32, (nk_f32_t)a_norm_sq_i32,
                                                (nk_f32_t)b_norm_sq_i32);
}

NK_PUBLIC void nk_sqeuclidean_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {

    __m256i distance_sq_low_i32x8 = _mm256_setzero_si256();
    __m256i distance_sq_high_i32x8 = _mm256_setzero_si256();
    __m256i const zeros_i8x32 = _mm256_setzero_si256();

    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_u8x32 = _mm256_loadu_si256((__m256i const *)(a + i));
        __m256i b_u8x32 = _mm256_loadu_si256((__m256i const *)(b + i));

        // Subtracting unsigned vectors in AVX2 is done by saturating subtraction:
        __m256i diff_u8x32 = _mm256_or_si256(_mm256_subs_epu8(a_u8x32, b_u8x32), _mm256_subs_epu8(b_u8x32, a_u8x32));

        // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
        // instructions instead of extracts, as they are much faster and more efficient.
        __m256i diff_low_i16x16 = _mm256_unpacklo_epi8(diff_u8x32, zeros_i8x32);
        __m256i diff_high_i16x16 = _mm256_unpackhi_epi8(diff_u8x32, zeros_i8x32);

        // Multiply and accumulate at `int16` level, accumulate at `int32` level:
        distance_sq_low_i32x8 = _mm256_add_epi32(distance_sq_low_i32x8,
                                                 _mm256_madd_epi16(diff_low_i16x16, diff_low_i16x16));
        distance_sq_high_i32x8 = _mm256_add_epi32(distance_sq_high_i32x8,
                                                  _mm256_madd_epi16(diff_high_i16x16, diff_high_i16x16));
    }

    // Accumulate the 32-bit integers from `distance_sq_high_i32x8` and `distance_sq_low_i32x8`
    nk_i32_t distance_sq_i32 = nk_reduce_add_i32x8_haswell_(
        _mm256_add_epi32(distance_sq_low_i32x8, distance_sq_high_i32x8));

    // Take care of the tail:
    for (; i < n; ++i) {
        nk_i32_t diff_i32 = (nk_i32_t)(a[i]) - b[i];
        distance_sq_i32 += diff_i32 * diff_i32;
    }

    *result = (nk_u32_t)distance_sq_i32;
}

NK_PUBLIC void nk_euclidean_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq_u32;
    nk_sqeuclidean_u8_haswell(a, b, n, &distance_sq_u32);
    *result = nk_f32_sqrt_haswell((nk_f32_t)distance_sq_u32);
}

NK_PUBLIC void nk_angular_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {

    __m256i dot_product_low_i32x8 = _mm256_setzero_si256();
    __m256i dot_product_high_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_low_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_high_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_low_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_high_i32x8 = _mm256_setzero_si256();
    __m256i const zeros_i8x32 = _mm256_setzero_si256();

    // AVX2 has no instructions for 8-bit signed integer dot products,
    // but it has a weird instruction for mixed signed-unsigned 8-bit dot product.
    // So we need to normalize the first vector to its absolute value,
    // and shift the product sign into the second vector.
    //
    //      __m256i a_i8_abs_vec = _mm256_abs_epi8(a_i8_vec);
    //      __m256i b_i8_flipped_vec = _mm256_sign_epi8(b_i8_vec, a_i8_vec);
    //      __m256i ab_i16_vec = _mm256_maddubs_epi16(a_i8_abs_vec, b_i8_flipped_vec);
    //
    // The problem with this approach, however, is the `-128` value in the second vector.
    // Flipping its sign will do nothing, and the result will be incorrect.
    // This can easily lead to noticeable numerical errors in the final result.
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_u8x32 = _mm256_loadu_si256((__m256i const *)(a + i));
        __m256i b_u8x32 = _mm256_loadu_si256((__m256i const *)(b + i));

        // Upcast `uint8` to `int16`. Unlike the signed version, we can use the unpacking
        // instructions instead of extracts, as they are much faster and more efficient.
        __m256i a_low_i16x16 = _mm256_unpacklo_epi8(a_u8x32, zeros_i8x32);
        __m256i a_high_i16x16 = _mm256_unpackhi_epi8(a_u8x32, zeros_i8x32);
        __m256i b_low_i16x16 = _mm256_unpacklo_epi8(b_u8x32, zeros_i8x32);
        __m256i b_high_i16x16 = _mm256_unpackhi_epi8(b_u8x32, zeros_i8x32);

        // Multiply and accumulate as `int16`, accumulate products as `int32`
        dot_product_low_i32x8 = _mm256_add_epi32(dot_product_low_i32x8, _mm256_madd_epi16(a_low_i16x16, b_low_i16x16));
        dot_product_high_i32x8 = _mm256_add_epi32(dot_product_high_i32x8,
                                                  _mm256_madd_epi16(a_high_i16x16, b_high_i16x16));
        a_norm_sq_low_i32x8 = _mm256_add_epi32(a_norm_sq_low_i32x8, _mm256_madd_epi16(a_low_i16x16, a_low_i16x16));
        a_norm_sq_high_i32x8 = _mm256_add_epi32(a_norm_sq_high_i32x8, _mm256_madd_epi16(a_high_i16x16, a_high_i16x16));
        b_norm_sq_low_i32x8 = _mm256_add_epi32(b_norm_sq_low_i32x8, _mm256_madd_epi16(b_low_i16x16, b_low_i16x16));
        b_norm_sq_high_i32x8 = _mm256_add_epi32(b_norm_sq_high_i32x8, _mm256_madd_epi16(b_high_i16x16, b_high_i16x16));
    }

    // Further reduce to a single sum for each vector
    nk_i32_t dot_product_i32 = nk_reduce_add_i32x8_haswell_(
        _mm256_add_epi32(dot_product_low_i32x8, dot_product_high_i32x8));
    nk_i32_t a_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(_mm256_add_epi32(a_norm_sq_low_i32x8, a_norm_sq_high_i32x8));
    nk_i32_t b_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(_mm256_add_epi32(b_norm_sq_low_i32x8, b_norm_sq_high_i32x8));

    // Take care of the tail:
    for (; i < n; ++i) {
        nk_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = nk_angular_normalize_f32_haswell_((nk_f32_t)dot_product_i32, (nk_f32_t)a_norm_sq_i32,
                                                (nk_f32_t)b_norm_sq_i32);
}

#pragma endregion I8 and U8 Integers
#pragma region F32 and F64 Floats

NK_PUBLIC void nk_sqeuclidean_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    // Upcast to f64 for higher precision accumulation
    __m256d sum_f64x4 = _mm256_setzero_pd();
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 a_f32x4 = _mm_loadu_ps(a + i);
        __m128 b_f32x4 = _mm_loadu_ps(b + i);
        __m256d a_f64x4 = _mm256_cvtps_pd(a_f32x4);
        __m256d b_f64x4 = _mm256_cvtps_pd(b_f32x4);
        __m256d diff_f64x4 = _mm256_sub_pd(a_f64x4, b_f64x4);
        sum_f64x4 = _mm256_fmadd_pd(diff_f64x4, diff_f64x4, sum_f64x4);
    }

    nk_f64_t sum_f64 = nk_reduce_add_f64x4_haswell_(sum_f64x4);
    for (; i < n; ++i) {
        nk_f64_t diff_f64 = (nk_f64_t)a[i] - b[i];
        sum_f64 += diff_f64 * diff_f64;
    }

    *result = sum_f64;
}

NK_PUBLIC void nk_euclidean_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    // Upcast to f64 for higher precision accumulation, use f64 sqrt before downcasting
    __m256d sum_f64x4 = _mm256_setzero_pd();
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 a_f32x4 = _mm_loadu_ps(a + i);
        __m128 b_f32x4 = _mm_loadu_ps(b + i);
        __m256d a_f64x4 = _mm256_cvtps_pd(a_f32x4);
        __m256d b_f64x4 = _mm256_cvtps_pd(b_f32x4);
        __m256d diff_f64x4 = _mm256_sub_pd(a_f64x4, b_f64x4);
        sum_f64x4 = _mm256_fmadd_pd(diff_f64x4, diff_f64x4, sum_f64x4);
    }

    nk_f64_t sum_f64 = nk_reduce_add_f64x4_haswell_(sum_f64x4);
    for (; i < n; ++i) {
        nk_f64_t diff_f64 = (nk_f64_t)a[i] - b[i];
        sum_f64 += diff_f64 * diff_f64;
    }

    *result = nk_f64_sqrt_haswell(sum_f64);
}

NK_PUBLIC void nk_angular_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    // Upcast to f64 for higher precision accumulation
    __m256d dot_f64x4 = _mm256_setzero_pd();
    __m256d a_norm_sq_f64x4 = _mm256_setzero_pd();
    __m256d b_norm_sq_f64x4 = _mm256_setzero_pd();
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 a_f32x4 = _mm_loadu_ps(a + i);
        __m128 b_f32x4 = _mm_loadu_ps(b + i);
        __m256d a_f64x4 = _mm256_cvtps_pd(a_f32x4);
        __m256d b_f64x4 = _mm256_cvtps_pd(b_f32x4);
        dot_f64x4 = _mm256_fmadd_pd(a_f64x4, b_f64x4, dot_f64x4);
        a_norm_sq_f64x4 = _mm256_fmadd_pd(a_f64x4, a_f64x4, a_norm_sq_f64x4);
        b_norm_sq_f64x4 = _mm256_fmadd_pd(b_f64x4, b_f64x4, b_norm_sq_f64x4);
    }

    nk_f64_t dot_f64 = nk_reduce_add_f64x4_haswell_(dot_f64x4);
    nk_f64_t a_norm_sq_f64 = nk_reduce_add_f64x4_haswell_(a_norm_sq_f64x4);
    nk_f64_t b_norm_sq_f64 = nk_reduce_add_f64x4_haswell_(b_norm_sq_f64x4);
    for (; i < n; ++i) {
        nk_f64_t a_f64 = a[i], b_f64 = b[i];
        dot_f64 += a_f64 * b_f64;
        a_norm_sq_f64 += a_f64 * a_f64;
        b_norm_sq_f64 += b_f64 * b_f64;
    }
    *result = nk_angular_normalize_f64_haswell_(dot_f64, a_norm_sq_f64, b_norm_sq_f64);
}

NK_PUBLIC void nk_sqeuclidean_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    __m256d sum_f64x4 = _mm256_setzero_pd();
    __m256d a_f64x4, b_f64x4;

nk_sqeuclidean_f64_haswell_cycle:
    if (n < 4) {
        nk_b256_vec_t a_tail, b_tail;
        nk_partial_load_b64x4_serial_(a, &a_tail, n);
        nk_partial_load_b64x4_serial_(b, &b_tail, n);
        a_f64x4 = a_tail.ymm_pd;
        b_f64x4 = b_tail.ymm_pd;
        n = 0;
    }
    else {
        a_f64x4 = _mm256_loadu_pd(a);
        b_f64x4 = _mm256_loadu_pd(b);
        a += 4, b += 4, n -= 4;
    }
    __m256d diff_f64x4 = _mm256_sub_pd(a_f64x4, b_f64x4);
    sum_f64x4 = _mm256_fmadd_pd(diff_f64x4, diff_f64x4, sum_f64x4);
    if (n) goto nk_sqeuclidean_f64_haswell_cycle;

    *result = nk_reduce_add_f64x4_haswell_(sum_f64x4);
}

NK_PUBLIC void nk_euclidean_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_sqeuclidean_f64_haswell(a, b, n, result);
    *result = nk_f64_sqrt_haswell(*result);
}

NK_PUBLIC void nk_angular_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    // Dot2 (Ogita-Rump-Oishi 2005) for cross-product a × b only - it may have cancellation.
    // Self-products ‖a‖² and ‖b‖² use simple FMA - all terms are non-negative, no cancellation.
    // Note: For cross-product we use Knuth TwoSum (6 ops) rather than Neumaier with blends (10 ops)
    // since products can be signed and Knuth handles any operand ordering efficiently.
    __m256d dot_sum_f64x4 = _mm256_setzero_pd();
    __m256d dot_compensation_f64x4 = _mm256_setzero_pd();
    __m256d a_norm_sq_f64x4 = _mm256_setzero_pd();
    __m256d b_norm_sq_f64x4 = _mm256_setzero_pd();
    __m256d a_f64x4, b_f64x4;

nk_angular_f64_haswell_cycle:
    if (n < 4) {
        nk_b256_vec_t a_tail, b_tail;
        nk_partial_load_b64x4_serial_(a, &a_tail, n);
        nk_partial_load_b64x4_serial_(b, &b_tail, n);
        a_f64x4 = a_tail.ymm_pd;
        b_f64x4 = b_tail.ymm_pd;
        n = 0;
    }
    else {
        a_f64x4 = _mm256_loadu_pd(a);
        b_f64x4 = _mm256_loadu_pd(b);
        a += 4, b += 4, n -= 4;
    }
    // TwoProd: product = a × b, error = fma(a, b, -product)
    __m256d x_f64x4 = _mm256_mul_pd(a_f64x4, b_f64x4);
    __m256d product_error_f64x4 = _mm256_fmsub_pd(a_f64x4, b_f64x4, x_f64x4);
    // Knuth TwoSum: error = (sum - (t - z)) + (x - z) where z = t - sum
    __m256d tentative_sum_f64x4 = _mm256_add_pd(dot_sum_f64x4, x_f64x4);
    __m256d virtual_addend_f64x4 = _mm256_sub_pd(tentative_sum_f64x4, dot_sum_f64x4);
    __m256d sum_error_f64x4 = _mm256_add_pd(
        _mm256_sub_pd(dot_sum_f64x4, _mm256_sub_pd(tentative_sum_f64x4, virtual_addend_f64x4)),
        _mm256_sub_pd(x_f64x4, virtual_addend_f64x4));
    dot_sum_f64x4 = tentative_sum_f64x4;
    dot_compensation_f64x4 = _mm256_add_pd(dot_compensation_f64x4, _mm256_add_pd(sum_error_f64x4, product_error_f64x4));
    // Simple FMA for self-products (no cancellation possible)
    a_norm_sq_f64x4 = _mm256_fmadd_pd(a_f64x4, a_f64x4, a_norm_sq_f64x4);
    b_norm_sq_f64x4 = _mm256_fmadd_pd(b_f64x4, b_f64x4, b_norm_sq_f64x4);
    if (n) goto nk_angular_f64_haswell_cycle;

    *result = nk_angular_normalize_f64_haswell_( //
        nk_dot_stable_sum_f64x4_haswell_(dot_sum_f64x4, dot_compensation_f64x4),
        nk_reduce_add_f64x4_haswell_(a_norm_sq_f64x4), nk_reduce_add_f64x4_haswell_(b_norm_sq_f64x4));
}

#pragma endregion F32 and F64 Floats
#pragma region FP8 Floats

NK_PUBLIC void nk_sqeuclidean_e2m3_haswell(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256 distance_sq_f32x8 = _mm256_setzero_ps();

nk_sqeuclidean_e2m3_haswell_cycle:
    if (n < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n);
        nk_partial_load_b8x16_serial_(b, &b_vec, n);
        __m256 a_f32x8 = nk_e2m3x8_to_f32x8_haswell_(a_vec.xmm);
        __m256 b_f32x8 = nk_e2m3x8_to_f32x8_haswell_(b_vec.xmm);
        __m256 diff_f32x8 = _mm256_sub_ps(a_f32x8, b_f32x8);
        distance_sq_f32x8 = _mm256_fmadd_ps(diff_f32x8, diff_f32x8, distance_sq_f32x8);
    }
    else {
        __m256 a_f32x8 = nk_e2m3x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)a));
        __m256 b_f32x8 = nk_e2m3x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)b));
        __m256 diff_f32x8 = _mm256_sub_ps(a_f32x8, b_f32x8);
        distance_sq_f32x8 = _mm256_fmadd_ps(diff_f32x8, diff_f32x8, distance_sq_f32x8);
        n -= 8, a += 8, b += 8;
        goto nk_sqeuclidean_e2m3_haswell_cycle;
    }

    *result = nk_reduce_add_f32x8_haswell_(distance_sq_f32x8);
}

NK_PUBLIC void nk_euclidean_e2m3_haswell(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e2m3_haswell(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_PUBLIC void nk_angular_e2m3_haswell(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256 dot_product_f32x8 = _mm256_setzero_ps();
    __m256 a_norm_sq_f32x8 = _mm256_setzero_ps();
    __m256 b_norm_sq_f32x8 = _mm256_setzero_ps();

nk_angular_e2m3_haswell_cycle:
    if (n < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n);
        nk_partial_load_b8x16_serial_(b, &b_vec, n);
        __m256 a_f32x8 = nk_e2m3x8_to_f32x8_haswell_(a_vec.xmm);
        __m256 b_f32x8 = nk_e2m3x8_to_f32x8_haswell_(b_vec.xmm);
        dot_product_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, dot_product_f32x8);
        a_norm_sq_f32x8 = _mm256_fmadd_ps(a_f32x8, a_f32x8, a_norm_sq_f32x8);
        b_norm_sq_f32x8 = _mm256_fmadd_ps(b_f32x8, b_f32x8, b_norm_sq_f32x8);
    }
    else {
        __m256 a_f32x8 = nk_e2m3x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)a));
        __m256 b_f32x8 = nk_e2m3x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)b));
        dot_product_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, dot_product_f32x8);
        a_norm_sq_f32x8 = _mm256_fmadd_ps(a_f32x8, a_f32x8, a_norm_sq_f32x8);
        b_norm_sq_f32x8 = _mm256_fmadd_ps(b_f32x8, b_f32x8, b_norm_sq_f32x8);
        n -= 8, a += 8, b += 8;
        goto nk_angular_e2m3_haswell_cycle;
    }

    nk_f32_t dot_product_f32 = nk_reduce_add_f32x8_haswell_(dot_product_f32x8);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(a_norm_sq_f32x8);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(b_norm_sq_f32x8);
    *result = nk_angular_normalize_f32_haswell_(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_PUBLIC void nk_sqeuclidean_e3m2_haswell(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256 distance_sq_f32x8 = _mm256_setzero_ps();

nk_sqeuclidean_e3m2_haswell_cycle:
    if (n < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n);
        nk_partial_load_b8x16_serial_(b, &b_vec, n);
        __m256 a_f32x8 = nk_e3m2x8_to_f32x8_haswell_(a_vec.xmm);
        __m256 b_f32x8 = nk_e3m2x8_to_f32x8_haswell_(b_vec.xmm);
        __m256 diff_f32x8 = _mm256_sub_ps(a_f32x8, b_f32x8);
        distance_sq_f32x8 = _mm256_fmadd_ps(diff_f32x8, diff_f32x8, distance_sq_f32x8);
    }
    else {
        __m256 a_f32x8 = nk_e3m2x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)a));
        __m256 b_f32x8 = nk_e3m2x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)b));
        __m256 diff_f32x8 = _mm256_sub_ps(a_f32x8, b_f32x8);
        distance_sq_f32x8 = _mm256_fmadd_ps(diff_f32x8, diff_f32x8, distance_sq_f32x8);
        n -= 8, a += 8, b += 8;
        goto nk_sqeuclidean_e3m2_haswell_cycle;
    }

    *result = nk_reduce_add_f32x8_haswell_(distance_sq_f32x8);
}

NK_PUBLIC void nk_euclidean_e3m2_haswell(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e3m2_haswell(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_PUBLIC void nk_angular_e3m2_haswell(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256 dot_product_f32x8 = _mm256_setzero_ps();
    __m256 a_norm_sq_f32x8 = _mm256_setzero_ps();
    __m256 b_norm_sq_f32x8 = _mm256_setzero_ps();

nk_angular_e3m2_haswell_cycle:
    if (n < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n);
        nk_partial_load_b8x16_serial_(b, &b_vec, n);
        __m256 a_f32x8 = nk_e3m2x8_to_f32x8_haswell_(a_vec.xmm);
        __m256 b_f32x8 = nk_e3m2x8_to_f32x8_haswell_(b_vec.xmm);
        dot_product_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, dot_product_f32x8);
        a_norm_sq_f32x8 = _mm256_fmadd_ps(a_f32x8, a_f32x8, a_norm_sq_f32x8);
        b_norm_sq_f32x8 = _mm256_fmadd_ps(b_f32x8, b_f32x8, b_norm_sq_f32x8);
    }
    else {
        __m256 a_f32x8 = nk_e3m2x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)a));
        __m256 b_f32x8 = nk_e3m2x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)b));
        dot_product_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, dot_product_f32x8);
        a_norm_sq_f32x8 = _mm256_fmadd_ps(a_f32x8, a_f32x8, a_norm_sq_f32x8);
        b_norm_sq_f32x8 = _mm256_fmadd_ps(b_f32x8, b_f32x8, b_norm_sq_f32x8);
        n -= 8, a += 8, b += 8;
        goto nk_angular_e3m2_haswell_cycle;
    }

    nk_f32_t dot_product_f32 = nk_reduce_add_f32x8_haswell_(dot_product_f32x8);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(a_norm_sq_f32x8);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(b_norm_sq_f32x8);
    *result = nk_angular_normalize_f32_haswell_(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_PUBLIC void nk_sqeuclidean_e4m3_haswell(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    // E4M3 has no free widen shift, so we call the Giesen-based 8-lane cast helper
    // twice per 16-lane iter and run with two F32 accumulators to break the FMA chain.
    __m256 first_acc_f32x8 = _mm256_setzero_ps();
    __m256 second_acc_f32x8 = _mm256_setzero_ps();
    __m128i a_u8x16, b_u8x16;

nk_sqeuclidean_e4m3_haswell_cycle:
    if (n < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n);
        nk_partial_load_b8x16_serial_(b, &b_vec, n);
        a_u8x16 = a_vec.xmm;
        b_u8x16 = b_vec.xmm;
        n = 0;
    }
    else {
        a_u8x16 = _mm_loadu_si128((__m128i const *)a);
        b_u8x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m256 a_low_f32x8 = nk_e4m3x8_to_f32x8_haswell_(a_u8x16);
    __m256 a_high_f32x8 = nk_e4m3x8_to_f32x8_haswell_(_mm_unpackhi_epi64(a_u8x16, a_u8x16));
    __m256 b_low_f32x8 = nk_e4m3x8_to_f32x8_haswell_(b_u8x16);
    __m256 b_high_f32x8 = nk_e4m3x8_to_f32x8_haswell_(_mm_unpackhi_epi64(b_u8x16, b_u8x16));
    __m256 diff_low_f32x8 = _mm256_sub_ps(a_low_f32x8, b_low_f32x8);
    __m256 diff_high_f32x8 = _mm256_sub_ps(a_high_f32x8, b_high_f32x8);
    first_acc_f32x8 = _mm256_fmadd_ps(diff_low_f32x8, diff_low_f32x8, first_acc_f32x8);
    second_acc_f32x8 = _mm256_fmadd_ps(diff_high_f32x8, diff_high_f32x8, second_acc_f32x8);
    if (n) goto nk_sqeuclidean_e4m3_haswell_cycle;

    *result = nk_reduce_add_f32x8_haswell_(_mm256_add_ps(first_acc_f32x8, second_acc_f32x8));
}

NK_PUBLIC void nk_euclidean_e4m3_haswell(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e4m3_haswell(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_PUBLIC void nk_angular_e4m3_haswell(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256 dot_product_f32x8 = _mm256_setzero_ps();
    __m256 a_norm_sq_f32x8 = _mm256_setzero_ps();
    __m256 b_norm_sq_f32x8 = _mm256_setzero_ps();
    __m128i a_u8x16, b_u8x16;

nk_angular_e4m3_haswell_cycle:
    if (n < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n);
        nk_partial_load_b8x16_serial_(b, &b_vec, n);
        a_u8x16 = a_vec.xmm;
        b_u8x16 = b_vec.xmm;
        n = 0;
    }
    else {
        a_u8x16 = _mm_loadu_si128((__m128i const *)a);
        b_u8x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m256 a_low_f32x8 = nk_e4m3x8_to_f32x8_haswell_(a_u8x16);
    __m256 a_high_f32x8 = nk_e4m3x8_to_f32x8_haswell_(_mm_unpackhi_epi64(a_u8x16, a_u8x16));
    __m256 b_low_f32x8 = nk_e4m3x8_to_f32x8_haswell_(b_u8x16);
    __m256 b_high_f32x8 = nk_e4m3x8_to_f32x8_haswell_(_mm_unpackhi_epi64(b_u8x16, b_u8x16));
    dot_product_f32x8 = _mm256_fmadd_ps(a_low_f32x8, b_low_f32x8, dot_product_f32x8);
    dot_product_f32x8 = _mm256_fmadd_ps(a_high_f32x8, b_high_f32x8, dot_product_f32x8);
    a_norm_sq_f32x8 = _mm256_fmadd_ps(a_low_f32x8, a_low_f32x8, a_norm_sq_f32x8);
    a_norm_sq_f32x8 = _mm256_fmadd_ps(a_high_f32x8, a_high_f32x8, a_norm_sq_f32x8);
    b_norm_sq_f32x8 = _mm256_fmadd_ps(b_low_f32x8, b_low_f32x8, b_norm_sq_f32x8);
    b_norm_sq_f32x8 = _mm256_fmadd_ps(b_high_f32x8, b_high_f32x8, b_norm_sq_f32x8);
    if (n) goto nk_angular_e4m3_haswell_cycle;

    nk_f32_t dot_product_f32 = nk_reduce_add_f32x8_haswell_(dot_product_f32x8);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(a_norm_sq_f32x8);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(b_norm_sq_f32x8);
    *result = nk_angular_normalize_f32_haswell_(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_PUBLIC void nk_sqeuclidean_e5m2_haswell(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    // E5M2 shares F16's exponent bias (15): `byte << 8` equals the matching F16 encoding.
    // `vpunpck*bw` against zero is the free widen+shift: zero byte in low half of each
    // 16-bit lane, E5M2 byte in high half. Per-128-bit-lane scrambled; commutative sum
    // reduction is invariant under that.
    __m256 first_acc_f32x8 = _mm256_setzero_ps();
    __m256 second_acc_f32x8 = _mm256_setzero_ps();
    __m128i const zero_u8x16 = _mm_setzero_si128();
    __m128i a_u8x16, b_u8x16;

nk_sqeuclidean_e5m2_haswell_cycle:
    if (n < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n);
        nk_partial_load_b8x16_serial_(b, &b_vec, n);
        a_u8x16 = a_vec.xmm;
        b_u8x16 = b_vec.xmm;
        n = 0;
    }
    else {
        a_u8x16 = _mm_loadu_si128((__m128i const *)a);
        b_u8x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m128i a_even_f16x8 = _mm_unpacklo_epi8(zero_u8x16, a_u8x16);
    __m128i a_odd_f16x8 = _mm_unpackhi_epi8(zero_u8x16, a_u8x16);
    __m128i b_even_f16x8 = _mm_unpacklo_epi8(zero_u8x16, b_u8x16);
    __m128i b_odd_f16x8 = _mm_unpackhi_epi8(zero_u8x16, b_u8x16);
    __m256 a_first_f32x8 = _mm256_cvtph_ps(a_even_f16x8);
    __m256 a_second_f32x8 = _mm256_cvtph_ps(a_odd_f16x8);
    __m256 b_first_f32x8 = _mm256_cvtph_ps(b_even_f16x8);
    __m256 b_second_f32x8 = _mm256_cvtph_ps(b_odd_f16x8);
    __m256 diff_first_f32x8 = _mm256_sub_ps(a_first_f32x8, b_first_f32x8);
    __m256 diff_second_f32x8 = _mm256_sub_ps(a_second_f32x8, b_second_f32x8);
    first_acc_f32x8 = _mm256_fmadd_ps(diff_first_f32x8, diff_first_f32x8, first_acc_f32x8);
    second_acc_f32x8 = _mm256_fmadd_ps(diff_second_f32x8, diff_second_f32x8, second_acc_f32x8);
    if (n) goto nk_sqeuclidean_e5m2_haswell_cycle;

    *result = nk_reduce_add_f32x8_haswell_(_mm256_add_ps(first_acc_f32x8, second_acc_f32x8));
}

NK_PUBLIC void nk_euclidean_e5m2_haswell(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e5m2_haswell(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_PUBLIC void nk_angular_e5m2_haswell(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256 dot_product_f32x8 = _mm256_setzero_ps();
    __m256 a_norm_sq_f32x8 = _mm256_setzero_ps();
    __m256 b_norm_sq_f32x8 = _mm256_setzero_ps();
    __m128i const zero_u8x16 = _mm_setzero_si128();
    __m128i a_u8x16, b_u8x16;

nk_angular_e5m2_haswell_cycle:
    if (n < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n);
        nk_partial_load_b8x16_serial_(b, &b_vec, n);
        a_u8x16 = a_vec.xmm;
        b_u8x16 = b_vec.xmm;
        n = 0;
    }
    else {
        a_u8x16 = _mm_loadu_si128((__m128i const *)a);
        b_u8x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m128i a_even_f16x8 = _mm_unpacklo_epi8(zero_u8x16, a_u8x16);
    __m128i a_odd_f16x8 = _mm_unpackhi_epi8(zero_u8x16, a_u8x16);
    __m128i b_even_f16x8 = _mm_unpacklo_epi8(zero_u8x16, b_u8x16);
    __m128i b_odd_f16x8 = _mm_unpackhi_epi8(zero_u8x16, b_u8x16);
    __m256 a_first_f32x8 = _mm256_cvtph_ps(a_even_f16x8);
    __m256 a_second_f32x8 = _mm256_cvtph_ps(a_odd_f16x8);
    __m256 b_first_f32x8 = _mm256_cvtph_ps(b_even_f16x8);
    __m256 b_second_f32x8 = _mm256_cvtph_ps(b_odd_f16x8);
    dot_product_f32x8 = _mm256_fmadd_ps(a_first_f32x8, b_first_f32x8, dot_product_f32x8);
    dot_product_f32x8 = _mm256_fmadd_ps(a_second_f32x8, b_second_f32x8, dot_product_f32x8);
    a_norm_sq_f32x8 = _mm256_fmadd_ps(a_first_f32x8, a_first_f32x8, a_norm_sq_f32x8);
    a_norm_sq_f32x8 = _mm256_fmadd_ps(a_second_f32x8, a_second_f32x8, a_norm_sq_f32x8);
    b_norm_sq_f32x8 = _mm256_fmadd_ps(b_first_f32x8, b_first_f32x8, b_norm_sq_f32x8);
    b_norm_sq_f32x8 = _mm256_fmadd_ps(b_second_f32x8, b_second_f32x8, b_norm_sq_f32x8);
    if (n) goto nk_angular_e5m2_haswell_cycle;

    nk_f32_t dot_product_f32 = nk_reduce_add_f32x8_haswell_(dot_product_f32x8);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(a_norm_sq_f32x8);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x8_haswell_(b_norm_sq_f32x8);
    *result = nk_angular_normalize_f32_haswell_(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma endregion FP8 Floats
#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X8664_
#endif // NK_SPATIAL_HASWELL_H
