/**
 *  @file include/numkong/spatial/skylake.h
 *  @author Ash Vardanian
 *  @date October 7, 2023
 *  @brief SIMD-accelerated spatial similarity measures for Skylake.
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_skylake_instructions Key AVX-512 Spatial Instructions
 *
 *  @verbatim
 *  Intrinsic          Instruction                  Skylake-X         Genoa
 *  _mm512_fmadd_ps    VFMADD132PS (ZMM, ZMM, ZMM)  4cy @ p05         4cy @ p01
 *  _mm512_sub_ps      VSUBPS (ZMM, ZMM, ZMM)       4cy @ p05         3cy @ p23
 *  _mm512_rsqrt14_ps  VRSQRT14PS (ZMM, ZMM)        7cy @ p0+p0+p05   5cy @ p01
 *  _mm512_sqrt_ps     VSQRTPS (ZMM, ZMM)           20cy @ p0+p0+p05  15cy @ p01
 *  @endverbatim
 *
 *  Distance computations benefit from Skylake-X's dual FMA units achieving 0.5cy throughput for
 *  fused multiply-add operations. VRSQRT14PS provides ~14-bit precision reciprocal square root;
 *  with Newton-Raphson refinement, this exceeds f32's 23-bit mantissa requirements.
 */
#ifndef NUMKONG_SPATIAL_SKYLAKE_H
#define NUMKONG_SPATIAL_SKYLAKE_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_SKYLAKE_

#include "numkong/types.h"
#include "numkong/reduce/skylake.h"  // `nk_reduce_add_f32x16_skylake_`
#include "numkong/cast/skylake.h"    // `nk_e4m3x16_to_f32x16_skylake_`
#include "numkong/dot/skylake.h"     // `nk_dot_f64x8_state_skylake_t`
#include "numkong/spatial/haswell.h" // `nk_angular_normalize_f32_haswell_`

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

/** Reciprocal square root of 16 floats with Newton-Raphson refinement (~28-bit precision), 0 for an
 *  infinite input. */
NUMKONG_INLINE __m512 nk_rsqrt_f32x16_skylake_(__m512 x) {
    __m512 rsqrt_f32x16 = _mm512_rsqrt14_ps(x);
    __m512 nr_f32x16 = _mm512_mul_ps(_mm512_mul_ps(x, rsqrt_f32x16), rsqrt_f32x16);
    nr_f32x16 = _mm512_sub_ps(_mm512_set1_ps(3.0f), nr_f32x16);
    // The Newton step turns the estimate of 0 for an infinite input into ∞ × 0 = NaN
    __mmask16 const finite_m16 = _mm512_cmp_ps_mask(x, _mm512_set1_ps(NUMKONG_F32_INF), _CMP_NEQ_UQ);
    return _mm512_maskz_mul_ps(finite_m16, _mm512_mul_ps(_mm512_set1_ps(0.5f), rsqrt_f32x16), nr_f32x16);
}

#pragma region F32 and F64 Floats

/** Squared Euclidean distance between two f32 vectors. */
NUMKONG_INLINE void nk_squared_distance_f32_skylake_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                     nk_f64_t *result) {
    // Upcast to f64 for higher precision accumulation
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m256 a_f32x8, b_f32x8;

nk_sqeuclidean_f32_skylake_cycle:
    if (n < 8) {
        __mmask8 mask_m8 = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x8 = _mm256_maskz_loadu_ps(mask_m8, a);
        b_f32x8 = _mm256_maskz_loadu_ps(mask_m8, b);
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

NUMKONG_INLINE nk_f64_t nk_angular_normalize_f64_skylake_(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {

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

/** Squared Euclidean distance between two f64 vectors, summed in Dot2. */
NUMKONG_INLINE void nk_squared_distance_f64_skylake_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                     nk_f64_t *result) {
    __m512d sum_f64x8 = _mm512_setzero_pd(), compensation_f64x8 = _mm512_setzero_pd();
    __m512d a_f64x8, b_f64x8;

nk_sqeuclidean_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask_m8 = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask_m8, a);
        b_f64x8 = _mm512_maskz_loadu_pd(mask_m8, b);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        b_f64x8 = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    __m512d diff_f64x8 = _mm512_sub_pd(a_f64x8, b_f64x8);
    nk_dot2_f64x8_skylake_(&sum_f64x8, &compensation_f64x8, diff_f64x8, diff_f64x8);
    if (n) goto nk_sqeuclidean_f64_skylake_cycle;

    *result = nk_dot_stable_sum_f64x8_skylake_(sum_f64x8, compensation_f64x8);
}

/** Angular from_dot for native f64: 1 − dot / (√q × √t) for 4 pairs, where q is @p query_sumsq and
 *  t each target's sum of squares, with the rules of the serial variant. Separate square roots
 *  avoid overflowing the product of two finite-but-large norms. */
NUMKONG_INLINE void nk_angular_f64x4_from_dot_skylake_(nk_b256_vec_t const *dots_vec, nk_f64_t query_sumsq,
                                                       nk_b256_vec_t const *target_sumsqs_vec,
                                                       nk_b256_vec_t *result_vec) {
    __m256d const zeros_f64x4 = _mm256_setzero_pd(), ones_f64x4 = _mm256_set1_pd(1.0), dots_f64x4 = dots_vec->ymm_pd;
    __m256d const query_sumsq_f64x4 = _mm256_set1_pd(query_sumsq), target_sumsqs_f64x4 = target_sumsqs_vec->ymm_pd;
    __m256d const norm_f64x4 = _mm256_mul_pd(_mm256_sqrt_pd(query_sumsq_f64x4), _mm256_sqrt_pd(target_sumsqs_f64x4));
    __m256d angular_f64x4 = _mm256_max_pd(_mm256_sub_pd(ones_f64x4, _mm256_div_pd(dots_f64x4, norm_f64x4)),
                                          zeros_f64x4);
    __mmask8 const query_zero_m8 = _mm256_cmp_pd_mask(query_sumsq_f64x4, zeros_f64x4, _CMP_EQ_OQ);
    __mmask8 const target_zero_m8 = _mm256_cmp_pd_mask(target_sumsqs_f64x4, zeros_f64x4, _CMP_EQ_OQ);
    __mmask8 const unit_m8 = query_zero_m8 | target_zero_m8 | _mm256_cmp_pd_mask(dots_f64x4, zeros_f64x4, _CMP_EQ_OQ);
    angular_f64x4 = _mm256_mask_mov_pd(angular_f64x4, unit_m8, ones_f64x4);
    angular_f64x4 = _mm256_mask_mov_pd(angular_f64x4, query_zero_m8 & target_zero_m8, zeros_f64x4);
    // A NaN dot outranks the zero-norm cases
    result_vec->ymm_pd = _mm256_mask_mov_pd(angular_f64x4, _mm256_cmp_pd_mask(dots_f64x4, dots_f64x4, _CMP_UNORD_Q),
                                            dots_f64x4);
}

/** Euclidean from_dot for native f64: √(q + t − 2 × dot) for 4 pairs, where q is @p query_sumsq and
 *  t each target's sum of squares. */
NUMKONG_INLINE void nk_euclidean_f64x4_from_dot_skylake_(nk_b256_vec_t const *dots_vec, nk_f64_t query_sumsq,
                                                         nk_b256_vec_t const *target_sumsqs_vec,
                                                         nk_b256_vec_t *result_vec) {
    __m256d const sum_sq_f64x4 = _mm256_add_pd(_mm256_set1_pd(query_sumsq), target_sumsqs_vec->ymm_pd);
    __m256d const dist_sq_f64x4 = _mm256_fnmadd_pd(_mm256_set1_pd(2.0), dots_vec->ymm_pd, sum_sq_f64x4);
    // Negatives from rounding become 0, while the unordered compare keeps a NaN
    __mmask8 const rooted_m8 = _mm256_cmp_pd_mask(dist_sq_f64x4, _mm256_setzero_pd(), _CMP_NLT_UQ);
    result_vec->ymm_pd = _mm256_maskz_sqrt_pd(rooted_m8, dist_sq_f64x4);
}

#pragma endregion F32 and F64 Floats

#pragma region F16 and BF16 Floats

/** Squared Euclidean distance between two f16 vectors. */
NUMKONG_INLINE void nk_squared_distance_f16_skylake_(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                     nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m256i a_f16x16, b_f16x16;

nk_sqeuclidean_f16_skylake_cycle:
    if (n < 16) {
        __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_f16x16 = _mm256_maskz_loadu_epi16(mask_m16, a);
        b_f16x16 = _mm256_maskz_loadu_epi16(mask_m16, b);
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

/** Squared Euclidean distance between two e4m3 vectors. */
NUMKONG_INLINE void nk_squared_distance_e4m3_skylake_(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                      nk_f32_t *result) {
    // E4M3 has no free widen shift (its 4-bit exponent doesn't line up with F16's 5-bit
    // at bit 10), so we call the Giesen-based 16-lane cast helper twice per iteration and
    // run with two F32 accumulators to break the FMA dependency chain.
    __m512 first_acc_f32x16 = _mm512_setzero_ps();
    __m512 second_acc_f32x16 = _mm512_setzero_ps();
    __m256i a_u8x32, b_u8x32;

nk_sqeuclidean_e4m3_skylake_cycle:
    if (n < 32) {
        __mmask32 mask_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)n);
        a_u8x32 = _mm256_maskz_loadu_epi8(mask_m32, a);
        b_u8x32 = _mm256_maskz_loadu_epi8(mask_m32, b);
        n = 0;
    }
    else {
        a_u8x32 = _mm256_loadu_si256((__m256i const *)a);
        b_u8x32 = _mm256_loadu_si256((__m256i const *)b);
        a += 32, b += 32, n -= 32;
    }
    __m512 a_low_f32x16 = nk_e4m3x16_to_f32x16_skylake_(_mm256_castsi256_si128(a_u8x32));
    __m512 a_high_f32x16 = nk_e4m3x16_to_f32x16_skylake_(_mm256_extracti128_si256(a_u8x32, 1));
    __m512 b_low_f32x16 = nk_e4m3x16_to_f32x16_skylake_(_mm256_castsi256_si128(b_u8x32));
    __m512 b_high_f32x16 = nk_e4m3x16_to_f32x16_skylake_(_mm256_extracti128_si256(b_u8x32, 1));
    __m512 diff_low_f32x16 = _mm512_sub_ps(a_low_f32x16, b_low_f32x16);
    __m512 diff_high_f32x16 = _mm512_sub_ps(a_high_f32x16, b_high_f32x16);
    first_acc_f32x16 = _mm512_fmadd_ps(diff_low_f32x16, diff_low_f32x16, first_acc_f32x16);
    second_acc_f32x16 = _mm512_fmadd_ps(diff_high_f32x16, diff_high_f32x16, second_acc_f32x16);
    if (n) goto nk_sqeuclidean_e4m3_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(_mm512_add_ps(first_acc_f32x16, second_acc_f32x16));
}

/** Squared Euclidean distance between two e5m2 vectors. */
NUMKONG_INLINE void nk_squared_distance_e5m2_skylake_(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                      nk_f32_t *result) {
    // Unpacking E5M2 bytes against zero gives F16 bits; the lane permutation preserves the sum.
    __m512 first_acc_f32x16 = _mm512_setzero_ps();
    __m512 second_acc_f32x16 = _mm512_setzero_ps();
    __m512i const zero_u8x64 = _mm512_setzero_si512();
    __m512i a_u8x64, b_u8x64;

nk_sqeuclidean_e5m2_skylake_cycle:
    if (n < 64) {
        __mmask64 mask_m64 = _bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)n);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, a);
        b_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, b);
        n = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_si512((__m512i const *)a);
        b_u8x64 = _mm512_loadu_si512((__m512i const *)b);
        a += 64, b += 64, n -= 64;
    }
    __m512i a_even_f16x32 = _mm512_unpacklo_epi8(zero_u8x64, a_u8x64);
    __m512i a_odd_f16x32 = _mm512_unpackhi_epi8(zero_u8x64, a_u8x64);
    __m512i b_even_f16x32 = _mm512_unpacklo_epi8(zero_u8x64, b_u8x64);
    __m512i b_odd_f16x32 = _mm512_unpackhi_epi8(zero_u8x64, b_u8x64);

    __m512 a_first_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(a_even_f16x32));
    __m512 a_second_f32x16 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(a_even_f16x32, 1));
    __m512 a_third_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(a_odd_f16x32));
    __m512 a_fourth_f32x16 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(a_odd_f16x32, 1));
    __m512 b_first_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(b_even_f16x32));
    __m512 b_second_f32x16 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(b_even_f16x32, 1));
    __m512 b_third_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(b_odd_f16x32));
    __m512 b_fourth_f32x16 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(b_odd_f16x32, 1));

    __m512 diff_first_f32x16 = _mm512_sub_ps(a_first_f32x16, b_first_f32x16);
    __m512 diff_second_f32x16 = _mm512_sub_ps(a_second_f32x16, b_second_f32x16);
    __m512 diff_third_f32x16 = _mm512_sub_ps(a_third_f32x16, b_third_f32x16);
    __m512 diff_fourth_f32x16 = _mm512_sub_ps(a_fourth_f32x16, b_fourth_f32x16);
    first_acc_f32x16 = _mm512_fmadd_ps(diff_first_f32x16, diff_first_f32x16, first_acc_f32x16);
    second_acc_f32x16 = _mm512_fmadd_ps(diff_second_f32x16, diff_second_f32x16, second_acc_f32x16);
    first_acc_f32x16 = _mm512_fmadd_ps(diff_third_f32x16, diff_third_f32x16, first_acc_f32x16);
    second_acc_f32x16 = _mm512_fmadd_ps(diff_fourth_f32x16, diff_fourth_f32x16, second_acc_f32x16);
    if (n) goto nk_sqeuclidean_e5m2_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(_mm512_add_ps(first_acc_f32x16, second_acc_f32x16));
}

/** Squared Euclidean distance between two e2m3 vectors. */
NUMKONG_INLINE void nk_squared_distance_e2m3_skylake_(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                      nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m128i a_e2m3_u8x16, b_e2m3_u8x16;

nk_sqeuclidean_e2m3_skylake_cycle:
    if (n < 16) {
        __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_e2m3_u8x16 = _mm_maskz_loadu_epi8(mask_m16, a);
        b_e2m3_u8x16 = _mm_maskz_loadu_epi8(mask_m16, b);
        n = 0;
    }
    else {
        a_e2m3_u8x16 = _mm_loadu_si128((__m128i const *)a);
        b_e2m3_u8x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = nk_e2m3x16_to_f32x16_skylake_(a_e2m3_u8x16);
    __m512 b_f32x16 = nk_e2m3x16_to_f32x16_skylake_(b_e2m3_u8x16);
    __m512 diff_f32x16 = _mm512_sub_ps(a_f32x16, b_f32x16);
    sum_f32x16 = _mm512_fmadd_ps(diff_f32x16, diff_f32x16, sum_f32x16);
    if (n) goto nk_sqeuclidean_e2m3_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

/** Squared Euclidean distance between two e3m2 vectors. */
NUMKONG_INLINE void nk_squared_distance_e3m2_skylake_(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                      nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m128i a_e3m2_u8x16, b_e3m2_u8x16;

nk_sqeuclidean_e3m2_skylake_cycle:
    if (n < 16) {
        __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_e3m2_u8x16 = _mm_maskz_loadu_epi8(mask_m16, a);
        b_e3m2_u8x16 = _mm_maskz_loadu_epi8(mask_m16, b);
        n = 0;
    }
    else {
        a_e3m2_u8x16 = _mm_loadu_si128((__m128i const *)a);
        b_e3m2_u8x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = nk_e3m2x16_to_f32x16_skylake_(a_e3m2_u8x16);
    __m512 b_f32x16 = nk_e3m2x16_to_f32x16_skylake_(b_e3m2_u8x16);
    __m512 diff_f32x16 = _mm512_sub_ps(a_f32x16, b_f32x16);
    sum_f32x16 = _mm512_fmadd_ps(diff_f32x16, diff_f32x16, sum_f32x16);
    if (n) goto nk_sqeuclidean_e3m2_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

#pragma endregion F16 and BF16 Floats

#if NUMKONG_TARGET_SKYLAKE

#pragma region F32 and F64 Floats

NUMKONG_API nk_status_t nk_sqeuclidean_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                   nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f32_skylake_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                 nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f32_skylake_(a, b, n, result);
    *result = _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd(*result)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // Upcast to f64 for higher precision accumulation
    __m512d dot_f64x8 = _mm512_setzero_pd();
    __m512d a_norm_sq_f64x8 = _mm512_setzero_pd();
    __m512d b_norm_sq_f64x8 = _mm512_setzero_pd();
    __m256 a_f32x8, b_f32x8;

nk_angular_f32_skylake_cycle:
    if (n < 8) {
        __mmask8 mask_m8 = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x8 = _mm256_maskz_loadu_ps(mask_m8, a);
        b_f32x8 = _mm256_maskz_loadu_ps(mask_m8, b);
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
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                   nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f64_skylake_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                 nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f64_skylake_(a, b, n, result);
    *result = _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd(*result)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    __m512d dot_sum_f64x8 = _mm512_setzero_pd(), dot_compensation_f64x8 = _mm512_setzero_pd();
    __m512d a_norm_sq_f64x8 = _mm512_setzero_pd(), a_compensation_f64x8 = _mm512_setzero_pd();
    __m512d b_norm_sq_f64x8 = _mm512_setzero_pd(), b_compensation_f64x8 = _mm512_setzero_pd();
    __m512d a_f64x8, b_f64x8;

nk_angular_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask_m8 = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask_m8, a);
        b_f64x8 = _mm512_maskz_loadu_pd(mask_m8, b);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        b_f64x8 = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    nk_dot2_f64x8_skylake_(&dot_sum_f64x8, &dot_compensation_f64x8, a_f64x8, b_f64x8);
    nk_dot2_f64x8_skylake_(&a_norm_sq_f64x8, &a_compensation_f64x8, a_f64x8, a_f64x8);
    nk_dot2_f64x8_skylake_(&b_norm_sq_f64x8, &b_compensation_f64x8, b_f64x8, b_f64x8);
    if (n) goto nk_angular_f64_skylake_cycle;

    nk_f64_t dot_product_f64 = nk_dot_stable_sum_f64x8_skylake_(dot_sum_f64x8, dot_compensation_f64x8);
    nk_f64_t a_norm_sq_f64 = nk_dot_stable_sum_f64x8_skylake_(a_norm_sq_f64x8, a_compensation_f64x8);
    nk_f64_t b_norm_sq_f64 = nk_dot_stable_sum_f64x8_skylake_(b_norm_sq_f64x8, b_compensation_f64x8);
    *result = nk_angular_normalize_f64_skylake_(dot_product_f64, a_norm_sq_f64, b_norm_sq_f64);
    return nk_success_k;
}

#pragma endregion F32 and F64 Floats

#pragma region F16 and BF16 Floats

NUMKONG_API nk_status_t nk_sqeuclidean_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                   nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f16_skylake_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f16_skylake_(a, b, n, result);
    *result = _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(*result)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    __m512 dot_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m256i a_f16x16, b_f16x16;

nk_angular_f16_skylake_cycle:
    if (n < 16) {
        __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_f16x16 = _mm256_maskz_loadu_epi16(mask_m16, a);
        b_f16x16 = _mm256_maskz_loadu_epi16(mask_m16, b);
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
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_e4m3_skylake_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_e4m3_skylake_(a, b, n, result);
    *result = _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(*result)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    __m512 dot_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m256i a_u8x32, b_u8x32;

nk_angular_e4m3_skylake_cycle:
    if (n < 32) {
        __mmask32 mask_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)n);
        a_u8x32 = _mm256_maskz_loadu_epi8(mask_m32, a);
        b_u8x32 = _mm256_maskz_loadu_epi8(mask_m32, b);
        n = 0;
    }
    else {
        a_u8x32 = _mm256_loadu_si256((__m256i const *)a);
        b_u8x32 = _mm256_loadu_si256((__m256i const *)b);
        a += 32, b += 32, n -= 32;
    }
    __m512 a_low_f32x16 = nk_e4m3x16_to_f32x16_skylake_(_mm256_castsi256_si128(a_u8x32));
    __m512 a_high_f32x16 = nk_e4m3x16_to_f32x16_skylake_(_mm256_extracti128_si256(a_u8x32, 1));
    __m512 b_low_f32x16 = nk_e4m3x16_to_f32x16_skylake_(_mm256_castsi256_si128(b_u8x32));
    __m512 b_high_f32x16 = nk_e4m3x16_to_f32x16_skylake_(_mm256_extracti128_si256(b_u8x32, 1));
    dot_f32x16 = _mm512_fmadd_ps(a_low_f32x16, b_low_f32x16, dot_f32x16);
    dot_f32x16 = _mm512_fmadd_ps(a_high_f32x16, b_high_f32x16, dot_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_low_f32x16, a_low_f32x16, a_norm_sq_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_high_f32x16, a_high_f32x16, a_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_low_f32x16, b_low_f32x16, b_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_high_f32x16, b_high_f32x16, b_norm_sq_f32x16);
    if (n) goto nk_angular_e4m3_skylake_cycle;

    nk_f32_t dot_f32 = nk_reduce_add_f32x16_skylake_(dot_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(a_norm_sq_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(b_norm_sq_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_f32, a_norm_sq_f32, b_norm_sq_f32);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_e5m2_skylake_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_e5m2_skylake_(a, b, n, result);
    *result = _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(*result)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    __m512 dot_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512i const zero_u8x64 = _mm512_setzero_si512();
    __m512i a_u8x64, b_u8x64;

nk_angular_e5m2_skylake_cycle:
    if (n < 64) {
        __mmask64 mask_m64 = _bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)n);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, a);
        b_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, b);
        n = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_si512((__m512i const *)a);
        b_u8x64 = _mm512_loadu_si512((__m512i const *)b);
        a += 64, b += 64, n -= 64;
    }
    __m512i a_even_f16x32 = _mm512_unpacklo_epi8(zero_u8x64, a_u8x64);
    __m512i a_odd_f16x32 = _mm512_unpackhi_epi8(zero_u8x64, a_u8x64);
    __m512i b_even_f16x32 = _mm512_unpacklo_epi8(zero_u8x64, b_u8x64);
    __m512i b_odd_f16x32 = _mm512_unpackhi_epi8(zero_u8x64, b_u8x64);

    __m512 a_first_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(a_even_f16x32));
    __m512 a_second_f32x16 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(a_even_f16x32, 1));
    __m512 a_third_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(a_odd_f16x32));
    __m512 a_fourth_f32x16 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(a_odd_f16x32, 1));
    __m512 b_first_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(b_even_f16x32));
    __m512 b_second_f32x16 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(b_even_f16x32, 1));
    __m512 b_third_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(b_odd_f16x32));
    __m512 b_fourth_f32x16 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(b_odd_f16x32, 1));

    dot_f32x16 = _mm512_fmadd_ps(a_first_f32x16, b_first_f32x16, dot_f32x16);
    dot_f32x16 = _mm512_fmadd_ps(a_second_f32x16, b_second_f32x16, dot_f32x16);
    dot_f32x16 = _mm512_fmadd_ps(a_third_f32x16, b_third_f32x16, dot_f32x16);
    dot_f32x16 = _mm512_fmadd_ps(a_fourth_f32x16, b_fourth_f32x16, dot_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_first_f32x16, a_first_f32x16, a_norm_sq_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_second_f32x16, a_second_f32x16, a_norm_sq_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_third_f32x16, a_third_f32x16, a_norm_sq_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_fourth_f32x16, a_fourth_f32x16, a_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_first_f32x16, b_first_f32x16, b_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_second_f32x16, b_second_f32x16, b_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_third_f32x16, b_third_f32x16, b_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_fourth_f32x16, b_fourth_f32x16, b_norm_sq_f32x16);
    if (n) goto nk_angular_e5m2_skylake_cycle;

    nk_f32_t dot_f32 = nk_reduce_add_f32x16_skylake_(dot_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(a_norm_sq_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(b_norm_sq_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_f32, a_norm_sq_f32, b_norm_sq_f32);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_e2m3_skylake(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_e2m3_skylake_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_e2m3_skylake(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_e2m3_skylake_(a, b, n, result);
    *result = _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(*result)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_e2m3_skylake(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    __m512 dot_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m128i a_e2m3_u8x16, b_e2m3_u8x16;

nk_angular_e2m3_skylake_cycle:
    if (n < 16) {
        __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_e2m3_u8x16 = _mm_maskz_loadu_epi8(mask_m16, a);
        b_e2m3_u8x16 = _mm_maskz_loadu_epi8(mask_m16, b);
        n = 0;
    }
    else {
        a_e2m3_u8x16 = _mm_loadu_si128((__m128i const *)a);
        b_e2m3_u8x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = nk_e2m3x16_to_f32x16_skylake_(a_e2m3_u8x16);
    __m512 b_f32x16 = nk_e2m3x16_to_f32x16_skylake_(b_e2m3_u8x16);
    dot_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, dot_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, a_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, b_norm_sq_f32x16);
    if (n) goto nk_angular_e2m3_skylake_cycle;

    nk_f32_t dot_f32 = nk_reduce_add_f32x16_skylake_(dot_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(a_norm_sq_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(b_norm_sq_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_f32, a_norm_sq_f32, b_norm_sq_f32);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_e3m2_skylake(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                    nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_e3m2_skylake_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_e3m2_skylake(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                  nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_e3m2_skylake_(a, b, n, result);
    *result = _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(*result)));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_e3m2_skylake(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    __m512 dot_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m128i a_e3m2_u8x16, b_e3m2_u8x16;

nk_angular_e3m2_skylake_cycle:
    if (n < 16) {
        __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_e3m2_u8x16 = _mm_maskz_loadu_epi8(mask_m16, a);
        b_e3m2_u8x16 = _mm_maskz_loadu_epi8(mask_m16, b);
        n = 0;
    }
    else {
        a_e3m2_u8x16 = _mm_loadu_si128((__m128i const *)a);
        b_e3m2_u8x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = nk_e3m2x16_to_f32x16_skylake_(a_e3m2_u8x16);
    __m512 b_f32x16 = nk_e3m2x16_to_f32x16_skylake_(b_e3m2_u8x16);
    dot_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, dot_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, a_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, b_norm_sq_f32x16);
    if (n) goto nk_angular_e3m2_skylake_cycle;

    nk_f32_t dot_f32 = nk_reduce_add_f32x16_skylake_(dot_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(a_norm_sq_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(b_norm_sq_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_f32, a_norm_sq_f32, b_norm_sq_f32);
    return nk_success_k;
}

#pragma endregion F16 and BF16 Floats

#endif // NUMKONG_TARGET_SKYLAKE

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_SKYLAKE_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_SPATIAL_SKYLAKE_H
