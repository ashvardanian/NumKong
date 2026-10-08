/**
 *  @file include/numkong/spatials/sapphireamx.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *  @brief Batched spatial distances for Sapphire Rapids, AMX, with AVX-512 finalization.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NUMKONG_SPATIALS_SAPPHIREAMX_H
#define NUMKONG_SPATIALS_SAPPHIREAMX_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_SAPPHIREAMX_

#include "numkong/spatial/skylake.h"
#include "numkong/spatial/serial.h"
#include "numkong/dots/serial.h"
#include "numkong/dots/sapphireamx.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                            \
    __attribute__((target(                                                                                               \
        "avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx512vbmi,f16c,fma,bmi,bmi2,amx-tile,amx-bf16,amx-int8"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "avx512vbmi", "f16c", "fma", \
                   "bmi", "bmi2", "amx-tile", "amx-bf16", "amx-int8")
#endif

#pragma region Row Finalize Helpers

/** Angular distances of 16 pairs from dots and target sums of squares, with the serial rules. */
NUMKONG_INLINE __m512 nk_angular_f32x16_from_dot_sapphireamx_(__m512 dots_f32x16, __m512 query_norm_sq_f32x16,
                                                              __m512 query_rsqrt_f32x16,
                                                              __m512 target_norms_sq_f32x16) {
    __m512 zero_f32x16 = _mm512_setzero_ps();
    __m512 target_rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(target_norms_sq_f32x16);
    __m512 rsqrt_f32x16 = _mm512_mul_ps(query_rsqrt_f32x16, target_rsqrt_f32x16);
    __m512 normalized_f32x16 = _mm512_mul_ps(dots_f32x16, rsqrt_f32x16);
    __m512 angular_f32x16 = _mm512_sub_ps(_mm512_set1_ps(1.0f), normalized_f32x16);
    __mmask16 query_zero_m16 = _mm512_cmp_ps_mask(query_norm_sq_f32x16, zero_f32x16, _CMP_EQ_OQ);
    __mmask16 target_zero_m16 = _mm512_cmp_ps_mask(target_norms_sq_f32x16, zero_f32x16, _CMP_EQ_OQ);
    __mmask16 one_m16 = query_zero_m16 | target_zero_m16 | _mm512_cmp_ps_mask(dots_f32x16, zero_f32x16, _CMP_EQ_OQ);
    angular_f32x16 = _mm512_mask_mov_ps(angular_f32x16, one_m16, _mm512_set1_ps(1.0f));
    angular_f32x16 = _mm512_mask_mov_ps(angular_f32x16, query_zero_m16 & target_zero_m16, zero_f32x16);
    angular_f32x16 = _mm512_mask_mov_ps(angular_f32x16, _mm512_cmp_ps_mask(dots_f32x16, dots_f32x16, _CMP_UNORD_Q),
                                        dots_f32x16);
    // `max` returns its second operand for a NaN, which keeps the NaN
    return _mm512_max_ps(zero_f32x16, angular_f32x16);
}

/** Euclidean distances of 16 pairs from dots and sums of squares, keeping a NaN dot. */
NUMKONG_INLINE __m512 nk_euclidean_f32x16_from_dot_sapphireamx_(__m512 dots_f32x16, __m512 query_norm_sq_f32x16,
                                                                __m512 target_norms_sq_f32x16) {
    __m512 sum_norms_f32x16 = _mm512_add_ps(query_norm_sq_f32x16, target_norms_sq_f32x16);
    __m512 dist_sq_f32x16 = _mm512_fnmadd_ps(_mm512_set1_ps(2.0f), dots_f32x16, sum_norms_f32x16);
    return _mm512_sqrt_ps(_mm512_max_ps(_mm512_setzero_ps(), dist_sq_f32x16));
}

/** Angular from 16 i32 dots: ab − d² is exact in u64 lanes, which AVX-512DQ rounds once. */
NUMKONG_INLINE __m512 nk_angular_i32x16_from_dot_sapphireamx_(__m512i dots_i32x16, __m512i query_norm_sq_u32x16,
                                                              __m512i target_norms_sq_u32x16) {
    // Multiplies read the even 32-bit lanes, so the odd ones shift down and interleave back after
    __m512i const interleave_i32x16 = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    __m512i const odd_dots_i32x16 = _mm512_srli_epi64(dots_i32x16, 32);
    __m512i const even_products_u64x8 = _mm512_mul_epu32(query_norm_sq_u32x16, target_norms_sq_u32x16);
    __m512i const odd_products_u64x8 = _mm512_mul_epu32(query_norm_sq_u32x16,
                                                        _mm512_srli_epi64(target_norms_sq_u32x16, 32));
    __m512i const even_gaps_u64x8 = _mm512_sub_epi64(even_products_u64x8, _mm512_mul_epi32(dots_i32x16, dots_i32x16));
    __m512i const odd_gaps_u64x8 = _mm512_sub_epi64(odd_products_u64x8,
                                                    _mm512_mul_epi32(odd_dots_i32x16, odd_dots_i32x16));
    __m512 const products_f32x16 = _mm512_permutex2var_ps(
        _mm512_castps256_ps512(_mm512_cvtepu64_ps(even_products_u64x8)), interleave_i32x16,
        _mm512_castps256_ps512(_mm512_cvtepu64_ps(odd_products_u64x8)));
    __m512 const gaps_f32x16 = _mm512_permutex2var_ps(_mm512_castps256_ps512(_mm512_cvtepu64_ps(even_gaps_u64x8)),
                                                      interleave_i32x16,
                                                      _mm512_castps256_ps512(_mm512_cvtepu64_ps(odd_gaps_u64x8)));
    __m512 const dots_f32x16 = _mm512_cvtepi32_ps(dots_i32x16), norms_f32x16 = _mm512_sqrt_ps(products_f32x16);
    // A positive d gives (ab − d²) / (ab + d · s), the rest (s − d) / s, neither cancelling
    __mmask16 const positive_m16 = _mm512_cmpgt_epi32_mask(dots_i32x16, _mm512_setzero_si512());
    __m512 const numerators_f32x16 = _mm512_mask_mov_ps(_mm512_sub_ps(norms_f32x16, dots_f32x16), positive_m16,
                                                        gaps_f32x16);
    __m512 const denominators_f32x16 = _mm512_mask_fmadd_ps(norms_f32x16, positive_m16, dots_f32x16, products_f32x16);
    __m512 const angular_f32x16 = _mm512_div_ps(numerators_f32x16, denominators_f32x16);
    __mmask16 const query_zero_m16 = _mm512_testn_epi32_mask(query_norm_sq_u32x16, query_norm_sq_u32x16);
    __mmask16 const target_zero_m16 = _mm512_testn_epi32_mask(target_norms_sq_u32x16, target_norms_sq_u32x16);
    __mmask16 const unit_m16 = query_zero_m16 | target_zero_m16 | _mm512_testn_epi32_mask(dots_i32x16, dots_i32x16);
    return _mm512_maskz_mov_ps((__mmask16) ~(query_zero_m16 & target_zero_m16),
                               _mm512_mask_mov_ps(angular_f32x16, unit_m16, _mm512_set1_ps(1.0f)));
}

/** Angular from 16 u32 dots: ab − d² is exact in u64 lanes, which AVX-512DQ rounds once. */
NUMKONG_INLINE __m512 nk_angular_u32x16_from_dot_sapphireamx_(__m512i dots_u32x16, __m512i query_norm_sq_u32x16,
                                                              __m512i target_norms_sq_u32x16) {
    // Multiplies read the even 32-bit lanes, so the odd ones shift down and interleave back after
    __m512i const interleave_i32x16 = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    __m512i const odd_dots_u32x16 = _mm512_srli_epi64(dots_u32x16, 32);
    __m512i const even_products_u64x8 = _mm512_mul_epu32(query_norm_sq_u32x16, target_norms_sq_u32x16);
    __m512i const odd_products_u64x8 = _mm512_mul_epu32(query_norm_sq_u32x16,
                                                        _mm512_srli_epi64(target_norms_sq_u32x16, 32));
    __m512i const even_gaps_u64x8 = _mm512_sub_epi64(even_products_u64x8, _mm512_mul_epu32(dots_u32x16, dots_u32x16));
    __m512i const odd_gaps_u64x8 = _mm512_sub_epi64(odd_products_u64x8,
                                                    _mm512_mul_epu32(odd_dots_u32x16, odd_dots_u32x16));
    __m512 const products_f32x16 = _mm512_permutex2var_ps(
        _mm512_castps256_ps512(_mm512_cvtepu64_ps(even_products_u64x8)), interleave_i32x16,
        _mm512_castps256_ps512(_mm512_cvtepu64_ps(odd_products_u64x8)));
    __m512 const gaps_f32x16 = _mm512_permutex2var_ps(_mm512_castps256_ps512(_mm512_cvtepu64_ps(even_gaps_u64x8)),
                                                      interleave_i32x16,
                                                      _mm512_castps256_ps512(_mm512_cvtepu64_ps(odd_gaps_u64x8)));
    __m512 const dots_f32x16 = _mm512_cvtepu32_ps(dots_u32x16), norms_f32x16 = _mm512_sqrt_ps(products_f32x16);
    // (ab − d²) / (ab + d · s) never cancels for a non-negative d, and the rules cover d = 0
    __m512 const angular_f32x16 = _mm512_div_ps(gaps_f32x16,
                                                _mm512_fmadd_ps(dots_f32x16, norms_f32x16, products_f32x16));
    __mmask16 const query_zero_m16 = _mm512_testn_epi32_mask(query_norm_sq_u32x16, query_norm_sq_u32x16);
    __mmask16 const target_zero_m16 = _mm512_testn_epi32_mask(target_norms_sq_u32x16, target_norms_sq_u32x16);
    __mmask16 const unit_m16 = query_zero_m16 | target_zero_m16 | _mm512_testn_epi32_mask(dots_u32x16, dots_u32x16);
    return _mm512_maskz_mov_ps((__mmask16) ~(query_zero_m16 & target_zero_m16),
                               _mm512_mask_mov_ps(angular_f32x16, unit_m16, _mm512_set1_ps(1.0f)));
}

/** Euclidean from 16 i32 dots: a + b − 2d is exact in i64 lanes, which AVX-512DQ rounds once. */
NUMKONG_INLINE __m512 nk_euclidean_i32x16_from_dot_sapphireamx_(__m512i dots_i32x16, __m512i query_norm_sq_u64x8,
                                                                __m512i target_norms_sq_u32x16) {
    // Shifts split the even and odd 32-bit lanes into 64-bit ones, doubling the dots on the way
    __m512i const interleave_i32x16 = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    __m512i const even_doubled_dots_i64x8 = _mm512_srai_epi64(_mm512_slli_epi64(dots_i32x16, 32), 31);
    __m512i const odd_doubled_dots_i64x8 = _mm512_srai_epi64(_mm512_maskz_mov_epi32(0xAAAA, dots_i32x16), 31);
    __m512i const even_sums_u64x8 = _mm512_add_epi64(query_norm_sq_u64x8,
                                                     _mm512_maskz_mov_epi32(0x5555, target_norms_sq_u32x16));
    __m512i const odd_sums_u64x8 = _mm512_add_epi64(query_norm_sq_u64x8, _mm512_srli_epi64(target_norms_sq_u32x16, 32));
    __m512i const even_dist_sq_i64x8 = _mm512_sub_epi64(even_sums_u64x8, even_doubled_dots_i64x8);
    __m512i const odd_dist_sq_i64x8 = _mm512_sub_epi64(odd_sums_u64x8, odd_doubled_dots_i64x8);
    __m512 const dist_sq_f32x16 = _mm512_permutex2var_ps(_mm512_castps256_ps512(_mm512_cvtepi64_ps(even_dist_sq_i64x8)),
                                                         interleave_i32x16,
                                                         _mm512_castps256_ps512(_mm512_cvtepi64_ps(odd_dist_sq_i64x8)));
    return _mm512_sqrt_ps(_mm512_max_ps(_mm512_setzero_ps(), dist_sq_f32x16));
}

/** Euclidean from 16 u32 dots: a + b − 2d is exact in i64 lanes, which AVX-512DQ rounds once. */
NUMKONG_INLINE __m512 nk_euclidean_u32x16_from_dot_sapphireamx_(__m512i dots_u32x16, __m512i query_norm_sq_u64x8,
                                                                __m512i target_norms_sq_u32x16) {
    // Shifts split the even and odd 32-bit lanes into 64-bit ones, doubling the dots on the way
    __m512i const interleave_i32x16 = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    __m512i const even_doubled_dots_u64x8 = _mm512_srli_epi64(_mm512_slli_epi64(dots_u32x16, 32), 31);
    __m512i const odd_doubled_dots_u64x8 = _mm512_srli_epi64(_mm512_maskz_mov_epi32(0xAAAA, dots_u32x16), 31);
    __m512i const even_sums_u64x8 = _mm512_add_epi64(query_norm_sq_u64x8,
                                                     _mm512_maskz_mov_epi32(0x5555, target_norms_sq_u32x16));
    __m512i const odd_sums_u64x8 = _mm512_add_epi64(query_norm_sq_u64x8, _mm512_srli_epi64(target_norms_sq_u32x16, 32));
    __m512i const even_dist_sq_i64x8 = _mm512_sub_epi64(even_sums_u64x8, even_doubled_dots_u64x8);
    __m512i const odd_dist_sq_i64x8 = _mm512_sub_epi64(odd_sums_u64x8, odd_doubled_dots_u64x8);
    __m512 const dist_sq_f32x16 = _mm512_permutex2var_ps(_mm512_castps256_ps512(_mm512_cvtepi64_ps(even_dist_sq_i64x8)),
                                                         interleave_i32x16,
                                                         _mm512_castps256_ps512(_mm512_cvtepi64_ps(odd_dist_sq_i64x8)));
    return _mm512_sqrt_ps(_mm512_max_ps(_mm512_setzero_ps(), dist_sq_f32x16));
}

NUMKONG_INLINE void nk_angulars_row_f32dots_sapphireamx_(nk_f32_t *results, nk_f32_t const *norms,
                                                         nk_f32_t query_norm_sq, nk_size_t count) {
    __m512 query_norm_sq_f32x16 = _mm512_set1_ps(query_norm_sq);
    // Separate reciprocal square roots avoid overflowing the product of two finite-but-large norms.
    __m512 query_rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(query_norm_sq_f32x16);
    nk_size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 dots_f32x16 = _mm512_loadu_ps(results + i);
        __m512 norms_f32x16 = _mm512_loadu_ps(norms + i);
        _mm512_storeu_ps(results + i, nk_angular_f32x16_from_dot_sapphireamx_(dots_f32x16, query_norm_sq_f32x16,
                                                                              query_rsqrt_f32x16, norms_f32x16));
    }
    if (i < count) {
        __mmask16 tail_m16 = (__mmask16)((1u << (count - i)) - 1);
        __m512 dots_f32x16 = _mm512_maskz_loadu_ps(tail_m16, results + i);
        __m512 norms_f32x16 = _mm512_maskz_loadu_ps(tail_m16, norms + i);
        _mm512_mask_storeu_ps(results + i, tail_m16,
                              nk_angular_f32x16_from_dot_sapphireamx_(dots_f32x16, query_norm_sq_f32x16,
                                                                      query_rsqrt_f32x16, norms_f32x16));
    }
}

NUMKONG_INLINE void nk_euclideans_row_f32dots_sapphireamx_(nk_f32_t *results, nk_f32_t const *norms,
                                                           nk_f32_t query_norm_sq, nk_size_t count) {
    __m512 query_norm_sq_f32x16 = _mm512_set1_ps(query_norm_sq);
    nk_size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 dots_f32x16 = _mm512_loadu_ps(results + i);
        __m512 norms_f32x16 = _mm512_loadu_ps(norms + i);
        _mm512_storeu_ps(results + i,
                         nk_euclidean_f32x16_from_dot_sapphireamx_(dots_f32x16, query_norm_sq_f32x16, norms_f32x16));
    }
    if (i < count) {
        __mmask16 tail_m16 = (__mmask16)((1u << (count - i)) - 1);
        __m512 dots_f32x16 = _mm512_maskz_loadu_ps(tail_m16, results + i);
        __m512 norms_f32x16 = _mm512_maskz_loadu_ps(tail_m16, norms + i);
        _mm512_mask_storeu_ps(
            results + i, tail_m16,
            nk_euclidean_f32x16_from_dot_sapphireamx_(dots_f32x16, query_norm_sq_f32x16, norms_f32x16));
    }
}

NUMKONG_INLINE void nk_angulars_row_i32dots_sapphireamx_(nk_f32_t *results, nk_u32_t const *norms,
                                                         nk_u32_t query_norm_sq, nk_size_t count) {
    nk_i32_t *results_i32 = (nk_i32_t *)results;
    __m512i query_norm_sq_u32x16 = _mm512_set1_epi32((nk_i32_t)query_norm_sq);
    nk_size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512i dots_i32x16 = _mm512_loadu_si512((__m512i const *)(results_i32 + i));
        __m512i norms_u32x16 = _mm512_loadu_si512((__m512i const *)(norms + i));
        _mm512_storeu_ps(results + i,
                         nk_angular_i32x16_from_dot_sapphireamx_(dots_i32x16, query_norm_sq_u32x16, norms_u32x16));
    }
    if (i < count) {
        __mmask16 tail_m16 = (__mmask16)((1u << (count - i)) - 1);
        __m512i dots_i32x16 = _mm512_maskz_loadu_epi32(tail_m16, results_i32 + i);
        __m512i norms_u32x16 = _mm512_maskz_loadu_epi32(tail_m16, norms + i);
        _mm512_mask_storeu_ps(results + i, tail_m16,
                              nk_angular_i32x16_from_dot_sapphireamx_(dots_i32x16, query_norm_sq_u32x16, norms_u32x16));
    }
}

NUMKONG_INLINE void nk_euclideans_row_i32dots_sapphireamx_(nk_f32_t *results, nk_u32_t const *norms,
                                                           nk_u32_t query_norm_sq, nk_size_t count) {
    nk_i32_t *results_i32 = (nk_i32_t *)results;
    __m512i query_norm_sq_u64x8 = _mm512_set1_epi64((nk_i64_t)query_norm_sq);
    nk_size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512i dots_i32x16 = _mm512_loadu_si512((__m512i const *)(results_i32 + i));
        __m512i norms_u32x16 = _mm512_loadu_si512((__m512i const *)(norms + i));
        _mm512_storeu_ps(results + i,
                         nk_euclidean_i32x16_from_dot_sapphireamx_(dots_i32x16, query_norm_sq_u64x8, norms_u32x16));
    }
    if (i < count) {
        __mmask16 tail_m16 = (__mmask16)((1u << (count - i)) - 1);
        __m512i dots_i32x16 = _mm512_maskz_loadu_epi32(tail_m16, results_i32 + i);
        __m512i norms_u32x16 = _mm512_maskz_loadu_epi32(tail_m16, norms + i);
        _mm512_mask_storeu_ps(
            results + i, tail_m16,
            nk_euclidean_i32x16_from_dot_sapphireamx_(dots_i32x16, query_norm_sq_u64x8, norms_u32x16));
    }
}

NUMKONG_INLINE void nk_angulars_row_u32dots_sapphireamx_(nk_f32_t *results, nk_u32_t const *norms,
                                                         nk_u32_t query_norm_sq, nk_size_t count) {
    nk_u32_t *results_u32 = (nk_u32_t *)results;
    __m512i query_norm_sq_u32x16 = _mm512_set1_epi32((nk_i32_t)query_norm_sq);
    nk_size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512i dots_u32x16 = _mm512_loadu_si512((__m512i const *)(results_u32 + i));
        __m512i norms_u32x16 = _mm512_loadu_si512((__m512i const *)(norms + i));
        _mm512_storeu_ps(results + i,
                         nk_angular_u32x16_from_dot_sapphireamx_(dots_u32x16, query_norm_sq_u32x16, norms_u32x16));
    }
    if (i < count) {
        __mmask16 tail_m16 = (__mmask16)((1u << (count - i)) - 1);
        __m512i dots_u32x16 = _mm512_maskz_loadu_epi32(tail_m16, results_u32 + i);
        __m512i norms_u32x16 = _mm512_maskz_loadu_epi32(tail_m16, norms + i);
        _mm512_mask_storeu_ps(results + i, tail_m16,
                              nk_angular_u32x16_from_dot_sapphireamx_(dots_u32x16, query_norm_sq_u32x16, norms_u32x16));
    }
}

NUMKONG_INLINE void nk_euclideans_row_u32dots_sapphireamx_(nk_f32_t *results, nk_u32_t const *norms,
                                                           nk_u32_t query_norm_sq, nk_size_t count) {
    nk_u32_t *results_u32 = (nk_u32_t *)results;
    __m512i query_norm_sq_u64x8 = _mm512_set1_epi64((nk_i64_t)query_norm_sq);
    nk_size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512i dots_u32x16 = _mm512_loadu_si512((__m512i const *)(results_u32 + i));
        __m512i norms_u32x16 = _mm512_loadu_si512((__m512i const *)(norms + i));
        _mm512_storeu_ps(results + i,
                         nk_euclidean_u32x16_from_dot_sapphireamx_(dots_u32x16, query_norm_sq_u64x8, norms_u32x16));
    }
    if (i < count) {
        __mmask16 tail_m16 = (__mmask16)((1u << (count - i)) - 1);
        __m512i dots_u32x16 = _mm512_maskz_loadu_epi32(tail_m16, results_u32 + i);
        __m512i norms_u32x16 = _mm512_maskz_loadu_epi32(tail_m16, norms + i);
        _mm512_mask_storeu_ps(
            results + i, tail_m16,
            nk_euclidean_u32x16_from_dot_sapphireamx_(dots_u32x16, query_norm_sq_u64x8, norms_u32x16));
    }
}

#pragma endregion Row Finalize Helpers

#pragma region Through BF16

/** Turns the dots in @p c into distances by @p row_fn, from the query norms of @p a and the packed
 *  column norms. */
NUMKONG_INLINE void nk_through_bf16_packed_finalize_sapphireamx_(
    nk_dots_bf16_rows_sapphireamx_t a, nk_f32_t tensor_scale, void const *b_packed, nk_f32_t *c, nk_size_t rows,
    nk_size_t columns, nk_size_t depth, nk_size_t c_stride_elements,
    void (*row_fn)(nk_f32_t *, nk_f32_t const *, nk_f32_t, nk_size_t)) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++)
        row_fn(c + row * c_stride_elements, b_norms, a.sumsq(&a, row, depth) * tensor_scale * tensor_scale, columns);
}

/** Turns the Gram dots in @p result into distances by @p row_fn above the diagonal, zeroing it. */
NUMKONG_INLINE void nk_through_bf16_symmetric_finalize_sapphireamx_(
    nk_dots_bf16_rows_sapphireamx_t vectors, nk_f32_t tensor_scale, nk_size_t vectors_count, nk_size_t depth,
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count,
    void (*row_fn)(nk_f32_t *, nk_f32_t const *, nk_f32_t, nk_size_t)) {
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = vectors.sumsq(&vectors, row, depth) * tensor_scale * tensor_scale;
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t const chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t column = chunk_start; column < chunk_end; column++)
            column_norms_cache[column - chunk_start] = vectors.sumsq(&vectors, column, depth) * tensor_scale *
                                                       tensor_scale;
        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *result_row = result + row * result_stride_elements;
            nk_size_t const column_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (column_start < chunk_end)
                row_fn(result_row + column_start, column_norms_cache + column_start - chunk_start, result_row[row],
                       chunk_end - column_start);
        }
    }
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#pragma endregion Through BF16

#pragma region BF16 Packed

NUMKONG_INLINE void nk_angulars_packed_bf16_sapphireamx_finalize_(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                  nk_size_t a_stride_elements,
                                                                  nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_bf16_skylake_(a + row * a_stride_elements, depth,
                                                                    sizeof(nk_bf16_t));
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NUMKONG_INLINE void nk_euclideans_packed_bf16_sapphireamx_finalize_(nk_bf16_t const *a, void const *b_packed,
                                                                    nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                    nk_size_t depth, nk_size_t a_stride_elements,
                                                                    nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_bf16_skylake_(a + row * a_stride_elements, depth,
                                                                    sizeof(nk_bf16_t));
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#pragma endregion BF16 Packed

#pragma region BF16 Symmetric

NUMKONG_INLINE void nk_angulars_symmetric_bf16_sapphireamx_finalize_(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_f32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_bf16_skylake_(vectors + row * stride_elements,
                                                                                        depth, sizeof(nk_bf16_t));

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_skylake_(vectors + col * stride_elements,
                                                                                       depth, sizeof(nk_bf16_t));

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NUMKONG_INLINE void nk_euclideans_symmetric_bf16_sapphireamx_finalize_(nk_bf16_t const *vectors,
                                                                       nk_size_t vectors_count, nk_size_t depth,
                                                                       nk_size_t stride_elements, nk_f32_t *result,
                                                                       nk_size_t result_stride_elements,
                                                                       nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_bf16_skylake_(vectors + row * stride_elements,
                                                                                        depth, sizeof(nk_bf16_t));

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_skylake_(vectors + col * stride_elements,
                                                                                       depth, sizeof(nk_bf16_t));

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#pragma endregion BF16 Symmetric

#pragma region I8 Packed

NUMKONG_INLINE void nk_angulars_packed_i8_sapphireamx_finalize_(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                                nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                nk_size_t a_stride_elements,
                                                                nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_u32_t query_norm_sq = nk_dots_reduce_sumsq_i8_skylake_(a + row * a_stride_elements, depth, sizeof(nk_i8_t));
        nk_angulars_row_i32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NUMKONG_INLINE void nk_euclideans_packed_i8_sapphireamx_finalize_(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                  nk_size_t a_stride_elements,
                                                                  nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_u32_t query_norm_sq = nk_dots_reduce_sumsq_i8_skylake_(a + row * a_stride_elements, depth, sizeof(nk_i8_t));
        nk_euclideans_row_i32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#pragma endregion I8 Packed

#pragma region I8 Symmetric

NUMKONG_INLINE void nk_angulars_symmetric_i8_sapphireamx_finalize_(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                                   nk_size_t depth, nk_size_t stride_elements,
                                                                   nk_f32_t *result, nk_size_t result_stride_elements,
                                                                   nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal (stored as u32 reinterpreted in f32 slot)
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        ((nk_u32_t *)(result + row * result_stride_elements))[row] = nk_dots_reduce_sumsq_i8_skylake_(
            vectors + row * stride_elements, depth, sizeof(nk_i8_t));

    // 256-column chunks with cached norms
    nk_u32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_skylake_(vectors + col * stride_elements,
                                                                                     depth, sizeof(nk_i8_t));

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_u32_t query_norm_sq = ((nk_u32_t *)r_row)[row];
            nk_angulars_row_i32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 query_norm_sq, chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NUMKONG_INLINE void nk_euclideans_symmetric_i8_sapphireamx_finalize_(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_f32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal (stored as u32 reinterpreted in f32 slot)
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        ((nk_u32_t *)(result + row * result_stride_elements))[row] = nk_dots_reduce_sumsq_i8_skylake_(
            vectors + row * stride_elements, depth, sizeof(nk_i8_t));

    // 256-column chunks with cached norms
    nk_u32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_skylake_(vectors + col * stride_elements,
                                                                                     depth, sizeof(nk_i8_t));

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_u32_t query_norm_sq = ((nk_u32_t *)r_row)[row];
            nk_euclideans_row_i32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   query_norm_sq, chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#pragma endregion I8 Symmetric

#pragma region U8 Packed

NUMKONG_INLINE void nk_angulars_packed_u8_sapphireamx_finalize_(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                                nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                nk_size_t a_stride_elements,
                                                                nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_u32_t query_norm_sq = nk_dots_reduce_sumsq_u8_skylake_(a + row * a_stride_elements, depth, sizeof(nk_u8_t));
        nk_angulars_row_u32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NUMKONG_INLINE void nk_euclideans_packed_u8_sapphireamx_finalize_(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                  nk_size_t a_stride_elements,
                                                                  nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_u32_t query_norm_sq = nk_dots_reduce_sumsq_u8_skylake_(a + row * a_stride_elements, depth, sizeof(nk_u8_t));
        nk_euclideans_row_u32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#pragma endregion U8 Packed

#pragma region U8 Symmetric

NUMKONG_INLINE void nk_angulars_symmetric_u8_sapphireamx_finalize_(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                                   nk_size_t depth, nk_size_t stride_elements,
                                                                   nk_f32_t *result, nk_size_t result_stride_elements,
                                                                   nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal (stored as u32 reinterpreted in f32 slot)
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        ((nk_u32_t *)(result + row * result_stride_elements))[row] = nk_dots_reduce_sumsq_u8_skylake_(
            vectors + row * stride_elements, depth, sizeof(nk_u8_t));

    // 256-column chunks with cached norms
    nk_u32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_skylake_(vectors + col * stride_elements,
                                                                                     depth, sizeof(nk_u8_t));

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_u32_t query_norm_sq = ((nk_u32_t *)r_row)[row];
            nk_angulars_row_u32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 query_norm_sq, chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NUMKONG_INLINE void nk_euclideans_symmetric_u8_sapphireamx_finalize_(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_f32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal (stored as u32 reinterpreted in f32 slot)
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        ((nk_u32_t *)(result + row * result_stride_elements))[row] = nk_dots_reduce_sumsq_u8_skylake_(
            vectors + row * stride_elements, depth, sizeof(nk_u8_t));

    // 256-column chunks with cached norms
    nk_u32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_skylake_(vectors + col * stride_elements,
                                                                                     depth, sizeof(nk_u8_t));

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_u32_t query_norm_sq = ((nk_u32_t *)r_row)[row];
            nk_euclideans_row_u32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   query_norm_sq, chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#pragma endregion U8 Symmetric

#pragma region E2M3 Packed

NUMKONG_INLINE void nk_angulars_packed_e2m3_sapphireamx_finalize_(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                  nk_size_t a_stride_elements,
                                                                  nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e2m3_skylake_(a + row * a_stride_elements, depth,
                                                                    sizeof(nk_e2m3_t));
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NUMKONG_INLINE void nk_euclideans_packed_e2m3_sapphireamx_finalize_(nk_e2m3_t const *a, void const *b_packed,
                                                                    nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                    nk_size_t depth, nk_size_t a_stride_elements,
                                                                    nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e2m3_skylake_(a + row * a_stride_elements, depth,
                                                                    sizeof(nk_e2m3_t));
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#pragma endregion E2M3 Packed

#pragma region E2M3 Symmetric

NUMKONG_INLINE void nk_angulars_symmetric_e2m3_sapphireamx_finalize_(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_f32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e2m3_skylake_(vectors + row * stride_elements,
                                                                                        depth, sizeof(nk_e2m3_t));

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_skylake_(vectors + col * stride_elements,
                                                                                       depth, sizeof(nk_e2m3_t));

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NUMKONG_INLINE void nk_euclideans_symmetric_e2m3_sapphireamx_finalize_(nk_e2m3_t const *vectors,
                                                                       nk_size_t vectors_count, nk_size_t depth,
                                                                       nk_size_t stride_elements, nk_f32_t *result,
                                                                       nk_size_t result_stride_elements,
                                                                       nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e2m3_skylake_(vectors + row * stride_elements,
                                                                                        depth, sizeof(nk_e2m3_t));

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_skylake_(vectors + col * stride_elements,
                                                                                       depth, sizeof(nk_e2m3_t));

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#pragma endregion E2M3 Symmetric

#pragma region E2M1 Packed

NUMKONG_INLINE void nk_angulars_packed_e2m1_sapphireamx_finalize_(nk_e2m1x2_t const *a, void const *b_packed,
                                                                  nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                  nk_size_t depth, nk_size_t a_stride_elements,
                                                                  nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e2m1_(a + row * a_stride_elements, depth, sizeof(nk_e2m1x2_t));
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NUMKONG_INLINE void nk_euclideans_packed_e2m1_sapphireamx_finalize_(nk_e2m1x2_t const *a, void const *b_packed,
                                                                    nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                    nk_size_t depth, nk_size_t a_stride_elements,
                                                                    nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e2m1_(a + row * a_stride_elements, depth, sizeof(nk_e2m1x2_t));
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#pragma endregion E2M1 Packed

#pragma region E2M1 Symmetric

NUMKONG_INLINE void nk_angulars_symmetric_e2m1_sapphireamx_finalize_(nk_e2m1x2_t const *vectors,
                                                                     nk_size_t vectors_count, nk_size_t depth,
                                                                     nk_size_t stride_elements, nk_f32_t *result,
                                                                     nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e2m1_(vectors + row * stride_elements, depth,
                                                                                sizeof(nk_e2m1x2_t));

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m1_(vectors + col * stride_elements, depth,
                                                                               sizeof(nk_e2m1x2_t));

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NUMKONG_INLINE void nk_euclideans_symmetric_e2m1_sapphireamx_finalize_(nk_e2m1x2_t const *vectors,
                                                                       nk_size_t vectors_count, nk_size_t depth,
                                                                       nk_size_t stride_elements, nk_f32_t *result,
                                                                       nk_size_t result_stride_elements,
                                                                       nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e2m1_(vectors + row * stride_elements, depth,
                                                                                sizeof(nk_e2m1x2_t));

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m1_(vectors + col * stride_elements, depth,
                                                                               sizeof(nk_e2m1x2_t));

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#pragma endregion E2M1 Symmetric

#pragma region E3M2 Packed

NUMKONG_INLINE void nk_angulars_packed_e3m2_sapphireamx_finalize_(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                  nk_size_t a_stride_elements,
                                                                  nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e3m2_skylake_(a + row * a_stride_elements, depth,
                                                                    sizeof(nk_e3m2_t));
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NUMKONG_INLINE void nk_euclideans_packed_e3m2_sapphireamx_finalize_(nk_e3m2_t const *a, void const *b_packed,
                                                                    nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                    nk_size_t depth, nk_size_t a_stride_elements,
                                                                    nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e3m2_skylake_(a + row * a_stride_elements, depth,
                                                                    sizeof(nk_e3m2_t));
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#pragma endregion E3M2 Packed

#pragma region E3M2 Symmetric

NUMKONG_INLINE void nk_angulars_symmetric_e3m2_sapphireamx_finalize_(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_f32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e3m2_skylake_(vectors + row * stride_elements,
                                                                                        depth, sizeof(nk_e3m2_t));

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_skylake_(vectors + col * stride_elements,
                                                                                       depth, sizeof(nk_e3m2_t));

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NUMKONG_INLINE void nk_euclideans_symmetric_e3m2_sapphireamx_finalize_(nk_e3m2_t const *vectors,
                                                                       nk_size_t vectors_count, nk_size_t depth,
                                                                       nk_size_t stride_elements, nk_f32_t *result,
                                                                       nk_size_t result_stride_elements,
                                                                       nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e3m2_skylake_(vectors + row * stride_elements,
                                                                                        depth, sizeof(nk_e3m2_t));

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_skylake_(vectors + col * stride_elements,
                                                                                       depth, sizeof(nk_e3m2_t));

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#pragma endregion E3M2 Symmetric

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_packed_bf16_sapphireamx( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,   //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,      //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_bf16_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride,
                                                                c_stride);
    if (status != nk_success_k) return status;
    nk_angulars_packed_bf16_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_bf16_sapphireamx( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,     //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,        //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_bf16_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride,
                                                                c_stride);
    if (status != nk_success_k) return status;
    nk_euclideans_packed_bf16_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_sapphireamx( //
    nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_bf16_sapphireamx_(vectors, vectors_count, depth, stride, (nk_f32_t *)result,
                                                         result_stride, row_start, row_count);
    if (status != nk_success_k) return status;
    nk_angulars_symmetric_bf16_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_sapphireamx( //
    nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_bf16_sapphireamx_(vectors, vectors_count, depth, stride, (nk_f32_t *)result,
                                                         result_stride, row_start, row_count);
    if (status != nk_success_k) return status;
    nk_euclideans_symmetric_bf16_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_i8_sapphireamx( //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c,   //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_i8_sapphireamx_(a, b_packed, (nk_i32_t *)c, rows, columns, depth,
                                                              a_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_angulars_packed_i8_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_i8_sapphireamx( //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c,     //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,      //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_i8_sapphireamx_(a, b_packed, (nk_i32_t *)c, rows, columns, depth,
                                                              a_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_euclideans_packed_i8_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_i8_sapphireamx( //
    nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_i8_sapphireamx_(vectors, vectors_count, depth, stride, (nk_i32_t *)result,
                                                       result_stride, row_start, row_count);
    if (status != nk_success_k) return status;
    nk_angulars_symmetric_i8_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                   result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_sapphireamx( //
    nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_i8_sapphireamx_(vectors, vectors_count, depth, stride, (nk_i32_t *)result,
                                                       result_stride, row_start, row_count);
    if (status != nk_success_k) return status;
    nk_euclideans_symmetric_i8_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_u8_sapphireamx( //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c,   //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_u8_sapphireamx_(a, b_packed, (nk_u32_t *)c, rows, columns, depth,
                                                              a_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_angulars_packed_u8_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_u8_sapphireamx( //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c,     //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,      //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_u8_sapphireamx_(a, b_packed, (nk_u32_t *)c, rows, columns, depth,
                                                              a_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_euclideans_packed_u8_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_u8_sapphireamx( //
    nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_u8_sapphireamx_(vectors, vectors_count, depth, stride, (nk_u32_t *)result,
                                                       result_stride, row_start, row_count);
    if (status != nk_success_k) return status;
    nk_angulars_symmetric_u8_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                   result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_sapphireamx( //
    nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_u8_sapphireamx_(vectors, vectors_count, depth, stride, (nk_u32_t *)result,
                                                       result_stride, row_start, row_count);
    if (status != nk_success_k) return status;
    nk_euclideans_symmetric_u8_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_e4m3_sapphireamx( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,   //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,      //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_dots_bf16_rows_sapphireamx_t const source = {
        (nk_u8_t const *)a, a_stride, NUMKONG_NULL, 0, nk_e4m3_widen_bf16_sapphireamx_, nk_e4m3_sumsq_sapphireamx_};
    nk_status_t const status = nk_gemm_packed_through_bf16_sapphireamx_(source, 1, b_packed, c, rows, columns, depth,
                                                                        c_stride);
    if (status != nk_success_k) return status;
    nk_through_bf16_packed_finalize_sapphireamx_(source, 1, b_packed, c, rows, columns, depth, c_stride_elements,
                                                 nk_angulars_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_sapphireamx( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,     //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,        //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_dots_bf16_rows_sapphireamx_t const source = {
        (nk_u8_t const *)a, a_stride, NUMKONG_NULL, 0, nk_e4m3_widen_bf16_sapphireamx_, nk_e4m3_sumsq_sapphireamx_};
    nk_status_t const status = nk_gemm_packed_through_bf16_sapphireamx_(source, 1, b_packed, c, rows, columns, depth,
                                                                        c_stride);
    if (status != nk_success_k) return status;
    nk_through_bf16_packed_finalize_sapphireamx_(source, 1, b_packed, c, rows, columns, depth, c_stride_elements,
                                                 nk_euclideans_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_e5m2_sapphireamx( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,   //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,      //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_dots_bf16_rows_sapphireamx_t const source = {
        (nk_u8_t const *)a, a_stride, NUMKONG_NULL, 0, nk_e5m2_widen_bf16_sapphireamx_, nk_e5m2_sumsq_sapphireamx_};
    nk_status_t const status = nk_gemm_packed_through_bf16_sapphireamx_(source, 1, b_packed, c, rows, columns, depth,
                                                                        c_stride);
    if (status != nk_success_k) return status;
    nk_through_bf16_packed_finalize_sapphireamx_(source, 1, b_packed, c, rows, columns, depth, c_stride_elements,
                                                 nk_angulars_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_sapphireamx( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,     //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,        //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_dots_bf16_rows_sapphireamx_t const source = {
        (nk_u8_t const *)a, a_stride, NUMKONG_NULL, 0, nk_e5m2_widen_bf16_sapphireamx_, nk_e5m2_sumsq_sapphireamx_};
    nk_status_t const status = nk_gemm_packed_through_bf16_sapphireamx_(source, 1, b_packed, c, rows, columns, depth,
                                                                        c_stride);
    if (status != nk_success_k) return status;
    nk_through_bf16_packed_finalize_sapphireamx_(source, 1, b_packed, c, rows, columns, depth, c_stride_elements,
                                                 nk_euclideans_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_sapphireamx( //
    nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_bf16_rows_sapphireamx_t const source = {
        (nk_u8_t const *)vectors, stride, NUMKONG_NULL, 0, nk_e5m2_widen_bf16_sapphireamx_, nk_e5m2_sumsq_sapphireamx_};
    nk_status_t const status = nk_gram_through_bf16_sapphireamx_(source, 1, vectors_count, depth, result, result_stride,
                                                                 row_start, row_count);
    if (status != nk_success_k) return status;
    nk_through_bf16_symmetric_finalize_sapphireamx_(source, 1, vectors_count, depth, result, result_stride_elements,
                                                    row_start, row_count, nk_angulars_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_sapphireamx( //
    nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_bf16_rows_sapphireamx_t const source = {
        (nk_u8_t const *)vectors, stride, NUMKONG_NULL, 0, nk_e5m2_widen_bf16_sapphireamx_, nk_e5m2_sumsq_sapphireamx_};
    nk_status_t const status = nk_gram_through_bf16_sapphireamx_(source, 1, vectors_count, depth, result, result_stride,
                                                                 row_start, row_count);
    if (status != nk_success_k) return status;
    nk_through_bf16_symmetric_finalize_sapphireamx_(source, 1, vectors_count, depth, result, result_stride_elements,
                                                    row_start, row_count, nk_euclideans_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_sapphireamx( //
    nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_bf16_rows_sapphireamx_t const source = {
        (nk_u8_t const *)vectors, stride, NUMKONG_NULL, 0, nk_e4m3_widen_bf16_sapphireamx_, nk_e4m3_sumsq_sapphireamx_};
    nk_status_t const status = nk_gram_through_bf16_sapphireamx_(source, 1, vectors_count, depth, result, result_stride,
                                                                 row_start, row_count);
    if (status != nk_success_k) return status;
    nk_through_bf16_symmetric_finalize_sapphireamx_(source, 1, vectors_count, depth, result, result_stride_elements,
                                                    row_start, row_count, nk_angulars_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_sapphireamx( //
    nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_bf16_rows_sapphireamx_t const source = {
        (nk_u8_t const *)vectors, stride, NUMKONG_NULL, 0, nk_e4m3_widen_bf16_sapphireamx_, nk_e4m3_sumsq_sapphireamx_};
    nk_status_t const status = nk_gram_through_bf16_sapphireamx_(source, 1, vectors_count, depth, result, result_stride,
                                                                 row_start, row_count);
    if (status != nk_success_k) return status;
    nk_through_bf16_symmetric_finalize_sapphireamx_(source, 1, vectors_count, depth, result, result_stride_elements,
                                                    row_start, row_count, nk_euclideans_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_sapphireamx(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                             void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_operand_t const a_unpacked = nk_cross_operand_(nk_nvfp4_k, a, a_stride);
    nk_dots_bf16_rows_sapphireamx_t const source = {(nk_u8_t const *)a_unpacked.elements,
                                                    a_stride,
                                                    a_unpacked.scales,
                                                    a_unpacked.scales_stride,
                                                    nk_nvfp4_widen_bf16_sapphireamx_,
                                                    nk_scaled_sumsq_sapphireamx_};
    nk_status_t const status = nk_gemm_packed_through_bf16_sapphireamx_(
        source, nk_cross_tensor_scale_(a_unpacked.tensor_scale), b_packed, c, rows, columns, depth, c_stride);
    if (status != nk_success_k) return status;
    nk_through_bf16_packed_finalize_sapphireamx_(source, nk_cross_tensor_scale_(a_unpacked.tensor_scale), b_packed, c,
                                                 rows, columns, depth, c_stride / sizeof(nk_f32_t),
                                                 nk_angulars_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_sapphireamx(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_cross_operand_t const vectors_unpacked = nk_cross_operand_(nk_nvfp4_k, vectors, stride);
    nk_dots_bf16_rows_sapphireamx_t const source = {(nk_u8_t const *)vectors_unpacked.elements,
                                                    stride,
                                                    vectors_unpacked.scales,
                                                    vectors_unpacked.scales_stride,
                                                    nk_nvfp4_widen_bf16_sapphireamx_,
                                                    nk_scaled_sumsq_sapphireamx_};
    nk_status_t const status = nk_gram_through_bf16_sapphireamx_(
        source, nk_cross_tensor_scale_(vectors_unpacked.tensor_scale), vectors_count, depth, result, result_stride,
        row_start, row_count);
    if (status != nk_success_k) return status;
    nk_through_bf16_symmetric_finalize_sapphireamx_(source, nk_cross_tensor_scale_(vectors_unpacked.tensor_scale),
                                                    vectors_count, depth, result, result_stride / sizeof(nk_f32_t),
                                                    row_start, row_count, nk_angulars_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_sapphireamx(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                               nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                               nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                               void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_operand_t const a_unpacked = nk_cross_operand_(nk_nvfp4_k, a, a_stride);
    nk_dots_bf16_rows_sapphireamx_t const source = {(nk_u8_t const *)a_unpacked.elements,
                                                    a_stride,
                                                    a_unpacked.scales,
                                                    a_unpacked.scales_stride,
                                                    nk_nvfp4_widen_bf16_sapphireamx_,
                                                    nk_scaled_sumsq_sapphireamx_};
    nk_status_t const status = nk_gemm_packed_through_bf16_sapphireamx_(
        source, nk_cross_tensor_scale_(a_unpacked.tensor_scale), b_packed, c, rows, columns, depth, c_stride);
    if (status != nk_success_k) return status;
    nk_through_bf16_packed_finalize_sapphireamx_(source, nk_cross_tensor_scale_(a_unpacked.tensor_scale), b_packed, c,
                                                 rows, columns, depth, c_stride / sizeof(nk_f32_t),
                                                 nk_euclideans_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_sapphireamx(nk_nvfp4_cref_t const *vectors,
                                                                  nk_size_t vectors_count, nk_size_t depth,
                                                                  nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_cross_operand_t const vectors_unpacked = nk_cross_operand_(nk_nvfp4_k, vectors, stride);
    nk_dots_bf16_rows_sapphireamx_t const source = {(nk_u8_t const *)vectors_unpacked.elements,
                                                    stride,
                                                    vectors_unpacked.scales,
                                                    vectors_unpacked.scales_stride,
                                                    nk_nvfp4_widen_bf16_sapphireamx_,
                                                    nk_scaled_sumsq_sapphireamx_};
    nk_status_t const status = nk_gram_through_bf16_sapphireamx_(
        source, nk_cross_tensor_scale_(vectors_unpacked.tensor_scale), vectors_count, depth, result, result_stride,
        row_start, row_count);
    if (status != nk_success_k) return status;
    nk_through_bf16_symmetric_finalize_sapphireamx_(source, nk_cross_tensor_scale_(vectors_unpacked.tensor_scale),
                                                    vectors_count, depth, result, result_stride / sizeof(nk_f32_t),
                                                    row_start, row_count, nk_euclideans_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_sapphireamx(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                             void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_operand_t const a_unpacked = nk_cross_operand_(nk_mxfp4_k, a, a_stride);
    nk_dots_bf16_rows_sapphireamx_t const source = {(nk_u8_t const *)a_unpacked.elements,
                                                    a_stride,
                                                    a_unpacked.scales,
                                                    a_unpacked.scales_stride,
                                                    nk_mxfp4_widen_bf16_sapphireamx_,
                                                    nk_scaled_sumsq_sapphireamx_};
    nk_status_t const status = nk_gemm_packed_through_bf16_sapphireamx_(
        source, nk_cross_tensor_scale_(a_unpacked.tensor_scale), b_packed, c, rows, columns, depth, c_stride);
    if (status != nk_success_k) return status;
    nk_through_bf16_packed_finalize_sapphireamx_(source, nk_cross_tensor_scale_(a_unpacked.tensor_scale), b_packed, c,
                                                 rows, columns, depth, c_stride / sizeof(nk_f32_t),
                                                 nk_angulars_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_sapphireamx(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_cross_operand_t const vectors_unpacked = nk_cross_operand_(nk_mxfp4_k, vectors, stride);
    nk_dots_bf16_rows_sapphireamx_t const source = {(nk_u8_t const *)vectors_unpacked.elements,
                                                    stride,
                                                    vectors_unpacked.scales,
                                                    vectors_unpacked.scales_stride,
                                                    nk_mxfp4_widen_bf16_sapphireamx_,
                                                    nk_scaled_sumsq_sapphireamx_};
    nk_status_t const status = nk_gram_through_bf16_sapphireamx_(
        source, nk_cross_tensor_scale_(vectors_unpacked.tensor_scale), vectors_count, depth, result, result_stride,
        row_start, row_count);
    if (status != nk_success_k) return status;
    nk_through_bf16_symmetric_finalize_sapphireamx_(source, nk_cross_tensor_scale_(vectors_unpacked.tensor_scale),
                                                    vectors_count, depth, result, result_stride / sizeof(nk_f32_t),
                                                    row_start, row_count, nk_angulars_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_sapphireamx(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                               nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                               nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                               void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_operand_t const a_unpacked = nk_cross_operand_(nk_mxfp4_k, a, a_stride);
    nk_dots_bf16_rows_sapphireamx_t const source = {(nk_u8_t const *)a_unpacked.elements,
                                                    a_stride,
                                                    a_unpacked.scales,
                                                    a_unpacked.scales_stride,
                                                    nk_mxfp4_widen_bf16_sapphireamx_,
                                                    nk_scaled_sumsq_sapphireamx_};
    nk_status_t const status = nk_gemm_packed_through_bf16_sapphireamx_(
        source, nk_cross_tensor_scale_(a_unpacked.tensor_scale), b_packed, c, rows, columns, depth, c_stride);
    if (status != nk_success_k) return status;
    nk_through_bf16_packed_finalize_sapphireamx_(source, nk_cross_tensor_scale_(a_unpacked.tensor_scale), b_packed, c,
                                                 rows, columns, depth, c_stride / sizeof(nk_f32_t),
                                                 nk_euclideans_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_sapphireamx(nk_mxfp4_cref_t const *vectors,
                                                                  nk_size_t vectors_count, nk_size_t depth,
                                                                  nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_cross_operand_t const vectors_unpacked = nk_cross_operand_(nk_mxfp4_k, vectors, stride);
    nk_dots_bf16_rows_sapphireamx_t const source = {(nk_u8_t const *)vectors_unpacked.elements,
                                                    stride,
                                                    vectors_unpacked.scales,
                                                    vectors_unpacked.scales_stride,
                                                    nk_mxfp4_widen_bf16_sapphireamx_,
                                                    nk_scaled_sumsq_sapphireamx_};
    nk_status_t const status = nk_gram_through_bf16_sapphireamx_(
        source, nk_cross_tensor_scale_(vectors_unpacked.tensor_scale), vectors_count, depth, result, result_stride,
        row_start, row_count);
    if (status != nk_success_k) return status;
    nk_through_bf16_symmetric_finalize_sapphireamx_(source, nk_cross_tensor_scale_(vectors_unpacked.tensor_scale),
                                                    vectors_count, depth, result, result_stride / sizeof(nk_f32_t),
                                                    row_start, row_count, nk_euclideans_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e4m3_sapphireamx(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                                 nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                 nk_size_t depth, nk_size_t a_stride,
                                                                 nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_operand_t const a_unpacked = nk_cross_operand_(nk_mxfp8e4m3_k, a, a_stride);
    nk_dots_bf16_rows_sapphireamx_t const source = {(nk_u8_t const *)a_unpacked.elements,
                                                    a_stride,
                                                    a_unpacked.scales,
                                                    a_unpacked.scales_stride,
                                                    nk_mxfp8e4m3_widen_bf16_sapphireamx_,
                                                    nk_scaled_sumsq_sapphireamx_};
    nk_status_t const status = nk_gemm_packed_through_bf16_sapphireamx_(
        source, nk_cross_tensor_scale_(a_unpacked.tensor_scale), b_packed, c, rows, columns, depth, c_stride);
    if (status != nk_success_k) return status;
    nk_through_bf16_packed_finalize_sapphireamx_(source, nk_cross_tensor_scale_(a_unpacked.tensor_scale), b_packed, c,
                                                 rows, columns, depth, c_stride / sizeof(nk_f32_t),
                                                 nk_angulars_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_sapphireamx(nk_mxfp8e4m3_cref_t const *vectors,
                                                                    nk_size_t vectors_count, nk_size_t depth,
                                                                    nk_size_t stride, nk_f32_t *result,
                                                                    nk_size_t result_stride, nk_size_t row_start,
                                                                    nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_cross_operand_t const vectors_unpacked = nk_cross_operand_(nk_mxfp8e4m3_k, vectors, stride);
    nk_dots_bf16_rows_sapphireamx_t const source = {(nk_u8_t const *)vectors_unpacked.elements,
                                                    stride,
                                                    vectors_unpacked.scales,
                                                    vectors_unpacked.scales_stride,
                                                    nk_mxfp8e4m3_widen_bf16_sapphireamx_,
                                                    nk_scaled_sumsq_sapphireamx_};
    nk_status_t const status = nk_gram_through_bf16_sapphireamx_(
        source, nk_cross_tensor_scale_(vectors_unpacked.tensor_scale), vectors_count, depth, result, result_stride,
        row_start, row_count);
    if (status != nk_success_k) return status;
    nk_through_bf16_symmetric_finalize_sapphireamx_(source, nk_cross_tensor_scale_(vectors_unpacked.tensor_scale),
                                                    vectors_count, depth, result, result_stride / sizeof(nk_f32_t),
                                                    row_start, row_count, nk_angulars_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_sapphireamx(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                                   nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                   nk_size_t depth, nk_size_t a_stride,
                                                                   nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_operand_t const a_unpacked = nk_cross_operand_(nk_mxfp8e4m3_k, a, a_stride);
    nk_dots_bf16_rows_sapphireamx_t const source = {(nk_u8_t const *)a_unpacked.elements,
                                                    a_stride,
                                                    a_unpacked.scales,
                                                    a_unpacked.scales_stride,
                                                    nk_mxfp8e4m3_widen_bf16_sapphireamx_,
                                                    nk_scaled_sumsq_sapphireamx_};
    nk_status_t const status = nk_gemm_packed_through_bf16_sapphireamx_(
        source, nk_cross_tensor_scale_(a_unpacked.tensor_scale), b_packed, c, rows, columns, depth, c_stride);
    if (status != nk_success_k) return status;
    nk_through_bf16_packed_finalize_sapphireamx_(source, nk_cross_tensor_scale_(a_unpacked.tensor_scale), b_packed, c,
                                                 rows, columns, depth, c_stride / sizeof(nk_f32_t),
                                                 nk_euclideans_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_sapphireamx(nk_mxfp8e4m3_cref_t const *vectors,
                                                                      nk_size_t vectors_count, nk_size_t depth,
                                                                      nk_size_t stride, nk_f32_t *result,
                                                                      nk_size_t result_stride, nk_size_t row_start,
                                                                      nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_cross_operand_t const vectors_unpacked = nk_cross_operand_(nk_mxfp8e4m3_k, vectors, stride);
    nk_dots_bf16_rows_sapphireamx_t const source = {(nk_u8_t const *)vectors_unpacked.elements,
                                                    stride,
                                                    vectors_unpacked.scales,
                                                    vectors_unpacked.scales_stride,
                                                    nk_mxfp8e4m3_widen_bf16_sapphireamx_,
                                                    nk_scaled_sumsq_sapphireamx_};
    nk_status_t const status = nk_gram_through_bf16_sapphireamx_(
        source, nk_cross_tensor_scale_(vectors_unpacked.tensor_scale), vectors_count, depth, result, result_stride,
        row_start, row_count);
    if (status != nk_success_k) return status;
    nk_through_bf16_symmetric_finalize_sapphireamx_(source, nk_cross_tensor_scale_(vectors_unpacked.tensor_scale),
                                                    vectors_count, depth, result, result_stride / sizeof(nk_f32_t),
                                                    row_start, row_count, nk_euclideans_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e5m2_sapphireamx(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                                 nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                 nk_size_t depth, nk_size_t a_stride,
                                                                 nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_operand_t const a_unpacked = nk_cross_operand_(nk_mxfp8e5m2_k, a, a_stride);
    nk_dots_bf16_rows_sapphireamx_t const source = {(nk_u8_t const *)a_unpacked.elements,
                                                    a_stride,
                                                    a_unpacked.scales,
                                                    a_unpacked.scales_stride,
                                                    nk_mxfp8e5m2_widen_bf16_sapphireamx_,
                                                    nk_scaled_sumsq_sapphireamx_};
    nk_status_t const status = nk_gemm_packed_through_bf16_sapphireamx_(
        source, nk_cross_tensor_scale_(a_unpacked.tensor_scale), b_packed, c, rows, columns, depth, c_stride);
    if (status != nk_success_k) return status;
    nk_through_bf16_packed_finalize_sapphireamx_(source, nk_cross_tensor_scale_(a_unpacked.tensor_scale), b_packed, c,
                                                 rows, columns, depth, c_stride / sizeof(nk_f32_t),
                                                 nk_angulars_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_sapphireamx(nk_mxfp8e5m2_cref_t const *vectors,
                                                                    nk_size_t vectors_count, nk_size_t depth,
                                                                    nk_size_t stride, nk_f32_t *result,
                                                                    nk_size_t result_stride, nk_size_t row_start,
                                                                    nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_cross_operand_t const vectors_unpacked = nk_cross_operand_(nk_mxfp8e5m2_k, vectors, stride);
    nk_dots_bf16_rows_sapphireamx_t const source = {(nk_u8_t const *)vectors_unpacked.elements,
                                                    stride,
                                                    vectors_unpacked.scales,
                                                    vectors_unpacked.scales_stride,
                                                    nk_mxfp8e5m2_widen_bf16_sapphireamx_,
                                                    nk_scaled_sumsq_sapphireamx_};
    nk_status_t const status = nk_gram_through_bf16_sapphireamx_(
        source, nk_cross_tensor_scale_(vectors_unpacked.tensor_scale), vectors_count, depth, result, result_stride,
        row_start, row_count);
    if (status != nk_success_k) return status;
    nk_through_bf16_symmetric_finalize_sapphireamx_(source, nk_cross_tensor_scale_(vectors_unpacked.tensor_scale),
                                                    vectors_count, depth, result, result_stride / sizeof(nk_f32_t),
                                                    row_start, row_count, nk_angulars_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_sapphireamx(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                                   nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                   nk_size_t depth, nk_size_t a_stride,
                                                                   nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_operand_t const a_unpacked = nk_cross_operand_(nk_mxfp8e5m2_k, a, a_stride);
    nk_dots_bf16_rows_sapphireamx_t const source = {(nk_u8_t const *)a_unpacked.elements,
                                                    a_stride,
                                                    a_unpacked.scales,
                                                    a_unpacked.scales_stride,
                                                    nk_mxfp8e5m2_widen_bf16_sapphireamx_,
                                                    nk_scaled_sumsq_sapphireamx_};
    nk_status_t const status = nk_gemm_packed_through_bf16_sapphireamx_(
        source, nk_cross_tensor_scale_(a_unpacked.tensor_scale), b_packed, c, rows, columns, depth, c_stride);
    if (status != nk_success_k) return status;
    nk_through_bf16_packed_finalize_sapphireamx_(source, nk_cross_tensor_scale_(a_unpacked.tensor_scale), b_packed, c,
                                                 rows, columns, depth, c_stride / sizeof(nk_f32_t),
                                                 nk_euclideans_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_sapphireamx(nk_mxfp8e5m2_cref_t const *vectors,
                                                                      nk_size_t vectors_count, nk_size_t depth,
                                                                      nk_size_t stride, nk_f32_t *result,
                                                                      nk_size_t result_stride, nk_size_t row_start,
                                                                      nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_cross_operand_t const vectors_unpacked = nk_cross_operand_(nk_mxfp8e5m2_k, vectors, stride);
    nk_dots_bf16_rows_sapphireamx_t const source = {(nk_u8_t const *)vectors_unpacked.elements,
                                                    stride,
                                                    vectors_unpacked.scales,
                                                    vectors_unpacked.scales_stride,
                                                    nk_mxfp8e5m2_widen_bf16_sapphireamx_,
                                                    nk_scaled_sumsq_sapphireamx_};
    nk_status_t const status = nk_gram_through_bf16_sapphireamx_(
        source, nk_cross_tensor_scale_(vectors_unpacked.tensor_scale), vectors_count, depth, result, result_stride,
        row_start, row_count);
    if (status != nk_success_k) return status;
    nk_through_bf16_symmetric_finalize_sapphireamx_(source, nk_cross_tensor_scale_(vectors_unpacked.tensor_scale),
                                                    vectors_count, depth, result, result_stride / sizeof(nk_f32_t),
                                                    row_start, row_count, nk_euclideans_row_f32dots_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_e2m3_sapphireamx( //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,   //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,      //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e2m3_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride,
                                                                c_stride);
    if (status != nk_success_k) return status;
    nk_angulars_packed_e2m3_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_sapphireamx( //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,     //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,        //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e2m3_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride,
                                                                c_stride);
    if (status != nk_success_k) return status;
    nk_euclideans_packed_e2m3_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_sapphireamx( //
    nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e2m3_sapphireamx_(vectors, vectors_count, depth, stride, (nk_f32_t *)result,
                                                         result_stride, row_start, row_count);
    if (status != nk_success_k) return status;
    nk_angulars_symmetric_e2m3_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_sapphireamx( //
    nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e2m3_sapphireamx_(vectors, vectors_count, depth, stride, (nk_f32_t *)result,
                                                         result_stride, row_start, row_count);
    if (status != nk_success_k) return status;
    nk_euclideans_symmetric_e2m3_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_e2m1_sapphireamx( //
    nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,      //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m1x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e2m1_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride,
                                                                c_stride);
    if (status != nk_success_k) return status;
    nk_angulars_packed_e2m1_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_sapphireamx( //
    nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,   //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,        //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m1x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e2m1_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride,
                                                                c_stride);
    if (status != nk_success_k) return status;
    nk_euclideans_packed_e2m1_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_sapphireamx( //
    nk_e2m1x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e2m1x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e2m1_sapphireamx_(vectors, vectors_count, depth, stride, (nk_f32_t *)result,
                                                         result_stride, row_start, row_count);
    if (status != nk_success_k) return status;
    nk_angulars_symmetric_e2m1_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_sapphireamx( //
    nk_e2m1x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e2m1x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e2m1_sapphireamx_(vectors, vectors_count, depth, stride, (nk_f32_t *)result,
                                                         result_stride, row_start, row_count);
    if (status != nk_success_k) return status;
    nk_euclideans_symmetric_e2m1_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_e3m2_sapphireamx( //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,   //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,      //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e3m2_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride,
                                                                c_stride);
    if (status != nk_success_k) return status;
    nk_angulars_packed_e3m2_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_sapphireamx( //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,     //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,        //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e3m2_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride,
                                                                c_stride);
    if (status != nk_success_k) return status;
    nk_euclideans_packed_e3m2_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_sapphireamx( //
    nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e3m2_sapphireamx_(vectors, vectors_count, depth, stride, (nk_f32_t *)result,
                                                         result_stride, row_start, row_count);
    if (status != nk_success_k) return status;
    nk_angulars_symmetric_e3m2_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_sapphireamx( //
    nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e3m2_sapphireamx_(vectors, vectors_count, depth, stride, (nk_f32_t *)result,
                                                         result_stride, row_start, row_count);
    if (status != nk_success_k) return status;
    nk_euclideans_symmetric_e3m2_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_SAPPHIREAMX_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_SPATIALS_SAPPHIREAMX_H
