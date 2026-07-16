/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for Diamond Rapids.
 *  @file include/numkong/spatial/diamond.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/spatial.h
 *
 *  For L2 distance, uses the identity: (a−b)² = a² + b² − 2 × a × b,
 *  with VCVTHF82PH/VCVTBF82PH for 1-instruction FP8→FP16 conversion and
 *  VDPPHPS for FP16-pair dot products accumulating into FP32.
 */
#ifndef NK_SPATIAL_DIAMOND_H
#define NK_SPATIAL_DIAMOND_H

#if NK_TARGET_X8664_
#if NK_TARGET_DIAMOND

#include "numkong/types.h"
#include "numkong/spatial/haswell.h" // `nk_angular_normalize_f32_haswell_`, `nk_f32_sqrt_haswell`
#include "numkong/reduce/skylake.h"  // `nk_reduce_add_f32x16_skylake_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                    \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx10.2-512,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "avx10.2-512", "f16c", "fma", \
                   "bmi", "bmi2")
#endif

NK_API_COMPTIME void nk_sqeuclidean_e4m3_diamond(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                 nk_f32_t *result) {
    __m512 a_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_sq_f32x16 = _mm512_setzero_ps();
    __m512 ab_f32x16 = _mm512_setzero_ps();
    __m256i a_e4m3x32, b_e4m3x32;

nk_sqeuclidean_e4m3_diamond_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_e4m3x32 = _mm256_maskz_loadu_epi8(mask, a);
        b_e4m3x32 = _mm256_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e4m3x32 = _mm256_loadu_epi8(a);
        b_e4m3x32 = _mm256_loadu_epi8(b);
        a += 32, b += 32, n -= 32;
    }
    __m512h a_f16x32 = _mm512_cvthf8_ph(a_e4m3x32);
    __m512h b_f16x32 = _mm512_cvthf8_ph(b_e4m3x32);
    a_sq_f32x16 = _mm512_dpph_ps(a_sq_f32x16, a_f16x32, a_f16x32);
    b_sq_f32x16 = _mm512_dpph_ps(b_sq_f32x16, b_f16x32, b_f16x32);
    ab_f32x16 = _mm512_dpph_ps(ab_f32x16, a_f16x32, b_f16x32);
    if (n) goto nk_sqeuclidean_e4m3_diamond_cycle;

    __m512 sum_sq_f32x16 = _mm512_add_ps(a_sq_f32x16, b_sq_f32x16);
    *result = nk_reduce_add_f32x16_skylake_(_mm512_fnmadd_ps(_mm512_set1_ps(2.0f), ab_f32x16, sum_sq_f32x16));
}

NK_API_COMPTIME void nk_euclidean_e4m3_diamond(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e4m3_diamond(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_API_COMPTIME void nk_angular_e4m3_diamond(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 dot_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m256i a_e4m3x32, b_e4m3x32;

nk_angular_e4m3_diamond_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_e4m3x32 = _mm256_maskz_loadu_epi8(mask, a);
        b_e4m3x32 = _mm256_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e4m3x32 = _mm256_loadu_epi8(a);
        b_e4m3x32 = _mm256_loadu_epi8(b);
        a += 32, b += 32, n -= 32;
    }
    __m512h a_f16x32 = _mm512_cvthf8_ph(a_e4m3x32);
    __m512h b_f16x32 = _mm512_cvthf8_ph(b_e4m3x32);
    dot_f32x16 = _mm512_dpph_ps(dot_f32x16, a_f16x32, b_f16x32);
    a_norm_sq_f32x16 = _mm512_dpph_ps(a_norm_sq_f32x16, a_f16x32, a_f16x32);
    b_norm_sq_f32x16 = _mm512_dpph_ps(b_norm_sq_f32x16, b_f16x32, b_f16x32);
    if (n) goto nk_angular_e4m3_diamond_cycle;

    nk_f32_t dot_f32 = nk_reduce_add_f32x16_skylake_(dot_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(a_norm_sq_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(b_norm_sq_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_API_COMPTIME void nk_sqeuclidean_e5m2_diamond(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                 nk_f32_t *result) {
    __m512 a_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_sq_f32x16 = _mm512_setzero_ps();
    __m512 ab_f32x16 = _mm512_setzero_ps();
    __m256i a_e5m2x32, b_e5m2x32;

nk_sqeuclidean_e5m2_diamond_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_e5m2x32 = _mm256_maskz_loadu_epi8(mask, a);
        b_e5m2x32 = _mm256_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e5m2x32 = _mm256_loadu_epi8(a);
        b_e5m2x32 = _mm256_loadu_epi8(b);
        a += 32, b += 32, n -= 32;
    }
    __m512h a_f16x32 = _mm512_cvtbf8_ph(a_e5m2x32);
    __m512h b_f16x32 = _mm512_cvtbf8_ph(b_e5m2x32);
    a_sq_f32x16 = _mm512_dpph_ps(a_sq_f32x16, a_f16x32, a_f16x32);
    b_sq_f32x16 = _mm512_dpph_ps(b_sq_f32x16, b_f16x32, b_f16x32);
    ab_f32x16 = _mm512_dpph_ps(ab_f32x16, a_f16x32, b_f16x32);
    if (n) goto nk_sqeuclidean_e5m2_diamond_cycle;

    __m512 sum_sq_f32x16 = _mm512_add_ps(a_sq_f32x16, b_sq_f32x16);
    *result = nk_reduce_add_f32x16_skylake_(_mm512_fnmadd_ps(_mm512_set1_ps(2.0f), ab_f32x16, sum_sq_f32x16));
}

NK_API_COMPTIME void nk_euclidean_e5m2_diamond(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e5m2_diamond(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_API_COMPTIME void nk_angular_e5m2_diamond(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 dot_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m256i a_e5m2x32, b_e5m2x32;

nk_angular_e5m2_diamond_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_e5m2x32 = _mm256_maskz_loadu_epi8(mask, a);
        b_e5m2x32 = _mm256_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e5m2x32 = _mm256_loadu_epi8(a);
        b_e5m2x32 = _mm256_loadu_epi8(b);
        a += 32, b += 32, n -= 32;
    }
    __m512h a_f16x32 = _mm512_cvtbf8_ph(a_e5m2x32);
    __m512h b_f16x32 = _mm512_cvtbf8_ph(b_e5m2x32);
    dot_f32x16 = _mm512_dpph_ps(dot_f32x16, a_f16x32, b_f16x32);
    a_norm_sq_f32x16 = _mm512_dpph_ps(a_norm_sq_f32x16, a_f16x32, a_f16x32);
    b_norm_sq_f32x16 = _mm512_dpph_ps(b_norm_sq_f32x16, b_f16x32, b_f16x32);
    if (n) goto nk_angular_e5m2_diamond_cycle;

    nk_f32_t dot_f32 = nk_reduce_add_f32x16_skylake_(dot_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(a_norm_sq_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(b_norm_sq_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_API_COMPTIME void nk_sqeuclidean_f16_diamond(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 a_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_sq_f32x16 = _mm512_setzero_ps();
    __m512 ab_f32x16 = _mm512_setzero_ps();
    __m512h a_f16x32, b_f16x32;

nk_sqeuclidean_f16_diamond_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        b_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
        n = 0;
    }
    else {
        a_f16x32 = _mm512_castsi512_ph(_mm512_loadu_epi16(a));
        b_f16x32 = _mm512_castsi512_ph(_mm512_loadu_epi16(b));
        a += 32, b += 32, n -= 32;
    }
    a_sq_f32x16 = _mm512_dpph_ps(a_sq_f32x16, a_f16x32, a_f16x32);
    b_sq_f32x16 = _mm512_dpph_ps(b_sq_f32x16, b_f16x32, b_f16x32);
    ab_f32x16 = _mm512_dpph_ps(ab_f32x16, a_f16x32, b_f16x32);
    if (n) goto nk_sqeuclidean_f16_diamond_cycle;

    __m512 sum_sq_f32x16 = _mm512_add_ps(a_sq_f32x16, b_sq_f32x16);
    *result = nk_reduce_add_f32x16_skylake_(_mm512_fnmadd_ps(_mm512_set1_ps(2.0f), ab_f32x16, sum_sq_f32x16));
}

NK_API_COMPTIME void nk_euclidean_f16_diamond(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_f16_diamond(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_API_COMPTIME void nk_angular_f16_diamond(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 dot_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512h a_f16x32, b_f16x32;

nk_angular_f16_diamond_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        b_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
        n = 0;
    }
    else {
        a_f16x32 = _mm512_castsi512_ph(_mm512_loadu_epi16(a));
        b_f16x32 = _mm512_castsi512_ph(_mm512_loadu_epi16(b));
        a += 32, b += 32, n -= 32;
    }
    dot_f32x16 = _mm512_dpph_ps(dot_f32x16, a_f16x32, b_f16x32);
    a_norm_sq_f32x16 = _mm512_dpph_ps(a_norm_sq_f32x16, a_f16x32, a_f16x32);
    b_norm_sq_f32x16 = _mm512_dpph_ps(b_norm_sq_f32x16, b_f16x32, b_f16x32);
    if (n) goto nk_angular_f16_diamond_cycle;

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

#endif // NK_TARGET_DIAMOND
#endif // NK_TARGET_X8664_
#endif // NK_SPATIAL_DIAMOND_H
