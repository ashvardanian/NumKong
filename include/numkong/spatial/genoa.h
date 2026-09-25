/**
 *  @file include/numkong/spatial/genoa.h
 *  @author Ash Vardanian
 *  @date June 4, 2024
 *  @brief SIMD-accelerated spatial similarity measures for Genoa.
 *
 *  @sa include/numkong/spatial.h
 */
#ifndef NK_SPATIAL_GENOA_H
#define NK_SPATIAL_GENOA_H

#if NK_TARGET_X8664_
#if NK_TARGET_GENOA

#include "numkong/types.h"
#include "numkong/spatial/haswell.h" // `nk_angular_normalize_f32_haswell_`, `nk_f32_sqrt_haswell`
#include "numkong/reduce/skylake.h"  // `nk_reduce_add_f32x16_skylake_`
#include "numkong/cast/icelake.h"    // `nk_e4m3x32_to_bf16x32_icelake_`

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

NK_API_COMPTIME void nk_sqeuclidean_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 a_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_sq_f32x16 = _mm512_setzero_ps();
    __m512 ab_f32x16 = _mm512_setzero_ps();
    __m512i a_bf16x32, b_bf16x32;

nk_sqeuclidean_bf16_genoa_cycle:
    if (n < 32) {
        __mmask32 mask_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x32 = _mm512_maskz_loadu_epi16(mask_m32, a);
        b_bf16x32 = _mm512_maskz_loadu_epi16(mask_m32, b);
        n = 0;
    }
    else {
        a_bf16x32 = _mm512_loadu_epi16(a);
        b_bf16x32 = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    a_sq_f32x16 = _mm512_dpbf16_ps(a_sq_f32x16, nk_m512bh_from_m512i_(a_bf16x32), nk_m512bh_from_m512i_(a_bf16x32));
    b_sq_f32x16 = _mm512_dpbf16_ps(b_sq_f32x16, nk_m512bh_from_m512i_(b_bf16x32), nk_m512bh_from_m512i_(b_bf16x32));
    ab_f32x16 = _mm512_dpbf16_ps(ab_f32x16, nk_m512bh_from_m512i_(a_bf16x32), nk_m512bh_from_m512i_(b_bf16x32));
    if (n) goto nk_sqeuclidean_bf16_genoa_cycle;

    // (a-b)² = a² + b² - 2ab
    __m512 sum_sq_f32x16 = _mm512_add_ps(a_sq_f32x16, b_sq_f32x16);
    *result = nk_reduce_add_f32x16_skylake_(_mm512_fnmadd_ps(_mm512_set1_ps(2.0f), ab_f32x16, sum_sq_f32x16));
}

NK_API_COMPTIME void nk_euclidean_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_bf16_genoa(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_API_COMPTIME void nk_angular_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 dot_product_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512i a_bf16x32, b_bf16x32;

nk_angular_bf16_genoa_cycle:
    if (n < 32) {
        __mmask32 mask_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x32 = _mm512_maskz_loadu_epi16(mask_m32, a);
        b_bf16x32 = _mm512_maskz_loadu_epi16(mask_m32, b);
        n = 0;
    }
    else {
        a_bf16x32 = _mm512_loadu_epi16(a);
        b_bf16x32 = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    dot_product_f32x16 = _mm512_dpbf16_ps(dot_product_f32x16, nk_m512bh_from_m512i_(a_bf16x32),
                                          nk_m512bh_from_m512i_(b_bf16x32));
    a_norm_sq_f32x16 = _mm512_dpbf16_ps(a_norm_sq_f32x16, nk_m512bh_from_m512i_(a_bf16x32),
                                        nk_m512bh_from_m512i_(a_bf16x32));
    b_norm_sq_f32x16 = _mm512_dpbf16_ps(b_norm_sq_f32x16, nk_m512bh_from_m512i_(b_bf16x32),
                                        nk_m512bh_from_m512i_(b_bf16x32));
    if (n) goto nk_angular_bf16_genoa_cycle;

    nk_f32_t dot_product_f32 = nk_reduce_add_f32x16_skylake_(dot_product_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(a_norm_sq_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(b_norm_sq_f32x16);
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

#endif // NK_TARGET_GENOA
#endif // NK_TARGET_X8664_
#endif // NK_SPATIAL_GENOA_H
