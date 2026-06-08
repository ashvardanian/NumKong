/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for Genoa.
 *  @file include/numkong/spatial/genoa.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
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

NK_INTERNAL __m512i nk_substract_bf16x32_genoa_(__m512i a_i16, __m512i b_i16) {

    nk_b512_vec_t d, a_f32_even, b_f32_even, d_f32_even, a_f32_odd, b_f32_odd, d_f32_odd;

    // There are several approaches to perform subtraction in `bf16`. The first one is:
    //
    //      Perform a couple of casts - each is a bitshift. To convert `bf16` to `f32`,
    //      expand it to 32-bit integers, then shift the bits by 16 to the left.
    //      Then subtract as floats, and shift back. During expansion, we will double the space,
    //      and should use separate registers for top and bottom halves.
    //      Some compilers don't have `_mm512_extracti32x8_epi32`, so we use `_mm512_extracti64x4_epi64`:
    //
    //          a_f32_bot.fvec = _mm512_castsi512_ps(_mm512_slli_epi32(
    //              _mm512_cvtepu16_epi32(_mm512_castsi512_si256(a_i16)), 16));
    //          b_f32_bot.fvec = _mm512_castsi512_ps(_mm512_slli_epi32(
    //              _mm512_cvtepu16_epi32(_mm512_castsi512_si256(b_i16)), 16));
    //          a_f32_top.fvec =_mm512_castsi512_ps(
    //              _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(a_i16, 1)), 16));
    //          b_f32_top.fvec =_mm512_castsi512_ps(
    //              _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(b_i16, 1)), 16));
    //          d_f32_top.fvec = _mm512_sub_ps(a_f32_top.fvec, b_f32_top.fvec);
    //          d_f32_bot.fvec = _mm512_sub_ps(a_f32_bot.fvec, b_f32_bot.fvec);
    //          d.ivec = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(
    //              _mm512_srli_epi32(_mm512_castps_si512(d_f32_bot.fvec), 16)));
    //          d.ivec = _mm512_inserti64x4(d.ivec, _mm512_cvtepi32_epi16(
    //              _mm512_srli_epi32(_mm512_castps_si512(d_f32_top.fvec), 16)), 1);
    //
    // Instead of using multple shifts and an insertion, we can achieve similar result with fewer expensive
    // calls to `_mm512_permutex2var_epi16`, or a cheap `_mm512_mask_shuffle_epi8` and blend:
    //
    a_f32_odd.zmm = _mm512_and_si512(a_i16, _mm512_set1_epi32(0xFFFF0000));
    a_f32_even.zmm = _mm512_slli_epi32(a_i16, 16);
    b_f32_odd.zmm = _mm512_and_si512(b_i16, _mm512_set1_epi32(0xFFFF0000));
    b_f32_even.zmm = _mm512_slli_epi32(b_i16, 16);

    d_f32_odd.zmm_ps = _mm512_sub_ps(a_f32_odd.zmm_ps, b_f32_odd.zmm_ps);
    d_f32_even.zmm_ps = _mm512_sub_ps(a_f32_even.zmm_ps, b_f32_even.zmm_ps);

    d_f32_even.zmm = _mm512_srli_epi32(d_f32_even.zmm, 16);
    d.zmm = _mm512_mask_blend_epi16(0x55555555, d_f32_odd.zmm, d_f32_even.zmm);

    return d.zmm;
}

NK_PUBLIC void nk_sqeuclidean_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 a_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_sq_f32x16 = _mm512_setzero_ps();
    __m512 ab_f32x16 = _mm512_setzero_ps();
    __m512i a_bf16x32, b_bf16x32;

nk_sqeuclidean_bf16_genoa_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x32 = _mm512_maskz_loadu_epi16(mask, a);
        b_bf16x32 = _mm512_maskz_loadu_epi16(mask, b);
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

NK_PUBLIC void nk_euclidean_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_bf16_genoa(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_PUBLIC void nk_angular_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 dot_product_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512i a_bf16x32, b_bf16x32;

nk_angular_bf16_genoa_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x32 = _mm512_maskz_loadu_epi16(mask, a);
        b_bf16x32 = _mm512_maskz_loadu_epi16(mask, b);
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
