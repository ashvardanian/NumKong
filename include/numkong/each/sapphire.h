/**
 *  @file include/numkong/each/sapphire.h
 *  @author Ash Vardanian
 *  @date October 19, 2024
 *  @brief SIMD-accelerated elementwise arithmetic for Sapphire Rapids.
 *
 *  @sa include/numkong/each.h
 *
 *  @section sapphire_elementwise_instructions Relevant Instructions
 *
 *  @verbatim
 *  Intrinsic                 Instruction                Sapphire   Genoa
 *  _mm512_add_ph             VADDPH (ZMM, ZMM, ZMM)     4cy @ p05  3cy @ p01
 *  _mm256_add_ph             VADDPH (YMM, YMM, YMM)     4cy @ p05  3cy @ p01
 *  _mm512_maskz_loadu_epi16  VMOVDQU16 (ZMM {K}, M512)  7cy @ p23  7cy @ p23
 *  _mm512_mask_storeu_epi16  VMOVDQU16 (M512 {K}, ZMM)  4cy @ p4   4cy @ p4
 *  @endverbatim
 */
#ifndef NK_EACH_SAPPHIRE_H
#define NK_EACH_SAPPHIRE_H

#if NK_TARGET_X8664_
#if NK_TARGET_SAPPHIRE

#include "numkong/types.h"
#include "numkong/cast/sapphire.h" // `nk_e4m3x16_to_f16x16_sapphire_`
#include "numkong/cast/icelake.h"  // `nk_cast_icelake`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512fp16,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512fp16", "f16c", "fma", "bmi", "bmi2")
#endif

NK_API_COMPTIME void nk_each_sum_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result) {
    __mmask32 mask_m32 = 0xFFFFFFFF;
    __m512h a_f16_vec, b_f16_vec;
    __m512h sum_f16_vec;
nk_each_sum_f16_sapphire_cycle:
    if (n < 32) {
        mask_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask_m32, a));
        b_f16_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask_m32, b));
        n = 0;
    }
    else {
        a_f16_vec = _mm512_castsi512_ph(_mm512_loadu_epi16(a));
        b_f16_vec = _mm512_castsi512_ph(_mm512_loadu_epi16(b));
        a += 32, b += 32, n -= 32;
    }
    sum_f16_vec = _mm512_add_ph(a_f16_vec, b_f16_vec);
    _mm512_mask_storeu_epi16(result, mask_m32, _mm512_castph_si512(sum_f16_vec));
    result += 32;
    if (n) goto nk_each_sum_f16_sapphire_cycle;
}

NK_API_COMPTIME void nk_each_sum_e4m3_sapphire(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_e4m3_t *result) {
    __m256i a_e4m3x32, b_e4m3x32;
    __m256h a_low_f16x16, a_high_f16x16, b_low_f16x16, b_high_f16x16;
    __m256h sum_low_f16x16, sum_high_f16x16;
    __m128i result_low_e4m3x16, result_high_e4m3x16;
    __mmask32 mask_m32 = 0xFFFFFFFF;
nk_each_sum_e4m3_sapphire_cycle:
    if (n < 32) {
        mask_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)n);
        a_e4m3x32 = _mm256_maskz_loadu_epi8(mask_m32, a);
        b_e4m3x32 = _mm256_maskz_loadu_epi8(mask_m32, b);
        n = 0;
    }
    else {
        a_e4m3x32 = _mm256_loadu_si256((__m256i const *)a);
        b_e4m3x32 = _mm256_loadu_si256((__m256i const *)b);
        a += 32, b += 32, n -= 32;
    }

    // Convert e4m3x16 → f16x16 (two halves)
    a_low_f16x16 = nk_e4m3x16_to_f16x16_sapphire_(_mm256_castsi256_si128(a_e4m3x32));
    a_high_f16x16 = nk_e4m3x16_to_f16x16_sapphire_(_mm256_extracti128_si256(a_e4m3x32, 1));
    b_low_f16x16 = nk_e4m3x16_to_f16x16_sapphire_(_mm256_castsi256_si128(b_e4m3x32));
    b_high_f16x16 = nk_e4m3x16_to_f16x16_sapphire_(_mm256_extracti128_si256(b_e4m3x32, 1));

    // Add in F16 - e4m3 sum is safe (max 896 < 65504)
    sum_low_f16x16 = _mm256_add_ph(a_low_f16x16, b_low_f16x16);
    sum_high_f16x16 = _mm256_add_ph(a_high_f16x16, b_high_f16x16);

    // Convert f16x16 → e4m3x16
    result_low_e4m3x16 = nk_f16x16_to_e4m3x16_sapphire_(sum_low_f16x16);
    result_high_e4m3x16 = nk_f16x16_to_e4m3x16_sapphire_(sum_high_f16x16);

    // Pack and store
    __m256i result_e4m3x32 = _mm256_inserti128_si256(_mm256_castsi128_si256(result_low_e4m3x16), result_high_e4m3x16,
                                                     1);
    _mm256_mask_storeu_epi8(result, mask_m32, result_e4m3x32);
    result += 32;
    if (n) goto nk_each_sum_e4m3_sapphire_cycle;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SAPPHIRE
#endif // NK_TARGET_X8664_
#endif // NK_EACH_SAPPHIRE_H
