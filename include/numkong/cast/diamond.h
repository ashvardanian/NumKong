/**
 *  @brief SIMD-accelerated Type Conversions for Diamond Rapids.
 *  @file include/numkong/cast/diamond.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/cast/icelake.h
 *
 *  Uses VCVTHF82PH (E4M3→FP16) and VCVTBF82PH (E5M2→FP16) for native 1-instruction
 *  FP8→FP16 conversion. Both conversions are exact (no rounding needed).
 */
#ifndef NK_CAST_DIAMOND_H
#define NK_CAST_DIAMOND_H

#if NK_TARGET_X8664_
#if NK_TARGET_DIAMOND

#include "numkong/types.h"

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

NK_HELPER_INLINE void nk_load_e4m3x32_to_f16x32_diamond_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm = nk_m512i_from_m512h_(_mm512_cvthf8_ph(_mm256_loadu_epi8(src)));
}

NK_HELPER_INLINE void nk_partial_load_e4m3x32_to_f16x32_diamond_(void const *src, nk_b512_vec_t *dst, nk_size_t count) {
    __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count);
    dst->zmm = nk_m512i_from_m512h_(_mm512_cvthf8_ph(_mm256_maskz_loadu_epi8(mask, src)));
}

NK_HELPER_INLINE void nk_load_e5m2x32_to_f16x32_diamond_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm = nk_m512i_from_m512h_(_mm512_cvtbf8_ph(_mm256_loadu_epi8(src)));
}

NK_HELPER_INLINE void nk_partial_load_e5m2x32_to_f16x32_diamond_(void const *src, nk_b512_vec_t *dst, nk_size_t count) {
    __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count);
    dst->zmm = nk_m512i_from_m512h_(_mm512_cvtbf8_ph(_mm256_maskz_loadu_epi8(mask, src)));
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
#endif // NK_CAST_DIAMOND_H
