/**
 *  @file include/numkong/cast/diamond.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *  @brief SIMD-accelerated Type Conversions for Diamond Rapids.
 *
 *  @sa include/numkong/cast/icelake.h
 *
 *  Uses VCVTHF82PH for E4M3 → FP16 and VCVTBF82PH for E5M2 → FP16, both native single-instruction
 *  conversions, exact and needing no rounding.
 */
#ifndef NUMKONG_CAST_DIAMOND_H
#define NUMKONG_CAST_DIAMOND_H

#if NUMKONG_ARCH_X86_64_
#if NUMKONG_TARGET_DIAMOND

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__) && __clang_major__ >= 22
#pragma clang attribute push(                                                                                \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx10.2,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                    \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx10.2-512,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "avx10.2", "f16c", "fma", \
                   "bmi", "bmi2")
#endif

NUMKONG_HELPER_INLINE void nk_load_e4m3x32_to_f16x32_diamond_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm = nk_m512i_from_m512h_(_mm512_cvthf8_ph(_mm256_loadu_epi8(src)));
}

NUMKONG_HELPER_INLINE void nk_partial_load_e4m3x32_to_f16x32_diamond_(void const *src, nk_b512_vec_t *dst,
                                                                      nk_size_t count) {
    __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count);
    dst->zmm = nk_m512i_from_m512h_(_mm512_cvthf8_ph(_mm256_maskz_loadu_epi8(mask, src)));
}

NUMKONG_HELPER_INLINE void nk_load_e5m2x32_to_f16x32_diamond_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm = nk_m512i_from_m512h_(_mm512_cvtbf8_ph(_mm256_loadu_epi8(src)));
}

NUMKONG_HELPER_INLINE void nk_partial_load_e5m2x32_to_f16x32_diamond_(void const *src, nk_b512_vec_t *dst,
                                                                      nk_size_t count) {
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

#endif // NUMKONG_TARGET_DIAMOND
#endif // NUMKONG_ARCH_X86_64_
#endif // NUMKONG_CAST_DIAMOND_H
