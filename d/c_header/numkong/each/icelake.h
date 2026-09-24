/**
 *  @brief SIMD-accelerated Elementwise Arithmetic for Ice Lake.
 *  @file include/numkong/each/icelake.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/each.h
 *
 *  @section ice_elementwise_instructions Relevant Instructions
 *
 *      Intrinsic                   Instruction                     Ice         Genoa
 *      _mm512_add_epi8             VPADDB (ZMM, ZMM, ZMM)          1cy @ p05   1cy @ p0123
 *      _mm512_adds_epi8            VPADDSB (ZMM, ZMM, ZMM)         1cy @ p05   1cy @ p0123
 *      _mm512_add_epi32            VPADDD (ZMM, ZMM, ZMM)          1cy @ p05   1cy @ p0123
 *      _mm512_cmpgt_epi32_mask     VPCMPGTD (K, ZMM, ZMM)          3cy @ p5    3cy @ p0
 *      _mm512_mask_blend_epi32     VPBLENDMD (ZMM, K, ZMM, ZMM)    1cy @ p05   1cy @ p0123
 *      _mm512_maskz_loadu_epi8     VMOVDQU8 (ZMM {K}, M512)        7cy @ p23   7cy @ p23
 *
 *  Ice Lake inherits Skylake's AVX-512 execution but reduces frequency throttling on client chips.
 *  Integer saturation arithmetic (VPADDSB, VPADDUSB) provides 1cy latency for overflow-safe addition.
 *  For i32/i64 saturation, manual overflow detection via compare-and-blend is required.
 */
#ifndef NK_EACH_ICELAKE_H
#define NK_EACH_ICELAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_ICELAKE

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vnni", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC void nk_each_sum_i8_icelake(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result) {
    __mmask64 mask = 0xFFFFFFFFFFFFFFFF;
    __m512i a_i8_vec, b_i8_vec;
    __m512i sum_i8_vec;
nk_each_sum_i8_icelake_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_i8_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_i8_vec = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_i8_vec = _mm512_loadu_epi8(a);
        b_i8_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n -= 64;
    }
    sum_i8_vec = _mm512_adds_epi8(a_i8_vec, b_i8_vec);
    _mm512_mask_storeu_epi8(result, mask, sum_i8_vec);
    result += 64;
    if (n) goto nk_each_sum_i8_icelake_cycle;
}

NK_PUBLIC void nk_each_sum_u8_icelake(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result) {
    __mmask64 mask = 0xFFFFFFFFFFFFFFFF;
    __m512i a_u8_vec, b_u8_vec;
    __m512i sum_u8_vec;
nk_each_sum_u8_icelake_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_u8_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_u8_vec = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_u8_vec = _mm512_loadu_epi8(a);
        b_u8_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n -= 64;
    }
    sum_u8_vec = _mm512_adds_epu8(a_u8_vec, b_u8_vec);
    _mm512_mask_storeu_epi8(result, mask, sum_u8_vec);
    result += 64;
    if (n) goto nk_each_sum_u8_icelake_cycle;
}

NK_PUBLIC void nk_each_sum_i16_icelake(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_i16_t *result) {
    __mmask32 mask = 0xFFFFFFFF;
    __m512i a_i16_vec, b_i16_vec;
    __m512i sum_i16_vec;
nk_each_sum_i16_icelake_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_i16_vec = _mm512_maskz_loadu_epi16(mask, a);
        b_i16_vec = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_i16_vec = _mm512_loadu_epi16(a);
        b_i16_vec = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    sum_i16_vec = _mm512_adds_epi16(a_i16_vec, b_i16_vec);
    _mm512_mask_storeu_epi16(result, mask, sum_i16_vec);
    result += 32;
    if (n) goto nk_each_sum_i16_icelake_cycle;
}

NK_PUBLIC void nk_each_sum_u16_icelake(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_u16_t *result) {
    __mmask32 mask = 0xFFFFFFFF;
    __m512i a_u16_vec, b_u16_vec;
    __m512i sum_u16_vec;
nk_each_sum_u16_icelake_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_u16_vec = _mm512_maskz_loadu_epi16(mask, a);
        b_u16_vec = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_u16_vec = _mm512_loadu_epi16(a);
        b_u16_vec = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    sum_u16_vec = _mm512_adds_epu16(a_u16_vec, b_u16_vec);
    _mm512_mask_storeu_epi16(result, mask, sum_u16_vec);
    result += 32;
    if (n) goto nk_each_sum_u16_icelake_cycle;
}

NK_INTERNAL __m512i _mm512_adds_epi32_icelake(__m512i a, __m512i b) {
    __m512i sum_i32x16 = _mm512_add_epi32(a, b);
    __m512i sign_i32x16 = _mm512_set1_epi32((int)0x80000000);
    // ~(a^b) & (sum^a): overflow iff same-sign inputs produce different-sign result
    __m512i overflow_i32x16 = _mm512_ternarylogic_epi64(a, b, sum_i32x16, 0x42);
    __mmask16 overflow_b32x16 = _mm512_test_epi32_mask(overflow_i32x16, sign_i32x16);
    // Positive overflow → INT32_MAX, negative overflow → INT32_MIN
    __m512i max_i32x16 = _mm512_set1_epi32(0x7FFFFFFF);
    __m512i min_i32x16 = _mm512_set1_epi32((int)0x80000000);
    __m512i saturated_i32x16 = _mm512_mask_blend_epi32(_mm512_movepi32_mask(a), max_i32x16, min_i32x16);
    return _mm512_mask_blend_epi32(overflow_b32x16, sum_i32x16, saturated_i32x16);
}

NK_INTERNAL __m512i _mm512_adds_epu32_icelake(__m512i a, __m512i b) {
    __m512i sum = _mm512_add_epi32(a, b);
    __mmask16 overflow_mask = _mm512_cmp_epu32_mask(sum, a, _MM_CMPINT_LT); // sum < a means overflow
    __m512i max_val = _mm512_set1_epi32(4294967295u);
    return _mm512_mask_blend_epi32(overflow_mask, sum, max_val);
}

NK_INTERNAL __m512i _mm512_adds_epi64_icelake(__m512i a, __m512i b) {
    __m512i sum_i64x8 = _mm512_add_epi64(a, b);
    __m512i sign_i64x8 = _mm512_set1_epi64((long long)0x8000000000000000);
    // ~(a^b) & (sum^a): overflow iff same-sign inputs produce different-sign result
    __m512i overflow_i64x8 = _mm512_ternarylogic_epi64(a, b, sum_i64x8, 0x42);
    __mmask8 overflow_b64x8 = _mm512_test_epi64_mask(overflow_i64x8, sign_i64x8);
    // Positive overflow → INT64_MAX, negative overflow → INT64_MIN
    __m512i max_i64x8 = _mm512_set1_epi64(9223372036854775807ll);
    __m512i min_i64x8 = _mm512_set1_epi64(-9223372036854775807ll - 1);
    __m512i saturated_i64x8 = _mm512_mask_blend_epi64(_mm512_movepi64_mask(a), max_i64x8, min_i64x8);
    return _mm512_mask_blend_epi64(overflow_b64x8, sum_i64x8, saturated_i64x8);
}

NK_INTERNAL __m512i _mm512_adds_epu64_icelake(__m512i a, __m512i b) {
    __m512i sum = _mm512_add_epi64(a, b);
    __mmask8 overflow_mask = _mm512_cmp_epu64_mask(sum, a, _MM_CMPINT_LT); // sum < a means overflow
    __m512i max_val = _mm512_set1_epi64(18446744073709551615ull);
    return _mm512_mask_blend_epi64(overflow_mask, sum, max_val);
}

NK_PUBLIC void nk_each_sum_i32_icelake(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_i32_t *result) {
    __mmask16 mask = 0xFFFF;
    __m512i a_i32_vec, b_i32_vec;
    __m512i sum_i32_vec;
nk_each_sum_i32_icelake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i32_vec = _mm512_maskz_loadu_epi32(mask, a);
        b_i32_vec = _mm512_maskz_loadu_epi32(mask, b);
        n = 0;
    }
    else {
        a_i32_vec = _mm512_loadu_epi32(a);
        b_i32_vec = _mm512_loadu_epi32(b);
        a += 16, b += 16, n -= 16;
    }
    sum_i32_vec = _mm512_adds_epi32_icelake(a_i32_vec, b_i32_vec);
    _mm512_mask_storeu_epi32(result, mask, sum_i32_vec);
    result += 16;
    if (n) goto nk_each_sum_i32_icelake_cycle;
}

NK_PUBLIC void nk_each_sum_u32_icelake(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_u32_t *result) {
    __mmask16 mask = 0xFFFF;
    __m512i a_u32_vec, b_u32_vec;
    __m512i sum_u32_vec;
nk_each_sum_u32_icelake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u32_vec = _mm512_maskz_loadu_epi32(mask, a);
        b_u32_vec = _mm512_maskz_loadu_epi32(mask, b);
        n = 0;
    }
    else {
        a_u32_vec = _mm512_loadu_epi32(a);
        b_u32_vec = _mm512_loadu_epi32(b);
        a += 16, b += 16, n -= 16;
    }
    sum_u32_vec = _mm512_adds_epu32_icelake(a_u32_vec, b_u32_vec);
    _mm512_mask_storeu_epi32(result, mask, sum_u32_vec);
    result += 16;
    if (n) goto nk_each_sum_u32_icelake_cycle;
}

NK_PUBLIC void nk_each_sum_i64_icelake(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_i64_t *result) {
    __mmask8 mask = 0xFF;
    __m512i a_i64_vec, b_i64_vec;
    __m512i sum_i64_vec;
nk_each_sum_i64_icelake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i64_vec = _mm512_maskz_loadu_epi64(mask, a);
        b_i64_vec = _mm512_maskz_loadu_epi64(mask, b);
        n = 0;
    }
    else {
        a_i64_vec = _mm512_loadu_epi64(a);
        b_i64_vec = _mm512_loadu_epi64(b);
        a += 8, b += 8, n -= 8;
    }
    sum_i64_vec = _mm512_adds_epi64_icelake(a_i64_vec, b_i64_vec);
    _mm512_mask_storeu_epi64(result, mask, sum_i64_vec);
    result += 8;
    if (n) goto nk_each_sum_i64_icelake_cycle;
}

NK_PUBLIC void nk_each_sum_u64_icelake(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_u64_t *result) {
    __mmask8 mask = 0xFF;
    __m512i a_u64_vec, b_u64_vec;
    __m512i sum_u64_vec;
nk_each_sum_u64_icelake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u64_vec = _mm512_maskz_loadu_epi64(mask, a);
        b_u64_vec = _mm512_maskz_loadu_epi64(mask, b);
        n = 0;
    }
    else {
        a_u64_vec = _mm512_loadu_epi64(a);
        b_u64_vec = _mm512_loadu_epi64(b);
        a += 8, b += 8, n -= 8;
    }
    sum_u64_vec = _mm512_adds_epu64_icelake(a_u64_vec, b_u64_vec);
    _mm512_mask_storeu_epi64(result, mask, sum_u64_vec);
    result += 8;
    if (n) goto nk_each_sum_u64_icelake_cycle;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_ICELAKE
#endif // NK_TARGET_X86_
#endif // NK_EACH_ICELAKE_H
