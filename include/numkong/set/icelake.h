/**
 *  @file include/numkong/set/icelake.h
 *  @author Ash Vardanian
 *  @date March 23, 2023
 *  @brief SIMD-accelerated set similarity measures for Ice Lake.
 *
 *  @sa include/numkong/set.h
 *
 *  @section set_icelake_instructions Key AVX-512 Set Instructions
 *
 *  @verbatim
 *  Intrinsic                Instruction              Ice Lake
 *  _mm512_popcnt_epi64      VPOPCNTQ (ZMM, ZMM)      3cy @ p5
 *  _mm512_and_si512         VPANDQ (ZMM, ZMM, ZMM)   1cy @ p05
 *  _mm512_or_si512          VPORQ (ZMM, ZMM, ZMM)    1cy @ p05
 *  _mm512_xor_si512         VPXORQ (ZMM, ZMM, ZMM)   1cy @ p05
 *  _mm512_maskz_loadu_epi8  VMOVDQU8 (ZMM, mem, k1)  7cy @ p23
 *  @endverbatim
 *
 *  Ice Lake has native VPOPCNTQ instruction via AVX-512 VPOPCNTDQ extension, enabling efficient
 *  64-bit element-wise popcount. We process 512 bits per iteration.
 */
#ifndef NUMKONG_SET_ICELAKE_H
#define NUMKONG_SET_ICELAKE_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_ICELAKE_

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push( \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512vpopcntdq,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512vpopcntdq", "f16c", "fma", "bmi", "bmi2")
#endif

#if NUMKONG_TARGET_ICELAKE

#pragma region Binary Sets

NUMKONG_API nk_status_t nk_hamming_u1_icelake(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t n_bytes = n / NUMKONG_BITS_PER_BYTE;

    nk_u32_t xor_count;
    // It's harder to squeeze out performance from tiny representations, so we unroll the loops for binary metrics.
    if (n_bytes <= 64) { // Up to 512 bits.
        __mmask64 mask_m64 = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
        __m512i a_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, a);
        __m512i b_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, b);
        __m512i xor_popcount_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_u8x64, b_u8x64));
        xor_count = _mm512_reduce_add_epi64(xor_popcount_u64x8);
    }
    else if (n_bytes <= 128) { // Up to 1024 bits.
        __mmask64 mask_m64 = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes - 64);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, a + 64);
        __m512i b_two_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, b + 64);
        __m512i xor_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_one_u8x64, b_one_u8x64));
        __m512i xor_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_two_u8x64, b_two_u8x64));
        xor_count = _mm512_reduce_add_epi64(_mm512_add_epi64(xor_popcount_two_u64x8, xor_popcount_one_u64x8));
    }
    else if (n_bytes <= 192) { // Up to 1536 bits.
        __mmask64 mask_m64 = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes - 128);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_loadu_epi8(a + 64);
        __m512i b_two_u8x64 = _mm512_loadu_epi8(b + 64);
        __m512i a_three_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, a + 128);
        __m512i b_three_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, b + 128);
        __m512i xor_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_one_u8x64, b_one_u8x64));
        __m512i xor_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_two_u8x64, b_two_u8x64));
        __m512i xor_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_three_u8x64, b_three_u8x64));
        xor_count = _mm512_reduce_add_epi64(_mm512_add_epi64(
            xor_popcount_three_u64x8, _mm512_add_epi64(xor_popcount_two_u64x8, xor_popcount_one_u64x8)));
    }
    else if (n_bytes <= 256) { // Up to 2048 bits.
        __mmask64 mask_m64 = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes - 192);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_loadu_epi8(a + 64);
        __m512i b_two_u8x64 = _mm512_loadu_epi8(b + 64);
        __m512i a_three_u8x64 = _mm512_loadu_epi8(a + 128);
        __m512i b_three_u8x64 = _mm512_loadu_epi8(b + 128);
        __m512i a_four_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, a + 192);
        __m512i b_four_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, b + 192);
        __m512i xor_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_one_u8x64, b_one_u8x64));
        __m512i xor_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_two_u8x64, b_two_u8x64));
        __m512i xor_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_three_u8x64, b_three_u8x64));
        __m512i xor_popcount_four_u64x8 = _mm512_popcnt_epi64(_mm512_xor_si512(a_four_u8x64, b_four_u8x64));
        xor_count = _mm512_reduce_add_epi64(
            _mm512_add_epi64(_mm512_add_epi64(xor_popcount_four_u64x8, xor_popcount_three_u64x8),
                             _mm512_add_epi64(xor_popcount_two_u64x8, xor_popcount_one_u64x8)));
    }
    else {
        __m512i xor_popcount_u64x8 = _mm512_setzero_si512();
        __m512i a_u8x64, b_u8x64;

    nk_hamming_u1_icelake_cycle:
        if (n_bytes < 64) {
            __mmask64 mask_m64 = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
            a_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, a);
            b_u8x64 = _mm512_maskz_loadu_epi8(mask_m64, b);
            n_bytes = 0;
        }
        else {
            a_u8x64 = _mm512_loadu_epi8(a);
            b_u8x64 = _mm512_loadu_epi8(b);
            a += 64, b += 64, n_bytes -= 64;
        }
        __m512i xor_u8x64 = _mm512_xor_si512(a_u8x64, b_u8x64);
        xor_popcount_u64x8 = _mm512_add_epi64(xor_popcount_u64x8, _mm512_popcnt_epi64(xor_u8x64));
        if (n_bytes) goto nk_hamming_u1_icelake_cycle;

        xor_count = _mm512_reduce_add_epi64(xor_popcount_u64x8);
    }
    *result = xor_count;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_jaccard_u1_icelake(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t n_bytes = n / NUMKONG_BITS_PER_BYTE;

    nk_u32_t intersection_count = 0, union_count = 0;
    //  It's harder to squeeze out performance from tiny representations, so we unroll the loops for binary metrics.
    if (n_bytes <= 64) { // Up to 512 bits.
        __mmask64 load_m64 = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
        __m512i a_u8x64 = _mm512_maskz_loadu_epi8(load_m64, a);
        __m512i b_u8x64 = _mm512_maskz_loadu_epi8(load_m64, b);
        __m512i intersection_popcount_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_u8x64, b_u8x64));
        __m512i union_popcount_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_u8x64, b_u8x64));
        intersection_count = _mm512_reduce_add_epi64(intersection_popcount_u64x8);
        union_count = _mm512_reduce_add_epi64(union_popcount_u64x8);
    }
    else if (n_bytes <= 128) { // Up to 1024 bits.
        __mmask64 load_m64 = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes - 64);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_maskz_loadu_epi8(load_m64, a + 64);
        __m512i b_two_u8x64 = _mm512_maskz_loadu_epi8(load_m64, b + 64);
        __m512i intersection_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_one_u8x64, b_one_u8x64));
        __m512i union_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_one_u8x64, b_one_u8x64));
        __m512i intersection_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_two_u8x64, b_two_u8x64));
        __m512i union_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_two_u8x64, b_two_u8x64));
        intersection_count = _mm512_reduce_add_epi64(
            _mm512_add_epi64(intersection_popcount_two_u64x8, intersection_popcount_one_u64x8));
        union_count = _mm512_reduce_add_epi64(_mm512_add_epi64(union_popcount_two_u64x8, union_popcount_one_u64x8));
    }
    else if (n_bytes <= 192) { // Up to 1536 bits.
        __mmask64 load_m64 = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes - 128);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_loadu_epi8(a + 64);
        __m512i b_two_u8x64 = _mm512_loadu_epi8(b + 64);
        __m512i a_three_u8x64 = _mm512_maskz_loadu_epi8(load_m64, a + 128);
        __m512i b_three_u8x64 = _mm512_maskz_loadu_epi8(load_m64, b + 128);
        __m512i intersection_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_one_u8x64, b_one_u8x64));
        __m512i union_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_one_u8x64, b_one_u8x64));
        __m512i intersection_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_two_u8x64, b_two_u8x64));
        __m512i union_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_two_u8x64, b_two_u8x64));
        __m512i intersection_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_three_u8x64, b_three_u8x64));
        __m512i union_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_three_u8x64, b_three_u8x64));
        intersection_count = _mm512_reduce_add_epi64( //
            _mm512_add_epi64(intersection_popcount_three_u64x8,
                             _mm512_add_epi64(intersection_popcount_two_u64x8, intersection_popcount_one_u64x8)));
        union_count = _mm512_reduce_add_epi64( //
            _mm512_add_epi64(union_popcount_three_u64x8,
                             _mm512_add_epi64(union_popcount_two_u64x8, union_popcount_one_u64x8)));
    }
    else if (n_bytes <= 256) { // Up to 2048 bits.
        __mmask64 load_m64 = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes - 192);
        __m512i a_one_u8x64 = _mm512_loadu_epi8(a);
        __m512i b_one_u8x64 = _mm512_loadu_epi8(b);
        __m512i a_two_u8x64 = _mm512_loadu_epi8(a + 64);
        __m512i b_two_u8x64 = _mm512_loadu_epi8(b + 64);
        __m512i a_three_u8x64 = _mm512_loadu_epi8(a + 128);
        __m512i b_three_u8x64 = _mm512_loadu_epi8(b + 128);
        __m512i a_four_u8x64 = _mm512_maskz_loadu_epi8(load_m64, a + 192);
        __m512i b_four_u8x64 = _mm512_maskz_loadu_epi8(load_m64, b + 192);
        __m512i intersection_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_one_u8x64, b_one_u8x64));
        __m512i union_popcount_one_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_one_u8x64, b_one_u8x64));
        __m512i intersection_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_two_u8x64, b_two_u8x64));
        __m512i union_popcount_two_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_two_u8x64, b_two_u8x64));
        __m512i intersection_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_three_u8x64, b_three_u8x64));
        __m512i union_popcount_three_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_three_u8x64, b_three_u8x64));
        __m512i intersection_popcount_four_u64x8 = _mm512_popcnt_epi64(_mm512_and_si512(a_four_u8x64, b_four_u8x64));
        __m512i union_popcount_four_u64x8 = _mm512_popcnt_epi64(_mm512_or_si512(a_four_u8x64, b_four_u8x64));
        intersection_count = _mm512_reduce_add_epi64(
            _mm512_add_epi64(_mm512_add_epi64(intersection_popcount_four_u64x8, intersection_popcount_three_u64x8),
                             _mm512_add_epi64(intersection_popcount_two_u64x8, intersection_popcount_one_u64x8)));
        union_count = _mm512_reduce_add_epi64(
            _mm512_add_epi64(_mm512_add_epi64(union_popcount_four_u64x8, union_popcount_three_u64x8),
                             _mm512_add_epi64(union_popcount_two_u64x8, union_popcount_one_u64x8)));
    }
    else {
        __m512i intersection_popcount_u64x8 = _mm512_setzero_si512();
        __m512i union_popcount_u64x8 = _mm512_setzero_si512();
        __m512i a_u8x64, b_u8x64;

    nk_jaccard_u1_icelake_cycle:
        if (n_bytes < 64) {
            __mmask64 load_m64 = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
            a_u8x64 = _mm512_maskz_loadu_epi8(load_m64, a);
            b_u8x64 = _mm512_maskz_loadu_epi8(load_m64, b);
            n_bytes = 0;
        }
        else {
            a_u8x64 = _mm512_loadu_epi8(a);
            b_u8x64 = _mm512_loadu_epi8(b);
            a += 64, b += 64, n_bytes -= 64;
        }
        __m512i intersection_u8x64 = _mm512_and_si512(a_u8x64, b_u8x64);
        __m512i union_u8x64 = _mm512_or_si512(a_u8x64, b_u8x64);
        intersection_popcount_u64x8 = _mm512_add_epi64(intersection_popcount_u64x8,
                                                       _mm512_popcnt_epi64(intersection_u8x64));
        union_popcount_u64x8 = _mm512_add_epi64(union_popcount_u64x8, _mm512_popcnt_epi64(union_u8x64));
        if (n_bytes) goto nk_jaccard_u1_icelake_cycle;

        intersection_count = _mm512_reduce_add_epi64(intersection_popcount_u64x8);
        union_count = _mm512_reduce_add_epi64(union_popcount_u64x8);
    }
    *result = (union_count != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)union_count : 0.0f;
    return nk_success_k;
}

#pragma endregion Binary Sets

#pragma region Integer Sets

NUMKONG_API nk_status_t nk_jaccard_u32_icelake(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u32_t intersection_count = 0;
    nk_size_t n_remaining = n;
    for (; n_remaining >= 16; n_remaining -= 16, a += 16, b += 16) {
        __m512i a_u32x16 = _mm512_loadu_epi32(a);
        __m512i b_u32x16 = _mm512_loadu_epi32(b);
        __mmask16 equality_m16 = _mm512_cmpeq_epi32_mask(a_u32x16, b_u32x16);
        intersection_count += _mm_popcnt_u32((unsigned int)equality_m16);
    }
    if (n_remaining) {
        __mmask16 load_m16 = (__mmask16)_bzhi_u32(0xFFFF, n_remaining);
        __m512i a_u32x16 = _mm512_maskz_loadu_epi32(load_m16, a);
        __m512i b_u32x16 = _mm512_maskz_loadu_epi32(load_m16, b);
        __mmask16 equality_m16 = _mm512_mask_cmpeq_epi32_mask(load_m16, a_u32x16, b_u32x16);
        intersection_count += _mm_popcnt_u32((unsigned int)equality_m16);
    }
    *result = (n != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)n : 0.0f;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hamming_u8_icelake(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u32_t differences = 0;
    nk_size_t n_remaining = n;
    for (; n_remaining >= 64; n_remaining -= 64, a += 64, b += 64) {
        __m512i a_u8x64 = _mm512_loadu_si512((__m512i const *)a);
        __m512i b_u8x64 = _mm512_loadu_si512((__m512i const *)b);
        __mmask64 neq_m64 = _mm512_cmpneq_epi8_mask(a_u8x64, b_u8x64);
        differences += _mm_popcnt_u64(neq_m64);
    }
    if (n_remaining) {
        __mmask64 load_m64 = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_remaining);
        __m512i a_u8x64 = _mm512_maskz_loadu_epi8(load_m64, a);
        __m512i b_u8x64 = _mm512_maskz_loadu_epi8(load_m64, b);
        __mmask64 neq_m64 = _mm512_mask_cmpneq_epi8_mask(load_m64, a_u8x64, b_u8x64);
        differences += _mm_popcnt_u64(neq_m64);
    }
    *result = differences;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_jaccard_u16_icelake(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u32_t matches = 0;
    nk_size_t n_remaining = n;
    for (; n_remaining >= 32; n_remaining -= 32, a += 32, b += 32) {
        __m512i a_u16x32 = _mm512_loadu_si512((__m512i const *)a);
        __m512i b_u16x32 = _mm512_loadu_si512((__m512i const *)b);
        __mmask32 equality_m32 = _mm512_cmpeq_epi16_mask(a_u16x32, b_u16x32);
        matches += _mm_popcnt_u32(equality_m32);
    }
    if (n_remaining) {
        __mmask32 load_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, n_remaining);
        __m512i a_u16x32 = _mm512_maskz_loadu_epi16(load_m32, a);
        __m512i b_u16x32 = _mm512_maskz_loadu_epi16(load_m32, b);
        __mmask32 equality_m32 = _mm512_mask_cmpeq_epi16_mask(load_m32, a_u16x32, b_u16x32);
        matches += _mm_popcnt_u32(equality_m32);
    }
    *result = (n != 0) ? 1.0f - (nk_f32_t)matches / (nk_f32_t)n : 0.0f;
    return nk_success_k;
}

#pragma endregion Integer Sets

#endif // NUMKONG_TARGET_ICELAKE

#pragma region Distances from Dot Products

/** Hamming from_dot: computes pop_a + pop_b - 2*dot for 4 pairs (Icelake). */
NUMKONG_INLINE void nk_hamming_u32x4_from_dot_icelake_(nk_b128_vec_t const *dots_vec, nk_u32_t query_pop,
                                                       nk_b128_vec_t const *target_pops_vec,
                                                       nk_b128_vec_t *result_vec) {
    __m128i dots_i32x4 = dots_vec->xmm;
    __m128i query_i32x4 = _mm_set1_epi32((int)query_pop);
    __m128i target_i32x4 = target_pops_vec->xmm;
    result_vec->xmm = _mm_sub_epi32(_mm_add_epi32(query_i32x4, target_i32x4), _mm_slli_epi32(dots_i32x4, 1));
}

/** Jaccard from_dot: computes 1 - dot / (pop_a + pop_b - dot) for 4 pairs (Icelake). */
NUMKONG_INLINE void nk_jaccard_f32x4_from_dot_icelake_(nk_b128_vec_t const *dots_vec, nk_u32_t query_pop,
                                                       nk_b128_vec_t const *target_pops_vec,
                                                       nk_b128_vec_t *result_vec) {
    __m128 dot_f32x4 = _mm_cvtepi32_ps(dots_vec->xmm);
    __m128 query_f32x4 = _mm_set1_ps((nk_f32_t)query_pop);
    __m128 target_f32x4 = _mm_cvtepi32_ps(target_pops_vec->xmm);
    __m128 union_f32x4 = _mm_sub_ps(_mm_add_ps(query_f32x4, target_f32x4), dot_f32x4);

    __m128 zero_union_b32x4 = _mm_cmpeq_ps(union_f32x4, _mm_setzero_ps());
    __m128 one_f32x4 = _mm_set1_ps(1.0f);
    __m128 safe_union_f32x4 = _mm_blendv_ps(union_f32x4, one_f32x4, zero_union_b32x4);

    // Exact division rather than `_mm_rcp14_ps` — batched results must match the serial reference bit-for-bit.
    __m128 ratio_f32x4 = _mm_div_ps(dot_f32x4, safe_union_f32x4);
    __m128 jaccard_f32x4 = _mm_sub_ps(one_f32x4, ratio_f32x4);
    result_vec->xmm_ps = _mm_blendv_ps(jaccard_f32x4, _mm_setzero_ps(), zero_union_b32x4);
}

#pragma endregion Distances from Dot Products

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_ICELAKE_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_SET_ICELAKE_H
