/**
 *  @file include/numkong/set/neon.h
 *  @author Ash Vardanian
 *  @date March 23, 2023
 *  @brief SIMD-accelerated set similarity measures for NEON.
 *
 *  @sa include/numkong/set.h
 *
 *  @section set_neon_instructions NEON Set Instructions
 *
 *  Key NEON instructions for binary/bitwise operations (Cortex-A76 class):
 *
 *  @verbatim
 *  Intrinsic   Instruction                A76       M5
 *  vcntq_u8    CNT (V.16B, V.16B)         2cy @ 2p  2cy @ 4p
 *  veorq_u8    EOR (V.16B, V.16B, V.16B)  1cy @ 2p  2cy @ 4p
 *  vandq_u8    AND (V.16B, V.16B, V.16B)  1cy @ 2p  2cy @ 4p
 *  vorrq_u8    ORR (V.16B, V.16B, V.16B)  1cy @ 2p  2cy @ 4p
 *  vpaddlq_u8  UADDLP (V.8H, V.16B)       2cy @ 2p  2cy @ 4p
 *  vaddvq_u32  ADDV (S, V.4S)             4cy @ 1p  5cy @ 1p
 *  @endverbatim
 *
 *  According to the available literature, the throughput for those basic integer ops is identical
 *  across most Apple, Qualcomm, and AWS Graviton chips. As long as we avoid widening operations and
 *  horizontal reductions, we won't face any reasonable bottlenecks.
 */
#ifndef NUMKONG_SET_NEON_H
#define NUMKONG_SET_NEON_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_NEON_

#include "numkong/types.h"      // `nk_u1x8_t`
#include "numkong/set/serial.h" // `nk_u1x8_popcount_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

#pragma region Binary Sets

/** Counts the bits that differ between two @p n -bit sets. */
NUMKONG_INLINE nk_u32_t nk_u1_xor_popcount_neon_(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n) {
    nk_size_t n_bytes = n / NUMKONG_BITS_PER_BYTE;
    nk_u32_t differences = 0;
    nk_size_t i = 0;
    // In each 8-bit word we may have up to 8 differences.
    // So for up-to 31 cycles (31 * 16 = 496 word-dimensions = 3968 bits)
    // we can aggregate the differences into a `uint8x16_t` vector,
    // where each component will be up-to 255.
    while (i + 16 <= n_bytes) {
        uint8x16_t popcount_u8x16 = vdupq_n_u8(0);
        for (nk_size_t cycle = 0; cycle < 31 && i + 16 <= n_bytes; ++cycle, i += 16) {
            uint8x16_t a_u8x16 = vld1q_u8(a + i);
            uint8x16_t b_u8x16 = vld1q_u8(b + i);
            uint8x16_t xor_popcount_u8x16 = vcntq_u8(veorq_u8(a_u8x16, b_u8x16));
            popcount_u8x16 = vaddq_u8(popcount_u8x16, xor_popcount_u8x16);
        }
        differences += (nk_u32_t)vaddlvq_u8(popcount_u8x16);
    }
    // Handle the tail
    for (; i != n_bytes; ++i) differences += nk_u1x8_popcount_(a[i] ^ b[i]);
    return differences;
}

/** Counts the bits in the intersection and in the union of two @p n -bit sets. */
NUMKONG_INLINE void nk_u1_and_or_popcounts_neon_(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n,
                                                 nk_u32_t *intersection_count_ptr, nk_u32_t *union_count_ptr) {
    nk_size_t n_bytes = n / NUMKONG_BITS_PER_BYTE;
    nk_u32_t intersection_count = 0, union_count = 0;
    nk_size_t i = 0;
    // In each 8-bit word we may have up to 8 intersections/unions.
    // So for up-to 31 cycles (31 * 16 = 496 word-dimensions = 3968 bits)
    // we can aggregate the intersections/unions into a `uint8x16_t` vector,
    // where each component will be up-to 255.
    while (i + 16 <= n_bytes) {
        uint8x16_t intersection_popcount_u8x16 = vdupq_n_u8(0);
        uint8x16_t union_popcount_u8x16 = vdupq_n_u8(0);
        for (nk_size_t cycle = 0; cycle < 31 && i + 16 <= n_bytes; ++cycle, i += 16) {
            uint8x16_t a_u8x16 = vld1q_u8(a + i);
            uint8x16_t b_u8x16 = vld1q_u8(b + i);
            intersection_popcount_u8x16 = vaddq_u8(intersection_popcount_u8x16, vcntq_u8(vandq_u8(a_u8x16, b_u8x16)));
            union_popcount_u8x16 = vaddq_u8(union_popcount_u8x16, vcntq_u8(vorrq_u8(a_u8x16, b_u8x16)));
        }
        intersection_count += (nk_u32_t)vaddlvq_u8(intersection_popcount_u8x16);
        union_count += (nk_u32_t)vaddlvq_u8(union_popcount_u8x16);
    }
    // Handle the tail
    for (; i != n_bytes; ++i)
        intersection_count += nk_u1x8_popcount_(a[i] & b[i]), union_count += nk_u1x8_popcount_(a[i] | b[i]);
    *intersection_count_ptr = intersection_count, *union_count_ptr = union_count;
}

#if NUMKONG_TARGET_NEON

NUMKONG_API nk_status_t nk_hamming_u1_neon(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    *result = nk_u1_xor_popcount_neon_(a, b, n);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_jaccard_u1_neon(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u32_t intersection_count, union_count;
    nk_u1_and_or_popcounts_neon_(a, b, n, &intersection_count, &union_count);
    *result = (union_count != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)union_count : 0.0f;
    return nk_success_k;
}

#pragma endregion Binary Sets

#pragma region Integer Sets

NUMKONG_API nk_status_t nk_jaccard_u32_neon(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u32_t intersection_count = 0;
    nk_size_t i = 0;
    uint32x4_t intersection_count_u32x4 = vdupq_n_u32(0);
    for (; i + 4 <= n; i += 4) {
        uint32x4_t a_u32x4 = vld1q_u32(a + i);
        uint32x4_t b_u32x4 = vld1q_u32(b + i);
        uint32x4_t equality_u32x4 = vceqq_u32(a_u32x4, b_u32x4);
        intersection_count_u32x4 = vaddq_u32(intersection_count_u32x4, vshrq_n_u32(equality_u32x4, 31));
    }
    intersection_count += vaddvq_u32(intersection_count_u32x4);
    for (; i != n; ++i) intersection_count += (a[i] == b[i]);
    *result = (n != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)n : 0.0f;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hamming_u8_neon(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t i = 0;
    uint32x4_t diff_count_u32x4 = vdupq_n_u32(0);
    // Process 16 bytes at a time using NEON with widening adds to avoid overflow.
    // Uses pairwise widening chain: 16 u8 → 8 u16 → 4 u32 per iteration.
    for (; i + 16 <= n; i += 16) {
        uint8x16_t a_u8x16 = vld1q_u8(a + i);
        uint8x16_t b_u8x16 = vld1q_u8(b + i);
        // vceqq_u8 returns 0xFF for equal, 0x00 for not-equal
        // Invert to get 0xFF for not-equal, then shift right by 7 to get 1
        uint8x16_t not_equal_u8x16 = vmvnq_u8(vceqq_u8(a_u8x16, b_u8x16));
        uint8x16_t diff_u8x16 = vshrq_n_u8(not_equal_u8x16, 7);
        // Widen: 16 u8 → 8 u16 → 4 u32 using pairwise add and widen
        uint16x8_t diff_u16x8 = vpaddlq_u8(diff_u8x16);
        uint32x4_t diff_u32x4 = vpaddlq_u16(diff_u16x8);
        diff_count_u32x4 = vaddq_u32(diff_count_u32x4, diff_u32x4);
    }
    nk_u32_t differences = vaddvq_u32(diff_count_u32x4);
    // Handle tail elements
    for (; i != n; ++i) differences += (a[i] != b[i]);
    *result = differences;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_jaccard_u16_neon(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u32_t matches = 0;
    nk_size_t i = 0;
    uint32x4_t match_count_u32x4 = vdupq_n_u32(0);
    // Process 8 u16 values at a time using NEON
    for (; i + 8 <= n; i += 8) {
        uint16x8_t a_u16x8 = vld1q_u16(a + i);
        uint16x8_t b_u16x8 = vld1q_u16(b + i);
        // vceqq_u16 returns 0xFFFF for equal, 0x0000 for not-equal
        uint16x8_t equality_u16x8 = vceqq_u16(a_u16x8, b_u16x8);
        // Count matches by shifting right by 15 to get 1 for match, 0 for non-match
        // Then widen and accumulate into u32
        uint16x8_t match_u16x8 = vshrq_n_u16(equality_u16x8, 15);
        // Pairwise add and widen to u32
        uint32x4_t match_u32x4 = vpaddlq_u16(match_u16x8);
        match_count_u32x4 = vaddq_u32(match_count_u32x4, match_u32x4);
    }
    matches += vaddvq_u32(match_count_u32x4);
    // Handle tail elements
    for (; i != n; ++i) matches += (a[i] == b[i]);
    *result = (n != 0) ? 1.0f - (nk_f32_t)matches / (nk_f32_t)n : 0.0f;
    return nk_success_k;
}

#pragma endregion Integer Sets

#endif // NUMKONG_TARGET_NEON

#pragma region Distances from Dot Products

/** Hamming from_dot: computes pop_a + pop_b - 2*dot for 4 pairs (NEON). */
NUMKONG_INLINE void nk_hamming_u32x4_from_dot_neon_(nk_b128_vec_t const *dots_vec, nk_u32_t query_pop,
                                                    nk_b128_vec_t const *target_pops_vec, nk_b128_vec_t *result_vec) {
    uint32x4_t dots_u32x4 = dots_vec->u32x4;
    uint32x4_t query_u32x4 = vdupq_n_u32(query_pop);
    uint32x4_t target_u32x4 = target_pops_vec->u32x4;
    result_vec->u32x4 = vsubq_u32(vaddq_u32(query_u32x4, target_u32x4), vshlq_n_u32(dots_u32x4, 1));
}

/** Jaccard from_dot: computes 1 - dot / (pop_a + pop_b - dot) for 4 pairs (NEON). */
NUMKONG_INLINE void nk_jaccard_f32x4_from_dot_neon_(nk_b128_vec_t const *dots_vec, nk_u32_t query_pop,
                                                    nk_b128_vec_t const *target_pops_vec, nk_b128_vec_t *result_vec) {
    float32x4_t dot_f32x4 = vcvtq_f32_u32(dots_vec->u32x4);
    float32x4_t query_f32x4 = vdupq_n_f32((nk_f32_t)query_pop);
    float32x4_t target_f32x4 = vcvtq_f32_u32(target_pops_vec->u32x4);
    float32x4_t union_f32x4 = vsubq_f32(vaddq_f32(query_f32x4, target_f32x4), dot_f32x4);

    float32x4_t one_f32x4 = vdupq_n_f32(1.0f);
    uint32x4_t zero_union_u32x4 = vceqq_f32(union_f32x4, vdupq_n_f32(0.0f));
    float32x4_t safe_union_f32x4 = vbslq_f32(zero_union_u32x4, one_f32x4, union_f32x4);

    // Exact division rather than `vrecpeq_f32` — batched results must match the serial reference bit-for-bit.
    float32x4_t ratio_f32x4 = vdivq_f32(dot_f32x4, safe_union_f32x4);
    float32x4_t jaccard_f32x4 = vsubq_f32(one_f32x4, ratio_f32x4);
    result_vec->f32x4 = vbslq_f32(zero_union_u32x4, vdupq_n_f32(0.0f), jaccard_f32x4);
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

#endif // NUMKONG_ARCH_ARM64_NEON_
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_SET_NEON_H
