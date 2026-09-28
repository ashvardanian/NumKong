/**
 *  @file include/numkong/set/haswell.h
 *  @author Ash Vardanian
 *  @date March 3, 2024
 *  @brief SIMD-accelerated set similarity measures for Haswell.
 *
 *  @sa include/numkong/set.h
 *
 *  @section set_haswell_instructions Key POPCNT/AVX2 Set Instructions
 *
 *  @verbatim
 *  Intrinsic                 Instruction                  Haswell     Genoa
 *  _mm_popcnt_u64            POPCNT (R64, R64)            3cy @ p1    1cy @ p0123
 *  _mm256_and_si256          VPAND (YMM, YMM, YMM)        1cy @ p015  1cy @ p0123
 *  _mm256_or_si256           VPOR (YMM, YMM, YMM)         1cy @ p015  1cy @ p0123
 *  _mm256_xor_si256          VPXOR (YMM, YMM, YMM)        1cy @ p015  1cy @ p0123
 *  _mm256_extracti128_si256  VEXTRACTI128 (XMM, YMM, I8)  3cy @ p5    1cy @ p0123
 *  @endverbatim
 *
 *  Haswell lacks SIMD popcount; we extract 64-bit words and use scalar POPCNT. The p1 port
 *  bottleneck limits throughput to 1 popcount/cycle. For Hamming distance, XOR + POPCNT; for
 *  Jaccard, compute AND/OR + POPCNT separately to get intersection and union counts.
 */
#ifndef NUMKONG_SET_HASWELL_H
#define NUMKONG_SET_HASWELL_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_HASWELL_

#include "numkong/types.h"
#include "numkong/set/serial.h" // `nk_u1x8_popcount_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,sse4.1,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "sse4.1", "popcnt")
#endif

#pragma region Binary Sets

#if NUMKONG_TARGET_HASWELL
NUMKONG_API nk_status_t nk_hamming_u1_haswell(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                              void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t n_bytes = n / NUMKONG_BITS_PER_BYTE;
    // x86 supports unaligned loads and works just fine with the scalar version for small vectors.
    nk_u32_t differences = 0;
    for (; n_bytes >= 8; n_bytes -= 8, a += 8, b += 8)
        differences += _mm_popcnt_u64(*(nk_u64_t const *)a ^ *(nk_u64_t const *)b);
    for (; n_bytes; --n_bytes, ++a, ++b) differences += _mm_popcnt_u32(*a ^ *b);
    *result = differences;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_jaccard_u1_haswell(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                              void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t n_bytes = n / NUMKONG_BITS_PER_BYTE;
    // x86 supports unaligned loads and works just fine with the scalar version for small vectors.
    nk_u32_t intersection_count = 0, union_count = 0;
    for (; n_bytes >= 8; n_bytes -= 8, a += 8, b += 8)
        intersection_count += (nk_u32_t)_mm_popcnt_u64(*(nk_u64_t const *)a & *(nk_u64_t const *)b),
            union_count += (nk_u32_t)_mm_popcnt_u64(*(nk_u64_t const *)a | *(nk_u64_t const *)b);
    for (; n_bytes; --n_bytes, ++a, ++b)
        intersection_count += nk_u1x8_popcount_(*a & *b), union_count += nk_u1x8_popcount_(*a | *b);
    *result = (union_count != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)union_count : 0.0f;
    return nk_success_k;
}

#pragma endregion Binary Sets

#pragma region Integer Sets

NUMKONG_API nk_status_t nk_jaccard_u32_haswell(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result,
                                               void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u32_t intersection_count = 0;
    nk_size_t n_remaining = n;
    for (; n_remaining >= 4; n_remaining -= 4, a += 4, b += 4) {
        __m128i a_u32x4 = _mm_loadu_si128((__m128i const *)a);
        __m128i b_u32x4 = _mm_loadu_si128((__m128i const *)b);
        __m128i equality_u32x4 = _mm_cmpeq_epi32(a_u32x4, b_u32x4);
        int equality_mask = _mm_movemask_ps(_mm_castsi128_ps(equality_u32x4));
        intersection_count += (nk_u32_t)_mm_popcnt_u32((unsigned int)equality_mask);
    }
    for (; n_remaining; --n_remaining, ++a, ++b) intersection_count += (*a == *b);
    *result = (n != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)n : 0.0f;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hamming_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                              void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // Process 32 bytes at a time using AVX2 (256-bit registers).
    // Compare bytes for equality, invert to get not-equal mask, then count mismatches.
    //
    // Haswell port analysis:
    // - `_mm256_loadu_si256`:   p23, 1cy latency (load)
    // - `_mm256_cmpeq_epi8`:    p015, 1cy latency, 0.33cy throughput
    // - `_mm256_extracti128`:   p5, 3cy latency, 1cy throughput
    // - `_mm_popcnt_u64`:       p1 only, 3cy latency, 1cy throughput (bottleneck)
    //
    // For counting mismatches, we XOR and popcount the resulting bits set to 1.
    // Alternative: compare → movemask → popcount, but movemask only works per-byte MSBs.
    // XOR approach: each differing byte produces 0xFF (8 bits set), need to count bytes not bits.

    nk_u32_t differences = 0;
    nk_size_t n_remaining = n;

    // Main loop: process 32 bytes at a time
    for (; n_remaining >= 32; n_remaining -= 32, a += 32, b += 32) {
        __m256i a_u8x32 = _mm256_loadu_si256((__m256i const *)a);
        __m256i b_u8x32 = _mm256_loadu_si256((__m256i const *)b);

        // Compare for equality: 0xFF where equal, 0x00 where different
        __m256i equality_u8x32 = _mm256_cmpeq_epi8(a_u8x32, b_u8x32);

        // Extract to two 128-bit halves for movemask
        // movemask extracts the MSB of each byte, giving us 16 bits per 128-bit half
        __m128i equality_low_u8x16 = _mm256_castsi256_si128(equality_u8x32);
        __m128i equality_high_u8x16 = _mm256_extracti128_si256(equality_u8x32, 1);

        // Get masks: bit set = equal (0xFF MSB = 1), bit clear = different
        int mask_low = _mm_movemask_epi8(equality_low_u8x16);   // 16 bits
        int mask_high = _mm_movemask_epi8(equality_high_u8x16); // 16 bits

        // Invert to count differences (bit set = different)
        // Then popcount to count mismatches
        differences += (nk_u32_t)_mm_popcnt_u32((unsigned int)(~mask_low & 0xFFFF));
        differences += (nk_u32_t)_mm_popcnt_u32((unsigned int)(~mask_high & 0xFFFF));
    }

    // Handle remaining bytes (0-31) with scalar code
    for (; n_remaining; --n_remaining, ++a, ++b) differences += (*a != *b);

    *result = differences;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_jaccard_u16_haswell(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result,
                                               void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // Process 16 u16 values at a time using AVX2 (256-bit registers).
    // Compare 16-bit integers for equality and count matches.
    //
    // Haswell port analysis:
    // - `_mm256_loadu_si256`:   p23, 1cy latency (load)
    // - `_mm256_cmpeq_epi16`:   p015, 1cy latency, 0.33cy throughput
    // - `_mm256_packs_epi16`:   p5, 1cy latency, 1cy throughput (pack 16 → 8 bit)
    // - `_mm_movemask_epi8`:    p0, 3cy latency (extracts MSB of each byte)
    // - `_mm_popcnt_u32`:       p1 only, 3cy latency, 1cy throughput

    nk_u32_t matches = 0;
    nk_size_t n_remaining = n;

    // Main loop: process 16 u16 values at a time
    for (; n_remaining >= 16; n_remaining -= 16, a += 16, b += 16) {
        __m256i a_u16x16 = _mm256_loadu_si256((__m256i const *)a);
        __m256i b_u16x16 = _mm256_loadu_si256((__m256i const *)b);

        // Compare for equality: 0xFFFF where equal, 0x0000 where different
        __m256i equality_u16x16 = _mm256_cmpeq_epi16(a_u16x16, b_u16x16);

        // Pack 16-bit results to 8-bit to use movemask efficiently.
        // _mm256_packs_epi16 saturates signed 16-bit to signed 8-bit:
        // 0xFFFF (-1) → 0x80 (-128), 0x0000 (0) → 0x00 (0)
        // Note: packs interleaves lanes, so we need to handle the permutation.
        // For counting, we just need the total popcount, so lane order doesn't matter.
        __m256i packed_i8x32 = _mm256_packs_epi16(equality_u16x16, equality_u16x16);

        // Extract to 128-bit halves
        __m128i packed_low_i8x16 = _mm256_castsi256_si128(packed_i8x32);
        __m128i packed_high_i8x16 = _mm256_extracti128_si256(packed_i8x32, 1);

        // movemask extracts MSB of each byte
        // After packs: 0x80 (MSB=1) for equal, 0x00 (MSB=0) for different
        // Each 128-bit half has 8 relevant bytes (lower 8 from each original lane)
        int mask_low = _mm_movemask_epi8(packed_low_i8x16) & 0xFF;   // Lower 8 bytes
        int mask_high = _mm_movemask_epi8(packed_high_i8x16) & 0xFF; // Lower 8 bytes from high lane

        matches += (nk_u32_t)_mm_popcnt_u32((unsigned int)mask_low);
        matches += (nk_u32_t)_mm_popcnt_u32((unsigned int)mask_high);
    }

    // Handle remaining elements (0-15) with scalar code
    for (; n_remaining; --n_remaining, ++a, ++b) matches += (*a == *b);

    *result = (n != 0) ? 1.0f - (nk_f32_t)matches / (nk_f32_t)n : 0.0f;
    return nk_success_k;
}
#endif // NUMKONG_TARGET_HASWELL

#pragma endregion Integer Sets

#pragma region Distances from Dot Products

/** Hamming from_dot: computes pop_a + pop_b - 2*dot for 4 pairs (Haswell). */
NUMKONG_INLINE void nk_hamming_u32x4_from_dot_haswell_(nk_b128_vec_t const *dots_vec, nk_u32_t query_pop,
                                                       nk_b128_vec_t const *target_pops_vec,
                                                       nk_b128_vec_t *result_vec) {
    __m128i dots_i32x4 = dots_vec->xmm;
    __m128i query_i32x4 = _mm_set1_epi32((int)query_pop);
    __m128i target_i32x4 = target_pops_vec->xmm;
    result_vec->xmm = _mm_sub_epi32(_mm_add_epi32(query_i32x4, target_i32x4), _mm_slli_epi32(dots_i32x4, 1));
}

/** Jaccard from_dot: computes 1 - dot / (pop_a + pop_b - dot) for 4 pairs (Haswell). */
NUMKONG_INLINE void nk_jaccard_f32x4_from_dot_haswell_(nk_b128_vec_t const *dots_vec, nk_u32_t query_pop,
                                                       nk_b128_vec_t const *target_pops_vec,
                                                       nk_b128_vec_t *result_vec) {
    __m128 dot_f32x4 = _mm_cvtepi32_ps(dots_vec->xmm);
    __m128 query_f32x4 = _mm_set1_ps((nk_f32_t)query_pop);
    __m128 target_f32x4 = _mm_cvtepi32_ps(target_pops_vec->xmm);
    __m128 union_f32x4 = _mm_sub_ps(_mm_add_ps(query_f32x4, target_f32x4), dot_f32x4);

    __m128 zero_union_b32x4 = _mm_cmpeq_ps(union_f32x4, _mm_setzero_ps());
    __m128 one_f32x4 = _mm_set1_ps(1.0f);
    __m128 safe_union_f32x4 = _mm_blendv_ps(union_f32x4, one_f32x4, zero_union_b32x4);

    // Exact division rather than `_mm_rcp_ps` — batched results must match the serial reference bit-for-bit.
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

#endif // NUMKONG_ARCH_X8664_HASWELL_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_SET_HASWELL_H
