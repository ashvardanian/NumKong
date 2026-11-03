/**
 *  @file include/numkong/set/haswell.h
 *  @author Ash Vardanian
 *  @date March 3, 2024
 *  @brief SIMD-accelerated set similarity measures for Haswell.
 *
 *  @sa c/numkong/set.h
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

/** Adds the 4 U64 lanes, as this family's target lacks what the reduce helpers require. */
NUMKONG_INLINE nk_u64_t nk_set_sum_u64x4_haswell_(__m256i sum_u64x4) {
    __m128i sum_u64x2 = _mm_add_epi64(_mm256_castsi256_si128(sum_u64x4), _mm256_extracti128_si256(sum_u64x4, 1));
    return (nk_u64_t)_mm_cvtsi128_si64(_mm_add_epi64(sum_u64x2, _mm_unpackhi_epi64(sum_u64x2, sum_u64x2)));
}

/** Adds the 8 U32 lanes. */
NUMKONG_INLINE nk_u32_t nk_set_sum_u32x8_haswell_(__m256i sum_u32x8) {
    __m128i sum_u32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_u32x8), _mm256_extracti128_si256(sum_u32x8, 1));
    sum_u32x4 = _mm_add_epi32(sum_u32x4, _mm_unpackhi_epi64(sum_u32x4, sum_u32x4));
    return (nk_u32_t)_mm_cvtsi128_si32(sum_u32x4) + (nk_u32_t)_mm_extract_epi32(sum_u32x4, 1);
}

/** Sums the popcounts of each 8 bytes of @p bytes_u8x32 , through nibble lookups, in U64 lanes. */
NUMKONG_INLINE __m256i nk_set_popcount_u64x4_haswell_(__m256i bytes_u8x32) {
    __m256i lut_i8x32 = _mm256_setr_epi8( //
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    __m256i mask_0f_i8x32 = _mm256_set1_epi8(0x0F);
    __m256i low_nibble_u8x32 = _mm256_and_si256(bytes_u8x32, mask_0f_i8x32);
    __m256i high_nibble_u8x32 = _mm256_and_si256(_mm256_srli_epi16(bytes_u8x32, 4), mask_0f_i8x32);
    return _mm256_sad_epu8(_mm256_add_epi8(_mm256_shuffle_epi8(lut_i8x32, low_nibble_u8x32),
                                           _mm256_shuffle_epi8(lut_i8x32, high_nibble_u8x32)),
                           _mm256_setzero_si256());
}

#if NUMKONG_TARGET_HASWELL

#pragma region Binary Sets

NUMKONG_API nk_status_t nk_hamming_u1_haswell(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t n_bytes = n / NUMKONG_BITS_PER_BYTE;
    nk_u32_t differences = 0;
    if (n_bytes >= 32) {
        __m256i differences_u64x4 = _mm256_setzero_si256();
        for (; n_bytes >= 32; n_bytes -= 32, a += 32, b += 32) {
            __m256i a_u8x32 = _mm256_loadu_si256((__m256i const *)a);
            __m256i b_u8x32 = _mm256_loadu_si256((__m256i const *)b);
            differences_u64x4 = _mm256_add_epi64(differences_u64x4,
                                                 nk_set_popcount_u64x4_haswell_(_mm256_xor_si256(a_u8x32, b_u8x32)));
        }
        differences = (nk_u32_t)nk_set_sum_u64x4_haswell_(differences_u64x4);
    }
    // x86 supports unaligned loads and works just fine with the scalar version for small vectors.
    for (; n_bytes >= 8; n_bytes -= 8, a += 8, b += 8)
        differences += _mm_popcnt_u64(*(nk_u64_t const *)a ^ *(nk_u64_t const *)b);
    for (; n_bytes; --n_bytes, ++a, ++b) differences += _mm_popcnt_u32(*a ^ *b);
    *result = differences;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_jaccard_u1_haswell(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t n_bytes = n / NUMKONG_BITS_PER_BYTE;
    nk_u32_t intersection_count = 0, union_count = 0;
    if (n_bytes >= 32) {
        __m256i intersection_u64x4 = _mm256_setzero_si256(), union_u64x4 = _mm256_setzero_si256();
        for (; n_bytes >= 32; n_bytes -= 32, a += 32, b += 32) {
            __m256i a_u8x32 = _mm256_loadu_si256((__m256i const *)a);
            __m256i b_u8x32 = _mm256_loadu_si256((__m256i const *)b);
            intersection_u64x4 = _mm256_add_epi64(intersection_u64x4,
                                                  nk_set_popcount_u64x4_haswell_(_mm256_and_si256(a_u8x32, b_u8x32)));
            union_u64x4 = _mm256_add_epi64(union_u64x4,
                                           nk_set_popcount_u64x4_haswell_(_mm256_or_si256(a_u8x32, b_u8x32)));
        }
        intersection_count = (nk_u32_t)nk_set_sum_u64x4_haswell_(intersection_u64x4);
        union_count = (nk_u32_t)nk_set_sum_u64x4_haswell_(union_u64x4);
    }
    // x86 supports unaligned loads and works just fine with the scalar version for small vectors.
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
                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t n_remaining = n;
    __m256i intersection_u32x8 = _mm256_setzero_si256();
    for (; n_remaining >= 8; n_remaining -= 8, a += 8, b += 8) {
        __m256i a_u32x8 = _mm256_loadu_si256((__m256i const *)a);
        __m256i b_u32x8 = _mm256_loadu_si256((__m256i const *)b);
        intersection_u32x8 = _mm256_sub_epi32(intersection_u32x8, _mm256_cmpeq_epi32(a_u32x8, b_u32x8));
    }
    nk_u32_t intersection_count = nk_set_sum_u32x8_haswell_(intersection_u32x8);
    for (; n_remaining; --n_remaining, ++a, ++b) intersection_count += (*a == *b);
    *result = (n != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)n : 0.0f;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hamming_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t n_remaining = n;
    __m256i differences_u64x4 = _mm256_setzero_si256();
    __m256i one_u8x32 = _mm256_set1_epi8(1), zero_u8x32 = _mm256_setzero_si256();

    // Equal bytes turn 0xFF and wrap to 0 on the increment, and `VPSADBW` cannot overflow U64 lanes
    for (; n_remaining >= 32; n_remaining -= 32, a += 32, b += 32) {
        __m256i a_u8x32 = _mm256_loadu_si256((__m256i const *)a);
        __m256i b_u8x32 = _mm256_loadu_si256((__m256i const *)b);
        __m256i different_u8x32 = _mm256_add_epi8(_mm256_cmpeq_epi8(a_u8x32, b_u8x32), one_u8x32);
        differences_u64x4 = _mm256_add_epi64(differences_u64x4, _mm256_sad_epu8(different_u8x32, zero_u8x32));
    }
    nk_u32_t differences = (nk_u32_t)nk_set_sum_u64x4_haswell_(differences_u64x4);

    // Handle remaining bytes (0-31) with scalar code
    for (; n_remaining; --n_remaining, ++a, ++b) differences += (*a != *b);

    *result = differences;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_jaccard_u16_haswell(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t n_remaining = n;
    __m256i matches_u32x8 = _mm256_setzero_si256();
    __m256i one_i16x16 = _mm256_set1_epi16(1);

    // `VPMADDWD` folds pairs of -1 masks into I32 lanes, which cannot overflow as U16 lanes would
    for (; n_remaining >= 16; n_remaining -= 16, a += 16, b += 16) {
        __m256i a_u16x16 = _mm256_loadu_si256((__m256i const *)a);
        __m256i b_u16x16 = _mm256_loadu_si256((__m256i const *)b);
        __m256i equality_u16x16 = _mm256_cmpeq_epi16(a_u16x16, b_u16x16);
        matches_u32x8 = _mm256_sub_epi32(matches_u32x8, _mm256_madd_epi16(equality_u16x16, one_i16x16));
    }
    nk_u32_t matches = nk_set_sum_u32x8_haswell_(matches_u32x8);

    // Handle remaining elements (0-15) with scalar code
    for (; n_remaining; --n_remaining, ++a, ++b) matches += (*a == *b);

    *result = (n != 0) ? 1.0f - (nk_f32_t)matches / (nk_f32_t)n : 0.0f;
    return nk_success_k;
}

#pragma endregion Integer Sets

#endif // NUMKONG_TARGET_HASWELL

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
