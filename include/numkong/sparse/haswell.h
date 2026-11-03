/**
 *  @file include/numkong/sparse/haswell.h
 *  @author Matt Stuchlik
 *  @date May 30, 2026
 *  @brief SIMD-accelerated sparse vector dot products for Haswell.
 *
 *  @sa c/numkong/sparse.h
 */
#ifndef NUMKONG_SPARSE_HASWELL_H
#define NUMKONG_SPARSE_HASWELL_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_HASWELL_

#include "numkong/types.h"
#include "numkong/sparse/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "popcnt")
#endif

NUMKONG_INLINE nk_size_t nk_sparse_lower_bound_gallop_u32_haswell_(nk_u32_t const *indices, nk_size_t length,
                                                                   nk_size_t position, nk_u32_t key) {
    if (position >= length || indices[position] >= key) return position;

    nk_size_t step = 1;
    while (position + step < length && indices[position + step] < key) step <<= 1;

    nk_size_t low = position + (step >> 1) + 1;
    nk_size_t high = position + step + 1;
    if (high > length) high = length;

    while (low < high) {
        nk_size_t middle = low + ((high - low) >> 1);
        if (indices[middle] < key) low = middle + 1;
        else high = middle;
    }
    return low;
}

NUMKONG_INLINE nk_f64_t nk_sparse_dot_gallop_a_u32f32_haswell_(nk_u32_t const *a, nk_u32_t const *b,
                                                               nk_f32_t const *a_weights, nk_f32_t const *b_weights,
                                                               nk_size_t a_length, nk_size_t b_length) {
    nk_f64_t sum = 0;
    nk_size_t j = 0;
    for (nk_size_t i = 0; i < a_length; ++i) {
        nk_u32_t a_index = a[i];
        j = nk_sparse_lower_bound_gallop_u32_haswell_(b, b_length, j, a_index);
        if (j == b_length) break;
        if (b[j] == a_index) sum += (nk_f64_t)a_weights[i] * (nk_f64_t)b_weights[j];
    }
    return sum;
}

NUMKONG_INLINE nk_f64_t nk_sparse_dot_gallop_b_u32f32_haswell_(nk_u32_t const *a, nk_u32_t const *b,
                                                               nk_f32_t const *a_weights, nk_f32_t const *b_weights,
                                                               nk_size_t a_length, nk_size_t b_length) {
    nk_f64_t sum = 0;
    nk_size_t i = 0;
    for (nk_size_t j = 0; j < b_length; ++j) {
        nk_u32_t b_index = b[j];
        i = nk_sparse_lower_bound_gallop_u32_haswell_(a, a_length, i, b_index);
        if (i == a_length) break;
        if (a[i] == b_index) sum += (nk_f64_t)a_weights[i] * (nk_f64_t)b_weights[j];
    }
    return sum;
}

NUMKONG_INLINE nk_f64_t nk_sparse_reduce_f64x4x2_haswell_(__m256d accumulator_low_f64x4,
                                                          __m256d accumulator_high_f64x4) {
    __m256d total_f64x4 = _mm256_add_pd(accumulator_low_f64x4, accumulator_high_f64x4);
    __m128d total_low_f64x2 = _mm256_castpd256_pd128(total_f64x4);
    __m128d total_high_f64x2 = _mm256_extractf128_pd(total_f64x4, 1);
    __m128d sum_f64x2 = _mm_add_pd(total_low_f64x2, total_high_f64x2);
    __m128d sum_f64x1 = _mm_add_sd(sum_f64x2, _mm_unpackhi_pd(sum_f64x2, sum_f64x2));
    return _mm_cvtsd_f64(sum_f64x1);
}

/** Compares every lane of @p a to all 8 lanes of @p b, which rotate through <tt>VPALIGNR</tt>. */
NUMKONG_INLINE __m128i nk_intersect_u16x8_haswell_(__m128i a, __m128i b) {
    __m128i matches_b16x8 = _mm_cmpeq_epi16(a, b);
    matches_b16x8 = _mm_or_si128(matches_b16x8, _mm_cmpeq_epi16(a, _mm_alignr_epi8(b, b, 2)));
    matches_b16x8 = _mm_or_si128(matches_b16x8, _mm_cmpeq_epi16(a, _mm_alignr_epi8(b, b, 4)));
    matches_b16x8 = _mm_or_si128(matches_b16x8, _mm_cmpeq_epi16(a, _mm_alignr_epi8(b, b, 6)));
    matches_b16x8 = _mm_or_si128(matches_b16x8, _mm_cmpeq_epi16(a, _mm_alignr_epi8(b, b, 8)));
    matches_b16x8 = _mm_or_si128(matches_b16x8, _mm_cmpeq_epi16(a, _mm_alignr_epi8(b, b, 10)));
    matches_b16x8 = _mm_or_si128(matches_b16x8, _mm_cmpeq_epi16(a, _mm_alignr_epi8(b, b, 12)));
    return _mm_or_si128(matches_b16x8, _mm_cmpeq_epi16(a, _mm_alignr_epi8(b, b, 14)));
}

/** Compares every lane of @p a to all 8 lanes of @p b, rotating inside each half, then across. */
NUMKONG_INLINE __m256i nk_intersect_u32x8_haswell_(__m256i a, __m256i b) {
    __m256i b_swapped_u32x8 = _mm256_permute2x128_si256(b, b, 1);
    __m256i matches_b32x8 = _mm256_cmpeq_epi32(a, b);
    matches_b32x8 = _mm256_or_si256(matches_b32x8, _mm256_cmpeq_epi32(a, _mm256_shuffle_epi32(b, 0x39)));
    matches_b32x8 = _mm256_or_si256(matches_b32x8, _mm256_cmpeq_epi32(a, _mm256_shuffle_epi32(b, 0x4E)));
    matches_b32x8 = _mm256_or_si256(matches_b32x8, _mm256_cmpeq_epi32(a, _mm256_shuffle_epi32(b, 0x93)));
    matches_b32x8 = _mm256_or_si256(matches_b32x8, _mm256_cmpeq_epi32(a, b_swapped_u32x8));
    matches_b32x8 = _mm256_or_si256(matches_b32x8, _mm256_cmpeq_epi32(a, _mm256_shuffle_epi32(b_swapped_u32x8, 0x39)));
    matches_b32x8 = _mm256_or_si256(matches_b32x8, _mm256_cmpeq_epi32(a, _mm256_shuffle_epi32(b_swapped_u32x8, 0x4E)));
    return _mm256_or_si256(matches_b32x8, _mm256_cmpeq_epi32(a, _mm256_shuffle_epi32(b_swapped_u32x8, 0x93)));
}

/** Compares every lane of @p a to all 4 lanes of @p b, swapping inside each half, then across. */
NUMKONG_INLINE __m256i nk_intersect_u64x4_haswell_(__m256i a, __m256i b) {
    __m256i b_swapped_u64x4 = _mm256_permute2x128_si256(b, b, 1);
    __m256i matches_b64x4 = _mm256_or_si256(_mm256_cmpeq_epi64(a, b),
                                            _mm256_cmpeq_epi64(a, _mm256_shuffle_epi32(b, 0x4E)));
    matches_b64x4 = _mm256_or_si256(matches_b64x4, _mm256_cmpeq_epi64(a, b_swapped_u64x4));
    return _mm256_or_si256(matches_b64x4, _mm256_cmpeq_epi64(a, _mm256_shuffle_epi32(b_swapped_u64x4, 0x4E)));
}

#if NUMKONG_TARGET_HASWELL

NUMKONG_API nk_status_t nk_sparse_intersect_u16_haswell(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                                        nk_size_t b_length, nk_u16_t *result, nk_size_t *count,
                                                        nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // Galloping through the longer array beats any block walk
    if (a_length / 64 >= b_length || b_length / 64 >= a_length) {
        nk_sparse_intersect_u16_serial_(a, b, a_length, b_length, result, count);
        return nk_success_k;
    }
    nk_u16_t const *const a_end = a + a_length;
    nk_u16_t const *const b_end = b + b_length;
    nk_size_t matches = 0;
    while (a + 8 <= a_end && b + 8 <= b_end) {
        __m128i a_u16x8 = _mm_loadu_si128((__m128i const *)a);
        __m128i b_u16x8 = _mm_loadu_si128((__m128i const *)b);
        __m128i a_max_u16x8 = _mm_set1_epi16((short)a[7]), b_max_u16x8 = _mm_set1_epi16((short)b[7]);
        __m128i matches_i8x16 = _mm_packs_epi16(nk_intersect_u16x8_haswell_(a_u16x8, b_u16x8), _mm_setzero_si128());
        unsigned lane_bits = (unsigned)_mm_movemask_epi8(matches_i8x16);
        // The loop ends on the last match, so it only touches slots the matches fill
        if (result) {
            for (nk_size_t lane = 0; lane_bits; ++lane, lane_bits >>= 1)
                result[matches] = a[lane], matches += lane_bits & 1;
        }
        else matches += (nk_size_t)_mm_popcnt_u32(lane_bits);
        __m128i a_inrange_b16x8 = _mm_cmpeq_epi16(_mm_min_epu16(a_u16x8, b_max_u16x8), a_u16x8);
        __m128i b_inrange_b16x8 = _mm_cmpeq_epi16(_mm_min_epu16(b_u16x8, a_max_u16x8), b_u16x8);
        unsigned inrange_bits = (unsigned)_mm_movemask_epi8(_mm_packs_epi16(a_inrange_b16x8, b_inrange_b16x8));
        a += _tzcnt_u32((~inrange_bits & 0xFF) | 0x100), b += _tzcnt_u32(((~inrange_bits >> 8) & 0xFF) | 0x100);
    }
    nk_size_t tail_count = 0;
    nk_sparse_intersect_u16_serial_(a, b, a_end - a, b_end - b, result ? result + matches : 0, &tail_count);
    *count = matches + tail_count;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sparse_intersect_u32_haswell(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                                        nk_size_t b_length, nk_u32_t *result, nk_size_t *count,
                                                        nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // Galloping through the longer array beats any block walk
    if (a_length / 64 >= b_length || b_length / 64 >= a_length) {
        nk_sparse_intersect_u32_serial_(a, b, a_length, b_length, result, count);
        return nk_success_k;
    }
    nk_u32_t const *const a_end = a + a_length;
    nk_u32_t const *const b_end = b + b_length;
    nk_size_t matches = 0;
    while (a + 8 <= a_end && b + 8 <= b_end) {
        __m256i a_u32x8 = _mm256_loadu_si256((__m256i const *)a);
        __m256i b_u32x8 = _mm256_loadu_si256((__m256i const *)b);
        __m256i a_max_u32x8 = _mm256_set1_epi32((int)a[7]), b_max_u32x8 = _mm256_set1_epi32((int)b[7]);
        __m256i matches_b32x8 = nk_intersect_u32x8_haswell_(a_u32x8, b_u32x8);
        unsigned lane_bits = (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(matches_b32x8));
        // The loop ends on the last match, so it only touches slots the matches fill
        if (result) {
            for (nk_size_t lane = 0; lane_bits; ++lane, lane_bits >>= 1)
                result[matches] = a[lane], matches += lane_bits & 1;
        }
        else matches += (nk_size_t)_mm_popcnt_u32(lane_bits);
        __m256i a_inrange_b32x8 = _mm256_cmpeq_epi32(_mm256_min_epu32(a_u32x8, b_max_u32x8), a_u32x8);
        __m256i b_inrange_b32x8 = _mm256_cmpeq_epi32(_mm256_min_epu32(b_u32x8, a_max_u32x8), b_u32x8);
        a += _tzcnt_u32(~(unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(a_inrange_b32x8)));
        b += _tzcnt_u32(~(unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(b_inrange_b32x8)));
    }
    nk_size_t tail_count = 0;
    nk_sparse_intersect_u32_serial_(a, b, a_end - a, b_end - b, result ? result + matches : 0, &tail_count);
    *count = matches + tail_count;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sparse_intersect_u64_haswell(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                                        nk_size_t b_length, nk_u64_t *result, nk_size_t *count,
                                                        nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // Galloping through the longer array beats any block walk
    if (a_length / 64 >= b_length || b_length / 64 >= a_length) {
        nk_sparse_intersect_u64_serial_(a, b, a_length, b_length, result, count);
        return nk_success_k;
    }
    nk_u64_t const *const a_end = a + a_length;
    nk_u64_t const *const b_end = b + b_length;
    nk_size_t matches = 0;
    __m256i const sign_bit_u64x4 = _mm256_set1_epi64x((long long)0x8000000000000000ull);
    while (a + 4 <= a_end && b + 4 <= b_end) {
        __m256i a_u64x4 = _mm256_loadu_si256((__m256i const *)a);
        __m256i b_u64x4 = _mm256_loadu_si256((__m256i const *)b);
        __m256i a_max_i64x4 = _mm256_set1_epi64x((long long)(a[3] ^ 0x8000000000000000ull));
        __m256i b_max_i64x4 = _mm256_set1_epi64x((long long)(b[3] ^ 0x8000000000000000ull));
        __m256i matches_b64x4 = nk_intersect_u64x4_haswell_(a_u64x4, b_u64x4);
        unsigned lane_bits = (unsigned)_mm256_movemask_pd(_mm256_castsi256_pd(matches_b64x4));
        // The loop ends on the last match, so it only touches slots the matches fill
        if (result) {
            for (nk_size_t lane = 0; lane_bits; ++lane, lane_bits >>= 1)
                result[matches] = a[lane], matches += lane_bits & 1;
        }
        else matches += (nk_size_t)_mm_popcnt_u32(lane_bits);
        // AVX2 compares 64-bit lanes only as signed, and sorted lanes stay past the other maximum
        __m256i a_beyond_b64x4 = _mm256_cmpgt_epi64(_mm256_xor_si256(a_u64x4, sign_bit_u64x4), b_max_i64x4);
        __m256i b_beyond_b64x4 = _mm256_cmpgt_epi64(_mm256_xor_si256(b_u64x4, sign_bit_u64x4), a_max_i64x4);
        a += _tzcnt_u32((unsigned)_mm256_movemask_pd(_mm256_castsi256_pd(a_beyond_b64x4)) | 0x10);
        b += _tzcnt_u32((unsigned)_mm256_movemask_pd(_mm256_castsi256_pd(b_beyond_b64x4)) | 0x10);
    }
    nk_size_t tail_count = 0;
    nk_sparse_intersect_u64_serial_(a, b, a_end - a, b_end - b, result ? result + matches : 0, &tail_count);
    *count = matches + tail_count;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sparse_dot_u32f32_haswell(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                                     nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                     nk_f64_t *product, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if ((a_length << 6) < b_length) {
        *product = nk_sparse_dot_gallop_a_u32f32_haswell_(a, b, a_weights, b_weights, a_length, b_length);
        return nk_success_k;
    }
    if ((b_length << 6) < a_length) {
        *product = nk_sparse_dot_gallop_b_u32f32_haswell_(a, b, a_weights, b_weights, a_length, b_length);
        return nk_success_k;
    }

    nk_size_t i = 0;
    nk_size_t j = 0;
    __m256d accumulator_low_f64x4 = _mm256_setzero_pd();
    __m256d accumulator_high_f64x4 = _mm256_setzero_pd();

    if (a_length >= 16 && b_length >= 8) {
        nk_size_t a_last_block_start = a_length - 16;
        nk_size_t b_last_block_start = b_length - 8;

        while (i <= a_last_block_start && j <= b_last_block_start) {
            // Compare a block of 16 A indices against 8 B indices. Since AVX2 lacks unsigned
            // i32 comparisons, bias both sides by the sign bit and use signed comparisons below.
            __m256i a_indices_low_u32x8 = _mm256_loadu_si256((__m256i const *)(a + i));
            __m256i a_indices_high_u32x8 = _mm256_loadu_si256((__m256i const *)(a + i + 8));
            __m256i b_indices_u32x8 = _mm256_loadu_si256((__m256i const *)(b + j));
            __m256 b_weights_f32x8 = _mm256_loadu_ps(b_weights + j);
            __m256i zero_i32x8 = _mm256_setzero_si256();
            __m256i sign_bit_u32x8 = _mm256_set1_epi32((int)0x80000000u);
            __m256i a_indices_low_biased_i32x8 = _mm256_xor_si256(a_indices_low_u32x8, sign_bit_u32x8);
            __m256i a_indices_high_biased_i32x8 = _mm256_xor_si256(a_indices_high_u32x8, sign_bit_u32x8);

            _mm_prefetch((char const *)(b + j + 512), _MM_HINT_T0);
            _mm_prefetch((char const *)(a + i + 512), _MM_HINT_T0);

            // Build the rank of each A index within the B block. Ranks are split into even
            // and odd accumulators to reduce dependency-chain pressure in this hot loop.
            __m256i b_index_0_biased_i32x8 = _mm256_set1_epi32((int)(b[j + 0] ^ (nk_u32_t)0x80000000u));
            __m256i rank_low_even_i32x8 = _mm256_sub_epi32(
                zero_i32x8, _mm256_cmpgt_epi32(a_indices_low_biased_i32x8, b_index_0_biased_i32x8));
            __m256i rank_high_even_i32x8 = _mm256_sub_epi32(
                zero_i32x8, _mm256_cmpgt_epi32(a_indices_high_biased_i32x8, b_index_0_biased_i32x8));

            __m256i b_index_1_biased_i32x8 = _mm256_set1_epi32((int)(b[j + 1] ^ (nk_u32_t)0x80000000u));
            __m256i rank_low_odd_i32x8 = _mm256_sub_epi32(
                zero_i32x8, _mm256_cmpgt_epi32(a_indices_low_biased_i32x8, b_index_1_biased_i32x8));
            __m256i rank_high_odd_i32x8 = _mm256_sub_epi32(
                zero_i32x8, _mm256_cmpgt_epi32(a_indices_high_biased_i32x8, b_index_1_biased_i32x8));

            __m256i b_index_2_biased_i32x8 = _mm256_set1_epi32((int)(b[j + 2] ^ (nk_u32_t)0x80000000u));
            rank_low_even_i32x8 = _mm256_sub_epi32(
                rank_low_even_i32x8, _mm256_cmpgt_epi32(a_indices_low_biased_i32x8, b_index_2_biased_i32x8));
            rank_high_even_i32x8 = _mm256_sub_epi32(
                rank_high_even_i32x8, _mm256_cmpgt_epi32(a_indices_high_biased_i32x8, b_index_2_biased_i32x8));

            __m256i b_index_3_biased_i32x8 = _mm256_set1_epi32((int)(b[j + 3] ^ (nk_u32_t)0x80000000u));
            rank_low_odd_i32x8 = _mm256_sub_epi32(
                rank_low_odd_i32x8, _mm256_cmpgt_epi32(a_indices_low_biased_i32x8, b_index_3_biased_i32x8));
            rank_high_odd_i32x8 = _mm256_sub_epi32(
                rank_high_odd_i32x8, _mm256_cmpgt_epi32(a_indices_high_biased_i32x8, b_index_3_biased_i32x8));

            __m256i b_index_4_biased_i32x8 = _mm256_set1_epi32((int)(b[j + 4] ^ (nk_u32_t)0x80000000u));
            rank_low_even_i32x8 = _mm256_sub_epi32(
                rank_low_even_i32x8, _mm256_cmpgt_epi32(a_indices_low_biased_i32x8, b_index_4_biased_i32x8));
            rank_high_even_i32x8 = _mm256_sub_epi32(
                rank_high_even_i32x8, _mm256_cmpgt_epi32(a_indices_high_biased_i32x8, b_index_4_biased_i32x8));

            __m256i b_index_5_biased_i32x8 = _mm256_set1_epi32((int)(b[j + 5] ^ (nk_u32_t)0x80000000u));
            rank_low_odd_i32x8 = _mm256_sub_epi32(
                rank_low_odd_i32x8, _mm256_cmpgt_epi32(a_indices_low_biased_i32x8, b_index_5_biased_i32x8));
            rank_high_odd_i32x8 = _mm256_sub_epi32(
                rank_high_odd_i32x8, _mm256_cmpgt_epi32(a_indices_high_biased_i32x8, b_index_5_biased_i32x8));

            __m256i b_index_6_biased_i32x8 = _mm256_set1_epi32((int)(b[j + 6] ^ (nk_u32_t)0x80000000u));
            rank_low_even_i32x8 = _mm256_sub_epi32(
                rank_low_even_i32x8, _mm256_cmpgt_epi32(a_indices_low_biased_i32x8, b_index_6_biased_i32x8));
            rank_high_even_i32x8 = _mm256_sub_epi32(
                rank_high_even_i32x8, _mm256_cmpgt_epi32(a_indices_high_biased_i32x8, b_index_6_biased_i32x8));

            __m256i rank_low_i32x8 = _mm256_add_epi32(rank_low_even_i32x8, rank_low_odd_i32x8);
            __m256i rank_high_i32x8 = _mm256_add_epi32(rank_high_even_i32x8, rank_high_odd_i32x8);

            // Use the ranks to align each A lane with its candidate B lane, then mask both
            // weights so non-matching NaN/Inf values cannot poison the floating-point sum.
            __m256i b_indices_permuted_high_u32x8 = _mm256_permutevar8x32_epi32(b_indices_u32x8, rank_high_i32x8);
            __m256i matches_high_b32x8 = _mm256_cmpeq_epi32(b_indices_permuted_high_u32x8, a_indices_high_u32x8);
            __m256 b_weights_permuted_high_f32x8 = _mm256_permutevar8x32_ps(b_weights_f32x8, rank_high_i32x8);
            __m256 a_weights_matched_high_f32x8 = _mm256_and_ps(_mm256_castsi256_ps(matches_high_b32x8),
                                                                _mm256_loadu_ps(a_weights + i + 8));
            __m256 b_weights_matched_high_f32x8 = _mm256_and_ps(_mm256_castsi256_ps(matches_high_b32x8),
                                                                b_weights_permuted_high_f32x8);

            __m256i b_indices_permuted_low_u32x8 = _mm256_permutevar8x32_epi32(b_indices_u32x8, rank_low_i32x8);
            __m256i matches_low_b32x8 = _mm256_cmpeq_epi32(b_indices_permuted_low_u32x8, a_indices_low_u32x8);
            __m256 b_weights_permuted_low_f32x8 = _mm256_permutevar8x32_ps(b_weights_f32x8, rank_low_i32x8);
            __m256 a_weights_matched_low_f32x8 = _mm256_and_ps(_mm256_castsi256_ps(matches_low_b32x8),
                                                               _mm256_loadu_ps(a_weights + i));
            __m256 b_weights_matched_low_f32x8 = _mm256_and_ps(_mm256_castsi256_ps(matches_low_b32x8),
                                                               b_weights_permuted_low_f32x8);

            // Widen matched f32 products to f64 before accumulation. The low/high accumulators
            // refer to the two f64 halves produced by converting each f32x8 vector.
            accumulator_low_f64x4 = _mm256_fmadd_pd(
                _mm256_cvtps_pd(_mm256_castps256_ps128(a_weights_matched_low_f32x8)),
                _mm256_cvtps_pd(_mm256_castps256_ps128(b_weights_matched_low_f32x8)), accumulator_low_f64x4);
            accumulator_high_f64x4 = _mm256_fmadd_pd(
                _mm256_cvtps_pd(_mm256_extractf128_ps(a_weights_matched_low_f32x8, 1)),
                _mm256_cvtps_pd(_mm256_extractf128_ps(b_weights_matched_low_f32x8, 1)), accumulator_high_f64x4);
            accumulator_low_f64x4 = _mm256_fmadd_pd(
                _mm256_cvtps_pd(_mm256_castps256_ps128(a_weights_matched_high_f32x8)),
                _mm256_cvtps_pd(_mm256_castps256_ps128(b_weights_matched_high_f32x8)), accumulator_low_f64x4);
            accumulator_high_f64x4 = _mm256_fmadd_pd(
                _mm256_cvtps_pd(_mm256_extractf128_ps(a_weights_matched_high_f32x8, 1)),
                _mm256_cvtps_pd(_mm256_extractf128_ps(b_weights_matched_high_f32x8, 1)), accumulator_high_f64x4);

            // Advance the side whose current block cannot overlap the other side's current block.
            // If both can still overlap, advance A by half a block to reuse the same B block.
            nk_size_t a_position_next = i + 8;
            nk_u32_t b_index_last = b[j + 7];
            if (a[i + 7] > b_index_last) a_position_next = i;
            if (a[i + 15] <= b_index_last) a_position_next = i + 16;

            nk_size_t b_position_next = j + 8;
            if (a[i + 15] < b_index_last) b_position_next = j;

            i = a_position_next;
            j = b_position_next;
        }
    }

    nk_f64_t vector_sum = nk_sparse_reduce_f64x4x2_haswell_(accumulator_low_f64x4, accumulator_high_f64x4);
    nk_f64_t tail_product = 0;
    nk_sparse_dot_u32f32_serial_(a + i, b + j, a_weights + i, b_weights + j, a_length - i, b_length - j, &tail_product);
    *product = vector_sum + tail_product;
    return nk_success_k;
}

#endif // NUMKONG_TARGET_HASWELL

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
#endif // NUMKONG_SPARSE_HASWELL_H
