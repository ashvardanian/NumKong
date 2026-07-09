/**
 *  @brief SIMD-accelerated Sparse Vector Dot Products for Haswell.
 *  @file include/numkong/sparse/haswell.h
 *  @author Matt Stuchlik
 *  @date May 30, 2026
 *
 *  @sa include/numkong/sparse.h
 */
#ifndef NK_SPARSE_HASWELL_H
#define NK_SPARSE_HASWELL_H

#if NK_TARGET_X8664_
#if NK_TARGET_HASWELL

#include "numkong/types.h"
#include "numkong/sparse/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

NK_INTERNAL nk_size_t nk_sparse_lower_bound_gallop_u32_haswell_(nk_u32_t const *indices, nk_size_t length,
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

NK_INTERNAL nk_f64_t nk_sparse_dot_gallop_a_u32f32_haswell_(nk_u32_t const *a, nk_u32_t const *b,
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

NK_INTERNAL nk_f64_t nk_sparse_dot_gallop_b_u32f32_haswell_(nk_u32_t const *a, nk_u32_t const *b,
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

NK_INTERNAL nk_f64_t nk_sparse_reduce_f64x4x2_haswell_(__m256d accumulator_low_f64x4, __m256d accumulator_high_f64x4) {
    __m256d total_f64x4 = _mm256_add_pd(accumulator_low_f64x4, accumulator_high_f64x4);
    __m128d total_low_f64x2 = _mm256_castpd256_pd128(total_f64x4);
    __m128d total_high_f64x2 = _mm256_extractf128_pd(total_f64x4, 1);
    __m128d sum_f64x2 = _mm_add_pd(total_low_f64x2, total_high_f64x2);
    __m128d sum_f64x1 = _mm_add_sd(sum_f64x2, _mm_unpackhi_pd(sum_f64x2, sum_f64x2));
    return _mm_cvtsd_f64(sum_f64x1);
}

NK_PUBLIC void nk_sparse_dot_u32f32_haswell(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                            nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                            nk_f64_t *product) {
    if ((a_length << 6) < b_length) {
        *product = nk_sparse_dot_gallop_a_u32f32_haswell_(a, b, a_weights, b_weights, a_length, b_length);
        return;
    }
    if ((b_length << 6) < a_length) {
        *product = nk_sparse_dot_gallop_b_u32f32_haswell_(a, b, a_weights, b_weights, a_length, b_length);
        return;
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
    nk_sparse_dot_u32f32_serial(a + i, b + j, a_weights + i, b_weights + j, a_length - i, b_length - j, &tail_product);
    *product = vector_sum + tail_product;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X8664_
#endif // NK_SPARSE_HASWELL_H
