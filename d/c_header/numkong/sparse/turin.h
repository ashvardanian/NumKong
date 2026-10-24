/**
 *  @brief Turin-accelerated Sparse Vector Operations.
 *  @file include/numkong/sparse/turin.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/sparse.h
 */
#ifndef NK_SPARSE_TURIN_H
#define NK_SPARSE_TURIN_H

#if NK_TARGET_X86_
#if NK_TARGET_TURIN

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                                    \
    __attribute__((target(                                                                                                       \
        "avx2,avx512f,avx512vl,bmi,bmi2,lzcnt,popcnt,avx512bw,avx512vbmi2,avx512bf16,avx512vnni,avx512vp2intersect,avx512dq"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi", "bmi2", "lzcnt", "popcnt", "avx512bw", "avx512vbmi2", \
                   "avx512bf16", "avx512vnni", "avx512vp2intersect", "avx512dq")
#endif

NK_PUBLIC void nk_sparse_intersect_u16_turin( //
    nk_u16_t const *a, nk_u16_t const *b,     //
    nk_size_t a_length, nk_size_t b_length,   //
    nk_u16_t *result, nk_size_t *count) {

    //! There is no such thing as `_mm512_2intersect_epi16`, only the 32-bit variant!
    //! So instead of jumping through 32 entries at a time, like on Ice Lake, we will
    //! step through 16 entries at a time.
    nk_u16_t const *const a_end = a + a_length;
    nk_u16_t const *const b_end = b + b_length;
    nk_size_t c = 0;
    nk_b256_vec_t a_vec, b_vec;

    // Broadcast index for last element (hoisted outside loop)
    __m256i const last_idx = _mm256_set1_epi16(15);
    while (a + 16 <= a_end && b + 16 <= b_end) {
        a_vec.ymm = _mm256_loadu_si256((__m256i const *)a);
        b_vec.ymm = _mm256_loadu_si256((__m256i const *)b);

        // Intersect the registers
        __m512i a_i32x16 = _mm512_cvtepu16_epi32(a_vec.ymm);
        __m512i b_i32x16 = _mm512_cvtepu16_epi32(b_vec.ymm);
        __mmask16 a_matches_any_in_b, b_matches_any_in_a;
        _mm512_2intersect_epi32(a_i32x16, b_i32x16, &a_matches_any_in_b, &b_matches_any_in_a);

        // Export matches if result buffer is provided
        if (result) { _mm256_mask_compressstoreu_epi16(result + c, a_matches_any_in_b, a_vec.ymm); }
        c += _mm_popcnt_u32(a_matches_any_in_b); // MSVC has no `_popcnt32`

        __m256i a_max_u16x16 = _mm256_permutexvar_epi16(last_idx, a_vec.ymm);
        __m256i b_max_u16x16 = _mm256_permutexvar_epi16(last_idx, b_vec.ymm);
        __mmask16 a_step_mask = _mm256_cmple_epu16_mask(a_vec.ymm, b_max_u16x16);
        __mmask16 b_step_mask = _mm256_cmple_epu16_mask(b_vec.ymm, a_max_u16x16);
        a += _tzcnt_u32(~(nk_u32_t)a_step_mask | 0x10000);
        b += _tzcnt_u32(~(nk_u32_t)b_step_mask | 0x10000);
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u16_serial(a, b, a_end - a, b_end - b, result ? result + c : 0, &tail_count);
    *count = c + tail_count;
}

NK_PUBLIC void nk_sparse_intersect_u32_turin( //
    nk_u32_t const *a, nk_u32_t const *b,     //
    nk_size_t a_length, nk_size_t b_length,   //
    nk_u32_t *result, nk_size_t *count) {

    nk_u32_t const *const a_end = a + a_length;
    nk_u32_t const *const b_end = b + b_length;
    nk_size_t c = 0;
    nk_b512_vec_t a_vec, b_vec;

    // Broadcast index for last element (hoisted outside loop)
    __m512i const last_idx = _mm512_set1_epi32(15);
    while (a + 16 <= a_end && b + 16 <= b_end) {
        a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
        b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);

        // Intersect the registers
        __mmask16 a_matches_any_in_b, b_matches_any_in_a;
        _mm512_2intersect_epi32(a_vec.zmm, b_vec.zmm, &a_matches_any_in_b, &b_matches_any_in_a);

        // Export matches if result buffer is provided
        if (result) { _mm512_mask_compressstoreu_epi32(result + c, a_matches_any_in_b, a_vec.zmm); }
        c += _mm_popcnt_u32(a_matches_any_in_b); // MSVC has no `_popcnt32`

        // Pure SIMD broadcasts - no scalar extraction needed
        __m512i a_max_u32x16 = _mm512_permutexvar_epi32(last_idx, a_vec.zmm);
        __m512i b_max_u32x16 = _mm512_permutexvar_epi32(last_idx, b_vec.zmm);
        __mmask16 a_step_mask = _mm512_cmple_epu32_mask(a_vec.zmm, b_max_u32x16);
        __mmask16 b_step_mask = _mm512_cmple_epu32_mask(b_vec.zmm, a_max_u32x16);
        a += _tzcnt_u32(~(nk_u32_t)a_step_mask | 0x10000);
        b += _tzcnt_u32(~(nk_u32_t)b_step_mask | 0x10000);
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u32_serial(a, b, a_end - a, b_end - b, result ? result + c : 0, &tail_count);
    *count = c + tail_count;
}

NK_PUBLIC void nk_sparse_intersect_u64_turin( //
    nk_u64_t const *a, nk_u64_t const *b,     //
    nk_size_t a_length, nk_size_t b_length,   //
    nk_u64_t *result, nk_size_t *count) {

    nk_u64_t const *const a_end = a + a_length;
    nk_u64_t const *const b_end = b + b_length;
    nk_size_t c = 0;
    nk_b512_vec_t a_vec, b_vec;

    // Broadcast index for last element (hoisted outside loop)
    __m512i const last_idx = _mm512_set1_epi64(7);
    while (a + 8 <= a_end && b + 8 <= b_end) {
        a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
        b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);

        // Intersect the registers
        __mmask8 a_matches_any_in_b, b_matches_any_in_a;
        _mm512_2intersect_epi64(a_vec.zmm, b_vec.zmm, &a_matches_any_in_b, &b_matches_any_in_a);

        // Export matches if result buffer is provided
        if (result) { _mm512_mask_compressstoreu_epi64(result + c, a_matches_any_in_b, a_vec.zmm); }
        c += _mm_popcnt_u32(a_matches_any_in_b); // MSVC has no `_popcnt32`

        // Pure SIMD broadcasts - no scalar extraction needed
        __m512i a_max_u64x8 = _mm512_permutexvar_epi64(last_idx, a_vec.zmm);
        __m512i b_max_u64x8 = _mm512_permutexvar_epi64(last_idx, b_vec.zmm);
        __mmask8 a_step_mask = _mm512_cmple_epu64_mask(a_vec.zmm, b_max_u64x8);
        __mmask8 b_step_mask = _mm512_cmple_epu64_mask(b_vec.zmm, a_max_u64x8);
        a += _tzcnt_u32(~(nk_u32_t)a_step_mask | 0x100);
        b += _tzcnt_u32(~(nk_u32_t)b_step_mask | 0x100);
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u64_serial(a, b, a_end - a, b_end - b, result ? result + c : 0, &tail_count);
    *count = c + tail_count;
}

NK_PUBLIC void nk_sparse_dot_u16bf16_turin(                 //
    nk_u16_t const *a, nk_u16_t const *b,                   //
    nk_bf16_t const *a_weights, nk_bf16_t const *b_weights, //
    nk_size_t a_length, nk_size_t b_length,                 //
    nk_f32_t *product) {

#if NK_ALLOW_ISA_REDIRECT
    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 64 && b_length < 64) {
        nk_sparse_dot_u16bf16_serial(a, b, a_weights, b_weights, a_length, b_length, product);
        return;
    }
#endif

    //! There is no such thing as `_mm512_2intersect_epi16`, only the 32-bit variant!
    //! So instead of jumping through 32 entries at a time, like on Ice Lake, we will
    //! step through 16 entries at a time.
    nk_u16_t const *const a_end = a + a_length;
    nk_u16_t const *const b_end = b + b_length;
    nk_b256_vec_t a_vec, b_vec;
    __m256 product_f32x8 = _mm256_setzero_ps();

    // Broadcast index for last element (hoisted outside loop)
    __m256i const last_idx = _mm256_set1_epi16(15);
    while (a + 16 <= a_end && b + 16 <= b_end) {
        a_vec.ymm = _mm256_loadu_si256((__m256i const *)a);
        b_vec.ymm = _mm256_loadu_si256((__m256i const *)b);

        // Intersecting registers with `_mm512_2intersect_epi16_mask` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u16_t a_min;
        nk_u16_t a_max = a_vec.u16s[15];
        nk_u16_t b_min = b_vec.u16s[0];
        nk_u16_t b_max = b_vec.u16s[15];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 32 <= a_end) {
            a += 16, a_weights += 16;
            a_vec.ymm = _mm256_loadu_si256((__m256i const *)a);
            a_max = a_vec.u16s[15];
        }
        a_min = a_vec.u16s[0];
        while (b_max < a_min && b + 32 <= b_end) {
            b += 16, b_weights += 16;
            b_vec.ymm = _mm256_loadu_si256((__m256i const *)b);
            b_max = b_vec.u16s[15];
        }
        b_min = b_vec.u16s[0];

        // Now we are likely to have some overlap, so we can intersect the registers
        __m512i a_i32x16 = _mm512_cvtepu16_epi32(a_vec.ymm);
        __m512i b_i32x16 = _mm512_cvtepu16_epi32(b_vec.ymm);
        __mmask16 a_matches_any_in_b, b_matches_any_in_a;
        _mm512_2intersect_epi32(a_i32x16, b_i32x16, &a_matches_any_in_b, &b_matches_any_in_a);

        // Load and shift all the relevant weights to the start of the vector before doing the dot product
        if (a_matches_any_in_b) {
            __m256i a_weights_bf16x16 = _mm256_loadu_si256((__m256i const *)a_weights);
            a_weights_bf16x16 = _mm256_maskz_compress_epi16(a_matches_any_in_b, a_weights_bf16x16);
            __m256i b_weights_bf16x16 = _mm256_loadu_si256((__m256i const *)b_weights);
            b_weights_bf16x16 = _mm256_maskz_compress_epi16(b_matches_any_in_a, b_weights_bf16x16);
            product_f32x8 = _mm256_dpbf16_ps(product_f32x8, nk_m256bh_from_m256i_(a_weights_bf16x16),
                                             nk_m256bh_from_m256i_(b_weights_bf16x16));
        }

        __m256i a_max_u16x16 = _mm256_permutexvar_epi16(last_idx, a_vec.ymm);
        __m256i b_max_u16x16 = _mm256_permutexvar_epi16(last_idx, b_vec.ymm);
        __mmask16 a_step_mask = _mm256_cmple_epu16_mask(a_vec.ymm, b_max_u16x16);
        __mmask16 b_step_mask = _mm256_cmple_epu16_mask(b_vec.ymm, a_max_u16x16);
        nk_size_t a_step = _tzcnt_u32(~(nk_u32_t)a_step_mask | 0x10000);
        nk_size_t b_step = _tzcnt_u32(~(nk_u32_t)b_step_mask | 0x10000);
        a += a_step, a_weights += a_step;
        b += b_step, b_weights += b_step;
    }
    nk_f32_t tail_product = 0;
    nk_sparse_dot_u16bf16_serial(a, b, a_weights, b_weights, a_end - a, b_end - b, &tail_product);
    *product = tail_product + _mm512_reduce_add_ps(_mm512_insertf32x8(_mm512_setzero_ps(), product_f32x8, 0));
}

NK_PUBLIC void nk_sparse_dot_u32f32_turin(                //
    nk_u32_t const *a, nk_u32_t const *b,                 //
    nk_f32_t const *a_weights, nk_f32_t const *b_weights, //
    nk_size_t a_length, nk_size_t b_length,               //
    nk_f64_t *product) {

#if NK_ALLOW_ISA_REDIRECT
    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 32 && b_length < 32) {
        nk_sparse_dot_u32f32_serial(a, b, a_weights, b_weights, a_length, b_length, product);
        return;
    }
#endif

    // Native VP2INTERSECTD works directly on u32 - no conversion needed!
    nk_u32_t const *const a_end = a + a_length;
    nk_u32_t const *const b_end = b + b_length;
    __m512d product_lower_f64x8 = _mm512_setzero_pd();
    __m512d product_upper_f64x8 = _mm512_setzero_pd();
    nk_b512_vec_t a_vec, b_vec;

    while (a + 16 <= a_end && b + 16 <= b_end) {
        a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
        b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);

        // Avoid expensive intersection if slices don't overlap at all
        nk_u32_t a_min;
        nk_u32_t a_max = a_vec.u32s[15];
        nk_u32_t b_min = b_vec.u32s[0];
        nk_u32_t b_max = b_vec.u32s[15];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 32 <= a_end) {
            a += 16, a_weights += 16;
            a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
            a_max = a_vec.u32s[15];
        }
        a_min = a_vec.u32s[0];
        while (b_max < a_min && b + 32 <= b_end) {
            b += 16, b_weights += 16;
            b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);
            b_max = b_vec.u32s[15];
        }
        b_min = b_vec.u32s[0];

        // Native u32 intersection - no conversion needed!
        __mmask16 a_matches, b_matches;
        _mm512_2intersect_epi32(a_vec.zmm, b_vec.zmm, &a_matches, &b_matches);

        // Load and compress matching weights, then FMA
        if (a_matches) {
            __m512 a_weights_f32x16 = _mm512_loadu_ps(a_weights);
            __m512 b_weights_f32x16 = _mm512_loadu_ps(b_weights);
            __m512 a_matched_f32x16 = _mm512_maskz_compress_ps(a_matches, a_weights_f32x16);
            __m512 b_matched_f32x16 = _mm512_maskz_compress_ps(b_matches, b_weights_f32x16);
            __m256 a_matched_lower_f32x8 = _mm512_castps512_ps256(a_matched_f32x16);
            __m256 a_matched_upper_f32x8 = _mm512_extractf32x8_ps(a_matched_f32x16, 1);
            __m256 b_matched_lower_f32x8 = _mm512_castps512_ps256(b_matched_f32x16);
            __m256 b_matched_upper_f32x8 = _mm512_extractf32x8_ps(b_matched_f32x16, 1);

            product_lower_f64x8 = _mm512_fmadd_pd(_mm512_cvtps_pd(a_matched_lower_f32x8),
                                                  _mm512_cvtps_pd(b_matched_lower_f32x8), product_lower_f64x8);
            product_upper_f64x8 = _mm512_fmadd_pd(_mm512_cvtps_pd(a_matched_upper_f32x8),
                                                  _mm512_cvtps_pd(b_matched_upper_f32x8), product_upper_f64x8);
        }

        __m512i a_max_u32x16 = _mm512_set1_epi32(*(int const *)&a_max);
        __m512i b_max_u32x16 = _mm512_set1_epi32(*(int const *)&b_max);
        __mmask16 a_step_mask = _mm512_cmple_epu32_mask(a_vec.zmm, b_max_u32x16);
        __mmask16 b_step_mask = _mm512_cmple_epu32_mask(b_vec.zmm, a_max_u32x16);
        nk_size_t a_step = _tzcnt_u32(~(nk_u32_t)a_step_mask | 0x10000);
        nk_size_t b_step = _tzcnt_u32(~(nk_u32_t)b_step_mask | 0x10000);
        a += a_step, a_weights += a_step;
        b += b_step, b_weights += b_step;
    }

    nk_f64_t tail_product = 0;
    nk_sparse_dot_u32f32_serial(a, b, a_weights, b_weights, a_end - a, b_end - b, &tail_product);
    *product = _mm512_reduce_add_pd(product_lower_f64x8) + _mm512_reduce_add_pd(product_upper_f64x8) + tail_product;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_TURIN
#endif // NK_TARGET_X86_
#endif // NK_SPARSE_TURIN_H
