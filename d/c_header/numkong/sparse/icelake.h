/**
 *  @brief Ice Lake-accelerated Sparse Vector Operations.
 *  @file include/numkong/sparse/icelake.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/sparse.h
 *
 *  The AVX-512 implementations are inspired by the "Faster-Than-Native Alternatives
 *  for x86 VP2INTERSECT Instructions" paper by Guille Diez-Canas, 2022.
 *
 *      https://github.com/mozonaut/vp2intersect
 *      https://arxiv.org/pdf/2112.06342.pdf
 *
 *  For R&D purposes, it's important to keep the following latencies in mind:
 *
 *   - `_mm512_permutex_epi64` (VPERMQ) - needs F - 3 cy latency, 1 cy throughput @ p5
 *   - `_mm512_shuffle_epi8` (VPSHUFB) - needs BW - 1 cy latency, 1 cy throughput @ p5
 *   - `_mm512_permutexvar_epi16` (VPERMW) - needs BW - 4-6 cy latency, 1 cy throughput @ p5
 *   - `_mm512_permutexvar_epi8` (VPERMB) - needs VBMI - 3 cy latency, 1 cy throughput @ p5
 */
#ifndef NK_SPARSE_ICELAKE_H
#define NK_SPARSE_ICELAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_ICELAKE

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                         \
    __attribute__((target("avx2,avx512f,avx512vl,avx512dq,bmi2,lzcnt,popcnt,avx512bw,avx512vbmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512dq", "bmi2", "lzcnt", "popcnt", "avx512bw", "avx512vbmi2")
#endif

/**
 *  @brief  Analogous to `_mm512_2intersect_epi16_mask`, but compatible with Ice Lake CPUs,
 *          slightly faster than the native Tiger Lake implementation, but returns only one mask.
 */
NK_INTERNAL nk_u32_t nk_intersect_u16x32_icelake_(__m512i a, __m512i b) {
    __m512i a1 = _mm512_alignr_epi32(a, a, 4);
    __m512i a2 = _mm512_alignr_epi32(a, a, 8);
    __m512i a3 = _mm512_alignr_epi32(a, a, 12);

    __m512i b1 = _mm512_shuffle_epi32(b, _MM_PERM_ADCB);
    __m512i b2 = _mm512_shuffle_epi32(b, _MM_PERM_BADC);
    __m512i b3 = _mm512_shuffle_epi32(b, _MM_PERM_CBAD);

    __m512i b01 = _mm512_shrdi_epi32(b, b, 16);
    __m512i b11 = _mm512_shrdi_epi32(b1, b1, 16);
    __m512i b21 = _mm512_shrdi_epi32(b2, b2, 16);
    __m512i b31 = _mm512_shrdi_epi32(b3, b3, 16);

    __mmask32 nm00 = _mm512_cmpneq_epi16_mask(a, b);
    __mmask32 nm01 = _mm512_cmpneq_epi16_mask(a1, b);
    __mmask32 nm02 = _mm512_cmpneq_epi16_mask(a2, b);
    __mmask32 nm03 = _mm512_cmpneq_epi16_mask(a3, b);

    __mmask32 nm10 = _mm512_mask_cmpneq_epi16_mask(nm00, a, b01);
    __mmask32 nm11 = _mm512_mask_cmpneq_epi16_mask(nm01, a1, b01);
    __mmask32 nm12 = _mm512_mask_cmpneq_epi16_mask(nm02, a2, b01);
    __mmask32 nm13 = _mm512_mask_cmpneq_epi16_mask(nm03, a3, b01);

    __mmask32 nm20 = _mm512_mask_cmpneq_epi16_mask(nm10, a, b1);
    __mmask32 nm21 = _mm512_mask_cmpneq_epi16_mask(nm11, a1, b1);
    __mmask32 nm22 = _mm512_mask_cmpneq_epi16_mask(nm12, a2, b1);
    __mmask32 nm23 = _mm512_mask_cmpneq_epi16_mask(nm13, a3, b1);

    __mmask32 nm30 = _mm512_mask_cmpneq_epi16_mask(nm20, a, b11);
    __mmask32 nm31 = _mm512_mask_cmpneq_epi16_mask(nm21, a1, b11);
    __mmask32 nm32 = _mm512_mask_cmpneq_epi16_mask(nm22, a2, b11);
    __mmask32 nm33 = _mm512_mask_cmpneq_epi16_mask(nm23, a3, b11);

    __mmask32 nm40 = _mm512_mask_cmpneq_epi16_mask(nm30, a, b2);
    __mmask32 nm41 = _mm512_mask_cmpneq_epi16_mask(nm31, a1, b2);
    __mmask32 nm42 = _mm512_mask_cmpneq_epi16_mask(nm32, a2, b2);
    __mmask32 nm43 = _mm512_mask_cmpneq_epi16_mask(nm33, a3, b2);

    __mmask32 nm50 = _mm512_mask_cmpneq_epi16_mask(nm40, a, b21);
    __mmask32 nm51 = _mm512_mask_cmpneq_epi16_mask(nm41, a1, b21);
    __mmask32 nm52 = _mm512_mask_cmpneq_epi16_mask(nm42, a2, b21);
    __mmask32 nm53 = _mm512_mask_cmpneq_epi16_mask(nm43, a3, b21);

    __mmask32 nm60 = _mm512_mask_cmpneq_epi16_mask(nm50, a, b3);
    __mmask32 nm61 = _mm512_mask_cmpneq_epi16_mask(nm51, a1, b3);
    __mmask32 nm62 = _mm512_mask_cmpneq_epi16_mask(nm52, a2, b3);
    __mmask32 nm63 = _mm512_mask_cmpneq_epi16_mask(nm53, a3, b3);

    __mmask32 nm70 = _mm512_mask_cmpneq_epi16_mask(nm60, a, b31);
    __mmask32 nm71 = _mm512_mask_cmpneq_epi16_mask(nm61, a1, b31);
    __mmask32 nm72 = _mm512_mask_cmpneq_epi16_mask(nm62, a2, b31);
    __mmask32 nm73 = _mm512_mask_cmpneq_epi16_mask(nm63, a3, b31);

    return ~(nk_u32_t)(nm70 & nk_u32_rol(nm71, 8) & nk_u32_rol(nm72, 16) & nk_u32_ror(nm73, 8));
}

/**
 *  @brief  Analogous to `_mm512_2intersect_epi32`, but compatible with Ice Lake CPUs,
 *          slightly faster than the native Tiger Lake implementation, but returns only one mask.
 */
NK_INTERNAL nk_u16_t nk_intersect_u32x16_icelake_(__m512i a, __m512i b) {
    __m512i a1 = _mm512_alignr_epi32(a, a, 4);
    __m512i b1 = _mm512_shuffle_epi32(b, _MM_PERM_ADCB);
    __mmask16 nm00 = _mm512_cmpneq_epi32_mask(a, b);

    __m512i a2 = _mm512_alignr_epi32(a, a, 8);
    __m512i a3 = _mm512_alignr_epi32(a, a, 12);
    __mmask16 nm01 = _mm512_cmpneq_epi32_mask(a1, b);
    __mmask16 nm02 = _mm512_cmpneq_epi32_mask(a2, b);

    __mmask16 nm03 = _mm512_cmpneq_epi32_mask(a3, b);
    __mmask16 nm10 = _mm512_mask_cmpneq_epi32_mask(nm00, a, b1);
    __mmask16 nm11 = _mm512_mask_cmpneq_epi32_mask(nm01, a1, b1);

    __m512i b2 = _mm512_shuffle_epi32(b, _MM_PERM_BADC);
    __mmask16 nm12 = _mm512_mask_cmpneq_epi32_mask(nm02, a2, b1);
    __mmask16 nm13 = _mm512_mask_cmpneq_epi32_mask(nm03, a3, b1);
    __mmask16 nm20 = _mm512_mask_cmpneq_epi32_mask(nm10, a, b2);

    __m512i b3 = _mm512_shuffle_epi32(b, _MM_PERM_CBAD);
    __mmask16 nm21 = _mm512_mask_cmpneq_epi32_mask(nm11, a1, b2);
    __mmask16 nm22 = _mm512_mask_cmpneq_epi32_mask(nm12, a2, b2);
    __mmask16 nm23 = _mm512_mask_cmpneq_epi32_mask(nm13, a3, b2);

    __mmask16 nm0 = _mm512_mask_cmpneq_epi32_mask(nm20, a, b3);
    __mmask16 nm1 = _mm512_mask_cmpneq_epi32_mask(nm21, a1, b3);
    __mmask16 nm2 = _mm512_mask_cmpneq_epi32_mask(nm22, a2, b3);
    __mmask16 nm3 = _mm512_mask_cmpneq_epi32_mask(nm23, a3, b3);

    return ~(nk_u16_t)(nm0 & nk_u16_rol(nm1, 4) & nk_u16_rol(nm2, 8) & nk_u16_ror(nm3, 4));
}

NK_PUBLIC void nk_sparse_intersect_u16_icelake( //
    nk_u16_t const *a, nk_u16_t const *b,       //
    nk_size_t a_length, nk_size_t b_length,     //
    nk_u16_t *result, nk_size_t *count) {

#if NK_ALLOW_ISA_REDIRECT
    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 64 && b_length < 64) {
        nk_sparse_intersect_u16_serial(a, b, a_length, b_length, result, count);
        return;
    }
#endif

    nk_u16_t const *const a_end = a + a_length;
    nk_u16_t const *const b_end = b + b_length;
    nk_size_t c = 0;
    nk_b512_vec_t a_vec, b_vec;

    while (a + 32 <= a_end && b + 32 <= b_end) {
        a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
        b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);

        // Intersecting registers with `nk_intersect_u16x32_icelake_` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u16_t a_min;
        nk_u16_t a_max = a_vec.u16s[31];
        nk_u16_t b_min = b_vec.u16s[0];
        nk_u16_t b_max = b_vec.u16s[31];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 64 <= a_end) {
            a += 32;
            a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
            a_max = a_vec.u16s[31];
        }
        a_min = a_vec.u16s[0];
        while (b_max < a_min && b + 64 <= b_end) {
            b += 32;
            b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);
            b_max = b_vec.u16s[31];
        }
        b_min = b_vec.u16s[0];

        __m512i a_max_u16x32 = _mm512_set1_epi16(*(short const *)&a_max);
        __m512i b_max_u16x32 = _mm512_set1_epi16(*(short const *)&b_max);
        __mmask32 a_step_mask = _mm512_cmple_epu16_mask(a_vec.zmm, b_max_u16x32);
        __mmask32 b_step_mask = _mm512_cmple_epu16_mask(b_vec.zmm, a_max_u16x32);
        a += 32 - _lzcnt_u32((nk_u32_t)a_step_mask);
        b += 32 - _lzcnt_u32((nk_u32_t)b_step_mask);

        // Now we are likely to have some overlap, so we can intersect the registers
        __mmask32 a_matches = nk_intersect_u16x32_icelake_(a_vec.zmm, b_vec.zmm);

        // Export matches if result buffer is provided
        if (result) { _mm512_mask_compressstoreu_epi16(result + c, a_matches, a_vec.zmm); }
        c += _mm_popcnt_u32(a_matches); // MSVC has no `_popcnt32`
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u16_serial(a, b, a_end - a, b_end - b, result ? result + c : 0, &tail_count);
    *count = c + tail_count;
}

NK_PUBLIC void nk_sparse_intersect_u32_icelake( //
    nk_u32_t const *a, nk_u32_t const *b,       //
    nk_size_t a_length, nk_size_t b_length,     //
    nk_u32_t *result, nk_size_t *count) {

#if NK_ALLOW_ISA_REDIRECT
    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 32 && b_length < 32) {
        nk_sparse_intersect_u32_serial(a, b, a_length, b_length, result, count);
        return;
    }
#endif

    nk_u32_t const *const a_end = a + a_length;
    nk_u32_t const *const b_end = b + b_length;
    nk_size_t c = 0;
    nk_b512_vec_t a_vec, b_vec;

    while (a + 16 <= a_end && b + 16 <= b_end) {
        a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
        b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);

        // Intersecting registers with `nk_intersect_u32x16_icelake_` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u32_t a_min;
        nk_u32_t a_max = a_vec.u32s[15];
        nk_u32_t b_min = b_vec.u32s[0];
        nk_u32_t b_max = b_vec.u32s[15];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 32 <= a_end) {
            a += 16;
            a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
            a_max = a_vec.u32s[15];
        }
        a_min = a_vec.u32s[0];
        while (b_max < a_min && b + 32 <= b_end) {
            b += 16;
            b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);
            b_max = b_vec.u32s[15];
        }
        b_min = b_vec.u32s[0];

        __m512i a_max_u32x16 = _mm512_set1_epi32(*(int const *)&a_max);
        __m512i b_max_u32x16 = _mm512_set1_epi32(*(int const *)&b_max);
        __mmask16 a_step_mask = _mm512_cmple_epu32_mask(a_vec.zmm, b_max_u32x16);
        __mmask16 b_step_mask = _mm512_cmple_epu32_mask(b_vec.zmm, a_max_u32x16);
        a += 32 - _lzcnt_u32((nk_u32_t)a_step_mask);
        b += 32 - _lzcnt_u32((nk_u32_t)b_step_mask);

        // Now we are likely to have some overlap, so we can intersect the registers
        __mmask16 a_matches = nk_intersect_u32x16_icelake_(a_vec.zmm, b_vec.zmm);

        // Export matches if result buffer is provided
        if (result) { _mm512_mask_compressstoreu_epi32(result + c, a_matches, a_vec.zmm); }
        c += _mm_popcnt_u32(a_matches); // MSVC has no `_popcnt32`
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u32_serial(a, b, a_end - a, b_end - b, result ? result + c : 0, &tail_count);
    *count = c + tail_count;
}

/**
 *  @brief  Analogous to `_mm512_2intersect_epi64`, but compatible with Ice Lake CPUs,
 *          returns only one mask indicating which elements in `a` have a match in `b`.
 */
NK_INTERNAL nk_u8_t nk_intersect_u64x8_icelake_(__m512i a, __m512i b) {
    __m512i a1 = _mm512_alignr_epi64(a, a, 2);
    __m512i b1 = _mm512_permutex_epi64(b, _MM_PERM_ADCB);
    __mmask8 nm00 = _mm512_cmpneq_epi64_mask(a, b);

    __m512i a2 = _mm512_alignr_epi64(a, a, 4);
    __m512i a3 = _mm512_alignr_epi64(a, a, 6);
    __mmask8 nm01 = _mm512_cmpneq_epi64_mask(a1, b);
    __mmask8 nm02 = _mm512_cmpneq_epi64_mask(a2, b);

    __m512i b2 = _mm512_permutex_epi64(b, _MM_PERM_BADC);
    __mmask8 nm03 = _mm512_cmpneq_epi64_mask(a3, b);
    __mmask8 nm10 = _mm512_mask_cmpneq_epi64_mask(nm00, a, b1);
    __mmask8 nm11 = _mm512_mask_cmpneq_epi64_mask(nm01, a1, b1);

    __m512i b3 = _mm512_permutex_epi64(b, _MM_PERM_CBAD);
    __mmask8 nm12 = _mm512_mask_cmpneq_epi64_mask(nm02, a2, b1);
    __mmask8 nm13 = _mm512_mask_cmpneq_epi64_mask(nm03, a3, b1);
    __mmask8 nm20 = _mm512_mask_cmpneq_epi64_mask(nm10, a, b2);

    __mmask8 nm21 = _mm512_mask_cmpneq_epi64_mask(nm11, a1, b2);
    __mmask8 nm22 = _mm512_mask_cmpneq_epi64_mask(nm12, a2, b2);
    __mmask8 nm23 = _mm512_mask_cmpneq_epi64_mask(nm13, a3, b2);

    __mmask8 nm0 = _mm512_mask_cmpneq_epi64_mask(nm20, a, b3);
    __mmask8 nm1 = _mm512_mask_cmpneq_epi64_mask(nm21, a1, b3);
    __mmask8 nm2 = _mm512_mask_cmpneq_epi64_mask(nm22, a2, b3);
    __mmask8 nm3 = _mm512_mask_cmpneq_epi64_mask(nm23, a3, b3);

    return ~(nk_u8_t)(nm0 & nk_u8_rol(nm1, 2) & nk_u8_rol(nm2, 4) & nk_u8_ror(nm3, 2));
}

NK_PUBLIC void nk_sparse_intersect_u64_icelake( //
    nk_u64_t const *a, nk_u64_t const *b,       //
    nk_size_t a_length, nk_size_t b_length,     //
    nk_u64_t *result, nk_size_t *count) {

#if NK_ALLOW_ISA_REDIRECT
    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 16 && b_length < 16) {
        nk_sparse_intersect_u64_serial(a, b, a_length, b_length, result, count);
        return;
    }
#endif

    nk_u64_t const *const a_end = a + a_length;
    nk_u64_t const *const b_end = b + b_length;
    nk_size_t c = 0;
    nk_b512_vec_t a_vec, b_vec;

    while (a + 8 <= a_end && b + 8 <= b_end) {
        a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
        b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);

        // Intersecting registers with `nk_intersect_u64x8_icelake_` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all.
        nk_u64_t a_min;
        nk_u64_t a_max = a_vec.u64s[7];
        nk_u64_t b_min = b_vec.u64s[0];
        nk_u64_t b_max = b_vec.u64s[7];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 16 <= a_end) {
            a += 8;
            a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
            a_max = a_vec.u64s[7];
        }
        a_min = a_vec.u64s[0];
        while (b_max < a_min && b + 16 <= b_end) {
            b += 8;
            b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);
            b_max = b_vec.u64s[7];
        }
        b_min = b_vec.u64s[0];

        __m512i a_max_u64x8 = _mm512_set1_epi64(*(long long const *)&a_max);
        __m512i b_max_u64x8 = _mm512_set1_epi64(*(long long const *)&b_max);
        __mmask8 a_step_mask = _mm512_cmple_epu64_mask(a_vec.zmm, b_max_u64x8);
        __mmask8 b_step_mask = _mm512_cmple_epu64_mask(b_vec.zmm, a_max_u64x8);
        a += 32 - _lzcnt_u32((nk_u32_t)a_step_mask);
        b += 32 - _lzcnt_u32((nk_u32_t)b_step_mask);

        // Now we are likely to have some overlap, so we can intersect the registers
        __mmask8 a_matches = nk_intersect_u64x8_icelake_(a_vec.zmm, b_vec.zmm);

        // Export matches if result buffer is provided
        if (result) { _mm512_mask_compressstoreu_epi64(result + c, a_matches, a_vec.zmm); }
        c += _mm_popcnt_u32(a_matches); // MSVC has no `_popcnt32`
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u64_serial(a, b, a_end - a, b_end - b, result ? result + c : 0, &tail_count);
    *count = c + tail_count;
}

NK_PUBLIC void nk_sparse_dot_u32f32_icelake(              //
    nk_u32_t const *a, nk_u32_t const *b,                 //
    nk_f32_t const *a_weights, nk_f32_t const *b_weights, //
    nk_size_t a_length, nk_size_t b_length, nk_f64_t *product) {

#if NK_ALLOW_ISA_REDIRECT
    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 32 && b_length < 32) {
        nk_sparse_dot_u32f32_serial(a, b, a_weights, b_weights, a_length, b_length, product);
        return;
    }
#endif

    nk_u32_t const *const a_end = a + a_length;
    nk_u32_t const *const b_end = b + b_length;
    __m512d product_lower_f64x8 = _mm512_setzero_pd();
    __m512d product_upper_f64x8 = _mm512_setzero_pd();
    nk_b512_vec_t a_vec, b_vec;

    while (a + 16 <= a_end && b + 16 <= b_end) {
        a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
        b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);

        // Intersecting registers with `nk_intersect_u32x16_icelake_` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all.
        nk_u32_t a_min;
        nk_u32_t a_max = a_vec.u32s[15];
        nk_u32_t b_min = b_vec.u32s[0];
        nk_u32_t b_max = b_vec.u32s[15];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 32 <= a_end) {
            a += 16;
            a_weights += 16;
            a_vec.zmm = _mm512_loadu_si512((__m512i const *)a);
            a_max = a_vec.u32s[15];
        }
        a_min = a_vec.u32s[0];
        while (b_max < a_min && b + 32 <= b_end) {
            b += 16;
            b_weights += 16;
            b_vec.zmm = _mm512_loadu_si512((__m512i const *)b);
            b_max = b_vec.u32s[15];
        }
        b_min = b_vec.u32s[0];

        __m512i a_max_u32x16 = _mm512_set1_epi32(*(int const *)&a_max);
        __m512i b_max_u32x16 = _mm512_set1_epi32(*(int const *)&b_max);
        __mmask16 a_step_mask = _mm512_cmple_epu32_mask(a_vec.zmm, b_max_u32x16);
        __mmask16 b_step_mask = _mm512_cmple_epu32_mask(b_vec.zmm, a_max_u32x16);
        nk_u32_t a_advance = 32 - _lzcnt_u32((nk_u32_t)a_step_mask);
        nk_u32_t b_advance = 32 - _lzcnt_u32((nk_u32_t)b_step_mask);

        // Now we are likely to have some overlap, so we can intersect the registers
        __mmask16 a_matches = nk_intersect_u32x16_icelake_(a_vec.zmm, b_vec.zmm);
        __mmask16 b_matches = nk_intersect_u32x16_icelake_(b_vec.zmm, a_vec.zmm);
        if (a_matches) {
            // Load and compress matching weights at current position
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

        // Advance pointers after processing
        a += a_advance;
        a_weights += a_advance;
        b += b_advance;
        b_weights += b_advance;
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

#endif // NK_TARGET_ICELAKE
#endif // NK_TARGET_X86_
#endif // NK_SPARSE_ICELAKE_H
