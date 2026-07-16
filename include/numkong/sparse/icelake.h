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

#if NK_TARGET_X8664_
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
NK_HELPER_INLINE nk_u32_t nk_intersect_u16x32_icelake_(__m512i a, __m512i b) {
    __m512i a1_u16x32 = _mm512_alignr_epi32(a, a, 4);
    __m512i a2_u16x32 = _mm512_alignr_epi32(a, a, 8);
    __m512i a3_u16x32 = _mm512_alignr_epi32(a, a, 12);

    __m512i b1_u16x32 = _mm512_shuffle_epi32(b, _MM_PERM_ADCB);
    __m512i b2_u16x32 = _mm512_shuffle_epi32(b, _MM_PERM_BADC);
    __m512i b3_u16x32 = _mm512_shuffle_epi32(b, _MM_PERM_CBAD);

    __m512i b01_u16x32 = _mm512_shrdi_epi32(b, b, 16);
    __m512i b11_u16x32 = _mm512_shrdi_epi32(b1_u16x32, b1_u16x32, 16);
    __m512i b21_u16x32 = _mm512_shrdi_epi32(b2_u16x32, b2_u16x32, 16);
    __m512i b31_u16x32 = _mm512_shrdi_epi32(b3_u16x32, b3_u16x32, 16);

    __mmask32 nm00_m32 = _mm512_cmpneq_epi16_mask(a, b);
    __mmask32 nm01_m32 = _mm512_cmpneq_epi16_mask(a1_u16x32, b);
    __mmask32 nm02_m32 = _mm512_cmpneq_epi16_mask(a2_u16x32, b);
    __mmask32 nm03_m32 = _mm512_cmpneq_epi16_mask(a3_u16x32, b);

    __mmask32 nm10_m32 = _mm512_mask_cmpneq_epi16_mask(nm00_m32, a, b01_u16x32);
    __mmask32 nm11_m32 = _mm512_mask_cmpneq_epi16_mask(nm01_m32, a1_u16x32, b01_u16x32);
    __mmask32 nm12_m32 = _mm512_mask_cmpneq_epi16_mask(nm02_m32, a2_u16x32, b01_u16x32);
    __mmask32 nm13_m32 = _mm512_mask_cmpneq_epi16_mask(nm03_m32, a3_u16x32, b01_u16x32);

    __mmask32 nm20_m32 = _mm512_mask_cmpneq_epi16_mask(nm10_m32, a, b1_u16x32);
    __mmask32 nm21_m32 = _mm512_mask_cmpneq_epi16_mask(nm11_m32, a1_u16x32, b1_u16x32);
    __mmask32 nm22_m32 = _mm512_mask_cmpneq_epi16_mask(nm12_m32, a2_u16x32, b1_u16x32);
    __mmask32 nm23_m32 = _mm512_mask_cmpneq_epi16_mask(nm13_m32, a3_u16x32, b1_u16x32);

    __mmask32 nm30_m32 = _mm512_mask_cmpneq_epi16_mask(nm20_m32, a, b11_u16x32);
    __mmask32 nm31_m32 = _mm512_mask_cmpneq_epi16_mask(nm21_m32, a1_u16x32, b11_u16x32);
    __mmask32 nm32_m32 = _mm512_mask_cmpneq_epi16_mask(nm22_m32, a2_u16x32, b11_u16x32);
    __mmask32 nm33_m32 = _mm512_mask_cmpneq_epi16_mask(nm23_m32, a3_u16x32, b11_u16x32);

    __mmask32 nm40_m32 = _mm512_mask_cmpneq_epi16_mask(nm30_m32, a, b2_u16x32);
    __mmask32 nm41_m32 = _mm512_mask_cmpneq_epi16_mask(nm31_m32, a1_u16x32, b2_u16x32);
    __mmask32 nm42_m32 = _mm512_mask_cmpneq_epi16_mask(nm32_m32, a2_u16x32, b2_u16x32);
    __mmask32 nm43_m32 = _mm512_mask_cmpneq_epi16_mask(nm33_m32, a3_u16x32, b2_u16x32);

    __mmask32 nm50_m32 = _mm512_mask_cmpneq_epi16_mask(nm40_m32, a, b21_u16x32);
    __mmask32 nm51_m32 = _mm512_mask_cmpneq_epi16_mask(nm41_m32, a1_u16x32, b21_u16x32);
    __mmask32 nm52_m32 = _mm512_mask_cmpneq_epi16_mask(nm42_m32, a2_u16x32, b21_u16x32);
    __mmask32 nm53_m32 = _mm512_mask_cmpneq_epi16_mask(nm43_m32, a3_u16x32, b21_u16x32);

    __mmask32 nm60_m32 = _mm512_mask_cmpneq_epi16_mask(nm50_m32, a, b3_u16x32);
    __mmask32 nm61_m32 = _mm512_mask_cmpneq_epi16_mask(nm51_m32, a1_u16x32, b3_u16x32);
    __mmask32 nm62_m32 = _mm512_mask_cmpneq_epi16_mask(nm52_m32, a2_u16x32, b3_u16x32);
    __mmask32 nm63_m32 = _mm512_mask_cmpneq_epi16_mask(nm53_m32, a3_u16x32, b3_u16x32);

    __mmask32 nm70_m32 = _mm512_mask_cmpneq_epi16_mask(nm60_m32, a, b31_u16x32);
    __mmask32 nm71_m32 = _mm512_mask_cmpneq_epi16_mask(nm61_m32, a1_u16x32, b31_u16x32);
    __mmask32 nm72_m32 = _mm512_mask_cmpneq_epi16_mask(nm62_m32, a2_u16x32, b31_u16x32);
    __mmask32 nm73_m32 = _mm512_mask_cmpneq_epi16_mask(nm63_m32, a3_u16x32, b31_u16x32);

    return ~(nk_u32_t)(nm70_m32 & nk_u32_rol(nm71_m32, 8) & nk_u32_rol(nm72_m32, 16) & nk_u32_ror(nm73_m32, 8));
}

/**
 *  @brief  Analogous to `_mm512_2intersect_epi32`, but compatible with Ice Lake CPUs,
 *          slightly faster than the native Tiger Lake implementation, but returns only one mask.
 */
NK_HELPER_INLINE nk_u16_t nk_intersect_u32x16_icelake_(__m512i a, __m512i b) {
    __m512i a1_u32x16 = _mm512_alignr_epi32(a, a, 4);
    __m512i b1_u32x16 = _mm512_shuffle_epi32(b, _MM_PERM_ADCB);
    __mmask16 nm00_m16 = _mm512_cmpneq_epi32_mask(a, b);

    __m512i a2_u32x16 = _mm512_alignr_epi32(a, a, 8);
    __m512i a3_u32x16 = _mm512_alignr_epi32(a, a, 12);
    __mmask16 nm01_m16 = _mm512_cmpneq_epi32_mask(a1_u32x16, b);
    __mmask16 nm02_m16 = _mm512_cmpneq_epi32_mask(a2_u32x16, b);

    __mmask16 nm03_m16 = _mm512_cmpneq_epi32_mask(a3_u32x16, b);
    __mmask16 nm10_m16 = _mm512_mask_cmpneq_epi32_mask(nm00_m16, a, b1_u32x16);
    __mmask16 nm11_m16 = _mm512_mask_cmpneq_epi32_mask(nm01_m16, a1_u32x16, b1_u32x16);

    __m512i b2_u32x16 = _mm512_shuffle_epi32(b, _MM_PERM_BADC);
    __mmask16 nm12_m16 = _mm512_mask_cmpneq_epi32_mask(nm02_m16, a2_u32x16, b1_u32x16);
    __mmask16 nm13_m16 = _mm512_mask_cmpneq_epi32_mask(nm03_m16, a3_u32x16, b1_u32x16);
    __mmask16 nm20_m16 = _mm512_mask_cmpneq_epi32_mask(nm10_m16, a, b2_u32x16);

    __m512i b3_u32x16 = _mm512_shuffle_epi32(b, _MM_PERM_CBAD);
    __mmask16 nm21_m16 = _mm512_mask_cmpneq_epi32_mask(nm11_m16, a1_u32x16, b2_u32x16);
    __mmask16 nm22_m16 = _mm512_mask_cmpneq_epi32_mask(nm12_m16, a2_u32x16, b2_u32x16);
    __mmask16 nm23_m16 = _mm512_mask_cmpneq_epi32_mask(nm13_m16, a3_u32x16, b2_u32x16);

    __mmask16 nm0_m16 = _mm512_mask_cmpneq_epi32_mask(nm20_m16, a, b3_u32x16);
    __mmask16 nm1_m16 = _mm512_mask_cmpneq_epi32_mask(nm21_m16, a1_u32x16, b3_u32x16);
    __mmask16 nm2_m16 = _mm512_mask_cmpneq_epi32_mask(nm22_m16, a2_u32x16, b3_u32x16);
    __mmask16 nm3_m16 = _mm512_mask_cmpneq_epi32_mask(nm23_m16, a3_u32x16, b3_u32x16);

    return ~(nk_u16_t)(nm0_m16 & nk_u16_rol(nm1_m16, 4) & nk_u16_rol(nm2_m16, 8) & nk_u16_ror(nm3_m16, 4));
}

NK_API_COMPTIME void nk_sparse_intersect_u16_icelake( //
    nk_u16_t const *a, nk_u16_t const *b,             //
    nk_size_t a_length, nk_size_t b_length,           //
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
        __mmask32 a_step_mask_m32 = _mm512_cmple_epu16_mask(a_vec.zmm, b_max_u16x32);
        __mmask32 b_step_mask_m32 = _mm512_cmple_epu16_mask(b_vec.zmm, a_max_u16x32);
        a += 32 - _lzcnt_u32((nk_u32_t)a_step_mask_m32);
        b += 32 - _lzcnt_u32((nk_u32_t)b_step_mask_m32);

        // Now we are likely to have some overlap, so we can intersect the registers
        __mmask32 a_matches_m32 = nk_intersect_u16x32_icelake_(a_vec.zmm, b_vec.zmm);

        // Export matches if result buffer is provided
        if (result) { _mm512_mask_compressstoreu_epi16(result + c, a_matches_m32, a_vec.zmm); }
        c += _mm_popcnt_u32(a_matches_m32); // MSVC has no `_popcnt32`
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u16_serial(a, b, a_end - a, b_end - b, result ? result + c : 0, &tail_count);
    *count = c + tail_count;
}

NK_API_COMPTIME void nk_sparse_intersect_u32_icelake( //
    nk_u32_t const *a, nk_u32_t const *b,             //
    nk_size_t a_length, nk_size_t b_length,           //
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
        __mmask16 a_step_mask_m16 = _mm512_cmple_epu32_mask(a_vec.zmm, b_max_u32x16);
        __mmask16 b_step_mask_m16 = _mm512_cmple_epu32_mask(b_vec.zmm, a_max_u32x16);
        a += 32 - _lzcnt_u32((nk_u32_t)a_step_mask_m16);
        b += 32 - _lzcnt_u32((nk_u32_t)b_step_mask_m16);

        // Now we are likely to have some overlap, so we can intersect the registers
        __mmask16 a_matches_m16 = nk_intersect_u32x16_icelake_(a_vec.zmm, b_vec.zmm);

        // Export matches if result buffer is provided
        if (result) { _mm512_mask_compressstoreu_epi32(result + c, a_matches_m16, a_vec.zmm); }
        c += _mm_popcnt_u32(a_matches_m16); // MSVC has no `_popcnt32`
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u32_serial(a, b, a_end - a, b_end - b, result ? result + c : 0, &tail_count);
    *count = c + tail_count;
}

/**
 *  @brief  Analogous to `_mm512_2intersect_epi64`, but compatible with Ice Lake CPUs,
 *          returns only one mask indicating which elements in `a` have a match in `b`.
 */
NK_HELPER_INLINE nk_u8_t nk_intersect_u64x8_icelake_(__m512i a, __m512i b) {
    __m512i a1_u64x8 = _mm512_alignr_epi64(a, a, 2);
    __m512i b1_u64x8 = _mm512_permutex_epi64(b, _MM_PERM_ADCB);
    __mmask8 nm00_m8 = _mm512_cmpneq_epi64_mask(a, b);

    __m512i a2_u64x8 = _mm512_alignr_epi64(a, a, 4);
    __m512i a3_u64x8 = _mm512_alignr_epi64(a, a, 6);
    __mmask8 nm01_m8 = _mm512_cmpneq_epi64_mask(a1_u64x8, b);
    __mmask8 nm02_m8 = _mm512_cmpneq_epi64_mask(a2_u64x8, b);

    __m512i b2_u64x8 = _mm512_permutex_epi64(b, _MM_PERM_BADC);
    __mmask8 nm03_m8 = _mm512_cmpneq_epi64_mask(a3_u64x8, b);
    __mmask8 nm10_m8 = _mm512_mask_cmpneq_epi64_mask(nm00_m8, a, b1_u64x8);
    __mmask8 nm11_m8 = _mm512_mask_cmpneq_epi64_mask(nm01_m8, a1_u64x8, b1_u64x8);

    __m512i b3_u64x8 = _mm512_permutex_epi64(b, _MM_PERM_CBAD);
    __mmask8 nm12_m8 = _mm512_mask_cmpneq_epi64_mask(nm02_m8, a2_u64x8, b1_u64x8);
    __mmask8 nm13_m8 = _mm512_mask_cmpneq_epi64_mask(nm03_m8, a3_u64x8, b1_u64x8);
    __mmask8 nm20_m8 = _mm512_mask_cmpneq_epi64_mask(nm10_m8, a, b2_u64x8);

    __mmask8 nm21_m8 = _mm512_mask_cmpneq_epi64_mask(nm11_m8, a1_u64x8, b2_u64x8);
    __mmask8 nm22_m8 = _mm512_mask_cmpneq_epi64_mask(nm12_m8, a2_u64x8, b2_u64x8);
    __mmask8 nm23_m8 = _mm512_mask_cmpneq_epi64_mask(nm13_m8, a3_u64x8, b2_u64x8);

    __mmask8 nm0_m8 = _mm512_mask_cmpneq_epi64_mask(nm20_m8, a, b3_u64x8);
    __mmask8 nm1_m8 = _mm512_mask_cmpneq_epi64_mask(nm21_m8, a1_u64x8, b3_u64x8);
    __mmask8 nm2_m8 = _mm512_mask_cmpneq_epi64_mask(nm22_m8, a2_u64x8, b3_u64x8);
    __mmask8 nm3_m8 = _mm512_mask_cmpneq_epi64_mask(nm23_m8, a3_u64x8, b3_u64x8);

    return ~(nk_u8_t)(nm0_m8 & nk_u8_rol(nm1_m8, 2) & nk_u8_rol(nm2_m8, 4) & nk_u8_ror(nm3_m8, 2));
}

NK_API_COMPTIME void nk_sparse_intersect_u64_icelake( //
    nk_u64_t const *a, nk_u64_t const *b,             //
    nk_size_t a_length, nk_size_t b_length,           //
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
        __mmask8 a_step_mask_m8 = _mm512_cmple_epu64_mask(a_vec.zmm, b_max_u64x8);
        __mmask8 b_step_mask_m8 = _mm512_cmple_epu64_mask(b_vec.zmm, a_max_u64x8);
        a += 32 - _lzcnt_u32((nk_u32_t)a_step_mask_m8);
        b += 32 - _lzcnt_u32((nk_u32_t)b_step_mask_m8);

        // Now we are likely to have some overlap, so we can intersect the registers
        __mmask8 a_matches_m8 = nk_intersect_u64x8_icelake_(a_vec.zmm, b_vec.zmm);

        // Export matches if result buffer is provided
        if (result) { _mm512_mask_compressstoreu_epi64(result + c, a_matches_m8, a_vec.zmm); }
        c += _mm_popcnt_u32(a_matches_m8); // MSVC has no `_popcnt32`
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u64_serial(a, b, a_end - a, b_end - b, result ? result + c : 0, &tail_count);
    *count = c + tail_count;
}

NK_API_COMPTIME void nk_sparse_dot_u32f32_icelake(        //
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
    __m512d product_low_f64x8 = _mm512_setzero_pd();
    __m512d product_high_f64x8 = _mm512_setzero_pd();
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
        __mmask16 a_step_mask_m16 = _mm512_cmple_epu32_mask(a_vec.zmm, b_max_u32x16);
        __mmask16 b_step_mask_m16 = _mm512_cmple_epu32_mask(b_vec.zmm, a_max_u32x16);
        nk_u32_t a_advance = 32 - _lzcnt_u32((nk_u32_t)a_step_mask_m16);
        nk_u32_t b_advance = 32 - _lzcnt_u32((nk_u32_t)b_step_mask_m16);

        // Now we are likely to have some overlap, so we can intersect the registers
        __mmask16 a_matches_m16 = nk_intersect_u32x16_icelake_(a_vec.zmm, b_vec.zmm);
        __mmask16 b_matches_m16 = nk_intersect_u32x16_icelake_(b_vec.zmm, a_vec.zmm);
        if (a_matches_m16) {
            // Load and compress matching weights at current position
            __m512 a_weights_f32x16 = _mm512_loadu_ps(a_weights);
            __m512 b_weights_f32x16 = _mm512_loadu_ps(b_weights);
            __m512 a_matched_f32x16 = _mm512_maskz_compress_ps(a_matches_m16, a_weights_f32x16);
            __m512 b_matched_f32x16 = _mm512_maskz_compress_ps(b_matches_m16, b_weights_f32x16);

            __m256 a_matched_low_f32x8 = _mm512_castps512_ps256(a_matched_f32x16);
            __m256 a_matched_high_f32x8 = _mm512_extractf32x8_ps(a_matched_f32x16, 1);
            __m256 b_matched_low_f32x8 = _mm512_castps512_ps256(b_matched_f32x16);
            __m256 b_matched_high_f32x8 = _mm512_extractf32x8_ps(b_matched_f32x16, 1);

            product_low_f64x8 = _mm512_fmadd_pd(_mm512_cvtps_pd(a_matched_low_f32x8),
                                                _mm512_cvtps_pd(b_matched_low_f32x8), product_low_f64x8);
            product_high_f64x8 = _mm512_fmadd_pd(_mm512_cvtps_pd(a_matched_high_f32x8),
                                                 _mm512_cvtps_pd(b_matched_high_f32x8), product_high_f64x8);
        }

        // Advance pointers after processing
        a += a_advance;
        a_weights += a_advance;
        b += b_advance;
        b_weights += b_advance;
    }

    nk_f64_t tail_product = 0;
    nk_sparse_dot_u32f32_serial(a, b, a_weights, b_weights, a_end - a, b_end - b, &tail_product);
    *product = _mm512_reduce_add_pd(product_low_f64x8) + _mm512_reduce_add_pd(product_high_f64x8) + tail_product;
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
#endif // NK_TARGET_X8664_
#endif // NK_SPARSE_ICELAKE_H
