/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for Sapphire Rapids.
 *  @file include/numkong/spatial/sapphire.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/spatial.h
 *
 *  Sapphire Rapids adds native FP16 support via AVX-512 FP16 extension.
 *  For e4m3 L2 distance, we can leverage F16 for the subtraction step:
 *  - e4m3 differences fit in F16 (max |a−b| = 896 < 65504)
 *  - But squared differences overflow F16 (896² = 802816 > 65504)
 *  - So: subtract in F16, convert to F32, then square and accumulate
 *
 *  For e2m3/e3m2 L2 distance, squared differences fit in FP16:
 *  - E2M3: max |a−b| = 15, max (a−b)² = 225 < 65504, flush cadence = 4 (conservative for uniformity)
 *  - E3M2: max |a−b| = 56, max (a−b)² = 3136 < 65504, flush cadence = 4
 *  So the entire sub+square+accumulate stays in FP16 with periodic F32 flush.
 *
 *  @section spatial_sapphire_instructions Relevant Instructions
 *
 *      Intrinsic                   Instruction                     Sapphire    Genoa
 *      _mm256_sub_ph               VSUBPH (YMM, YMM, YMM)          4cy @ p05   3cy @ p01
 *      _mm512_cvtph_ps             VCVTPH2PS (ZMM, YMM)            5cy @ p05   5cy @ p01
 *      _mm512_fmadd_ps             VFMADD (ZMM, ZMM, ZMM)          4cy @ p05   4cy @ p01
 *      _mm512_reduce_add_ps        (pseudo: VHADDPS chain)         ~8cy        ~8cy
 *      _mm_maskz_loadu_epi8        VMOVDQU8 (XMM {K}, M128)        7cy @ p23   7cy @ p23
 */
#ifndef NK_SPATIAL_SAPPHIRE_H
#define NK_SPATIAL_SAPPHIRE_H

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIRE

#include "numkong/types.h"
#include "numkong/cast/sapphire.h"   // `nk_e4m3x16_to_f16x16_sapphire_`
#include "numkong/dot/sapphire.h"    // `nk_e2m3x32_to_f16x32_sapphire_`, `nk_flush_f16_to_f32_sapphire_`
#include "numkong/spatial/haswell.h" // `nk_angular_normalize_f32_haswell_`, `nk_f32_sqrt_haswell`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC void nk_sqeuclidean_e4m3_sapphire(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars,
                                            nk_size_t count_scalars, nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero_ps();

    while (count_scalars > 0) {
        nk_size_t const n = count_scalars < 16 ? count_scalars : 16;
        __mmask16 const mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        __m128i a_e4m3x16 = _mm_maskz_loadu_epi8(mask, a_scalars);
        __m128i b_e4m3x16 = _mm_maskz_loadu_epi8(mask, b_scalars);

        // Convert e4m3 → f16
        __m256h a_f16x16 = nk_e4m3x16_to_f16x16_sapphire_(a_e4m3x16);
        __m256h b_f16x16 = nk_e4m3x16_to_f16x16_sapphire_(b_e4m3x16);

        // Subtract in F16 − differences fit (max 896 < 65504)
        __m256h diff_f16x16 = _mm256_sub_ph(a_f16x16, b_f16x16);

        // Convert to F32 before squaring (896² = 802816 overflows F16!)
        __m512 diff_f32x16 = _mm512_cvtph_ps(_mm256_castph_si256(diff_f16x16));

        // Square and accumulate in F32
        sum_f32x16 = _mm512_fmadd_ps(diff_f32x16, diff_f32x16, sum_f32x16);
        a_scalars += n, b_scalars += n, count_scalars -= n;
    }

    *result = _mm512_reduce_add_ps(sum_f32x16);
}

NK_PUBLIC void nk_euclidean_e4m3_sapphire(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars,
                                          nk_size_t count_scalars, nk_f32_t *result) {
    nk_sqeuclidean_e4m3_sapphire(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_PUBLIC void nk_sqeuclidean_e2m3_sapphire(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars,
                                            nk_size_t count_scalars, nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero_ps();

    // Main loop: 4-way unrolled, 128 elements per flush
    while (count_scalars >= 128) {
        __m512h acc_f16x32 = _mm512_setzero_ph();
        __m512h a_f16x32, b_f16x32, diff_f16x32;
        // Iteration 1
        a_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_loadu_epi8(a_scalars));
        b_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_loadu_epi8(b_scalars));
        diff_f16x32 = _mm512_sub_ph(a_f16x32, b_f16x32);
        acc_f16x32 = _mm512_fmadd_ph(diff_f16x32, diff_f16x32, acc_f16x32);
        // Iteration 2
        a_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_loadu_epi8(a_scalars + 32));
        b_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_loadu_epi8(b_scalars + 32));
        diff_f16x32 = _mm512_sub_ph(a_f16x32, b_f16x32);
        acc_f16x32 = _mm512_fmadd_ph(diff_f16x32, diff_f16x32, acc_f16x32);
        // Iteration 3
        a_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_loadu_epi8(a_scalars + 64));
        b_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_loadu_epi8(b_scalars + 64));
        diff_f16x32 = _mm512_sub_ph(a_f16x32, b_f16x32);
        acc_f16x32 = _mm512_fmadd_ph(diff_f16x32, diff_f16x32, acc_f16x32);
        // Iteration 4
        a_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_loadu_epi8(a_scalars + 96));
        b_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_loadu_epi8(b_scalars + 96));
        diff_f16x32 = _mm512_sub_ph(a_f16x32, b_f16x32);
        acc_f16x32 = _mm512_fmadd_ph(diff_f16x32, diff_f16x32, acc_f16x32);
        // Flush to F32
        sum_f32x16 = nk_flush_f16_to_f32_sapphire_(acc_f16x32, sum_f32x16);
        a_scalars += 128, b_scalars += 128, count_scalars -= 128;
    }

    // Tail: remaining 0–127 elements, 32 at a time via masked loads
    __m512h acc_f16x32 = _mm512_setzero_ph();
    while (count_scalars > 0) {
        nk_size_t const n = count_scalars < 32 ? count_scalars : 32;
        __mmask32 const mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        __m512h a_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_maskz_loadu_epi8(mask, a_scalars));
        __m512h b_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_maskz_loadu_epi8(mask, b_scalars));
        __m512h diff_f16x32 = _mm512_sub_ph(a_f16x32, b_f16x32);
        acc_f16x32 = _mm512_fmadd_ph(diff_f16x32, diff_f16x32, acc_f16x32);
        a_scalars += n, b_scalars += n, count_scalars -= n;
    }
    sum_f32x16 = nk_flush_f16_to_f32_sapphire_(acc_f16x32, sum_f32x16);

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_sqeuclidean_e3m2_sapphire(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars,
                                            nk_size_t count_scalars, nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero_ps();

    // Main loop: 4-way unrolled, 128 elements per flush
    while (count_scalars >= 128) {
        __m512h acc_f16x32 = _mm512_setzero_ph();
        __m512h a_f16x32, b_f16x32, diff_f16x32;
        // Iteration 1
        a_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_loadu_epi8(a_scalars));
        b_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_loadu_epi8(b_scalars));
        diff_f16x32 = _mm512_sub_ph(a_f16x32, b_f16x32);
        acc_f16x32 = _mm512_fmadd_ph(diff_f16x32, diff_f16x32, acc_f16x32);
        // Iteration 2
        a_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_loadu_epi8(a_scalars + 32));
        b_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_loadu_epi8(b_scalars + 32));
        diff_f16x32 = _mm512_sub_ph(a_f16x32, b_f16x32);
        acc_f16x32 = _mm512_fmadd_ph(diff_f16x32, diff_f16x32, acc_f16x32);
        // Iteration 3
        a_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_loadu_epi8(a_scalars + 64));
        b_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_loadu_epi8(b_scalars + 64));
        diff_f16x32 = _mm512_sub_ph(a_f16x32, b_f16x32);
        acc_f16x32 = _mm512_fmadd_ph(diff_f16x32, diff_f16x32, acc_f16x32);
        // Iteration 4
        a_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_loadu_epi8(a_scalars + 96));
        b_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_loadu_epi8(b_scalars + 96));
        diff_f16x32 = _mm512_sub_ph(a_f16x32, b_f16x32);
        acc_f16x32 = _mm512_fmadd_ph(diff_f16x32, diff_f16x32, acc_f16x32);
        // Flush to F32
        sum_f32x16 = nk_flush_f16_to_f32_sapphire_(acc_f16x32, sum_f32x16);
        a_scalars += 128, b_scalars += 128, count_scalars -= 128;
    }

    // Tail: remaining 0–127 elements, 32 at a time via masked loads
    __m512h acc_f16x32 = _mm512_setzero_ph();
    while (count_scalars > 0) {
        nk_size_t const n = count_scalars < 32 ? count_scalars : 32;
        __mmask32 const mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        __m512h a_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_maskz_loadu_epi8(mask, a_scalars));
        __m512h b_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_maskz_loadu_epi8(mask, b_scalars));
        __m512h diff_f16x32 = _mm512_sub_ph(a_f16x32, b_f16x32);
        acc_f16x32 = _mm512_fmadd_ph(diff_f16x32, diff_f16x32, acc_f16x32);
        a_scalars += n, b_scalars += n, count_scalars -= n;
    }
    sum_f32x16 = nk_flush_f16_to_f32_sapphire_(acc_f16x32, sum_f32x16);

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_euclidean_e2m3_sapphire(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars,
                                          nk_size_t count_scalars, nk_f32_t *result) {
    nk_sqeuclidean_e2m3_sapphire(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_PUBLIC void nk_euclidean_e3m2_sapphire(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars,
                                          nk_size_t count_scalars, nk_f32_t *result) {
    nk_sqeuclidean_e3m2_sapphire(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_PUBLIC void nk_angular_e2m3_sapphire(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                                        nk_f32_t *result) {
    __m512 sum_dot_f32x16 = _mm512_setzero_ps();
    __m512 sum_a_f32x16 = _mm512_setzero_ps();
    __m512 sum_b_f32x16 = _mm512_setzero_ps();

    // Main loop: 4-way unrolled, 128 elements per flush
    while (count_scalars >= 128) {
        __m512h dot_acc = _mm512_setzero_ph();
        __m512h a_norm_acc = _mm512_setzero_ph();
        __m512h b_norm_acc = _mm512_setzero_ph();
        __m512h a_f16x32, b_f16x32;
        // Iteration 1
        a_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_loadu_epi8(a_scalars));
        b_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_loadu_epi8(b_scalars));
        dot_acc = _mm512_fmadd_ph(a_f16x32, b_f16x32, dot_acc);
        a_norm_acc = _mm512_fmadd_ph(a_f16x32, a_f16x32, a_norm_acc);
        b_norm_acc = _mm512_fmadd_ph(b_f16x32, b_f16x32, b_norm_acc);
        // Iteration 2
        a_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_loadu_epi8(a_scalars + 32));
        b_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_loadu_epi8(b_scalars + 32));
        dot_acc = _mm512_fmadd_ph(a_f16x32, b_f16x32, dot_acc);
        a_norm_acc = _mm512_fmadd_ph(a_f16x32, a_f16x32, a_norm_acc);
        b_norm_acc = _mm512_fmadd_ph(b_f16x32, b_f16x32, b_norm_acc);
        // Iteration 3
        a_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_loadu_epi8(a_scalars + 64));
        b_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_loadu_epi8(b_scalars + 64));
        dot_acc = _mm512_fmadd_ph(a_f16x32, b_f16x32, dot_acc);
        a_norm_acc = _mm512_fmadd_ph(a_f16x32, a_f16x32, a_norm_acc);
        b_norm_acc = _mm512_fmadd_ph(b_f16x32, b_f16x32, b_norm_acc);
        // Iteration 4
        a_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_loadu_epi8(a_scalars + 96));
        b_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_loadu_epi8(b_scalars + 96));
        dot_acc = _mm512_fmadd_ph(a_f16x32, b_f16x32, dot_acc);
        a_norm_acc = _mm512_fmadd_ph(a_f16x32, a_f16x32, a_norm_acc);
        b_norm_acc = _mm512_fmadd_ph(b_f16x32, b_f16x32, b_norm_acc);
        // Flush to F32
        sum_dot_f32x16 = nk_flush_f16_to_f32_sapphire_(dot_acc, sum_dot_f32x16);
        sum_a_f32x16 = nk_flush_f16_to_f32_sapphire_(a_norm_acc, sum_a_f32x16);
        sum_b_f32x16 = nk_flush_f16_to_f32_sapphire_(b_norm_acc, sum_b_f32x16);
        a_scalars += 128, b_scalars += 128, count_scalars -= 128;
    }

    // Tail: remaining 0–127 elements, 32 at a time via masked loads
    __m512h dot_acc = _mm512_setzero_ph();
    __m512h a_norm_acc = _mm512_setzero_ph();
    __m512h b_norm_acc = _mm512_setzero_ph();
    while (count_scalars > 0) {
        nk_size_t const n = count_scalars < 32 ? count_scalars : 32;
        __mmask32 const mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        __m512h a_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_maskz_loadu_epi8(mask, a_scalars));
        __m512h b_f16x32 = nk_e2m3x32_to_f16x32_sapphire_(_mm256_maskz_loadu_epi8(mask, b_scalars));
        dot_acc = _mm512_fmadd_ph(a_f16x32, b_f16x32, dot_acc);
        a_norm_acc = _mm512_fmadd_ph(a_f16x32, a_f16x32, a_norm_acc);
        b_norm_acc = _mm512_fmadd_ph(b_f16x32, b_f16x32, b_norm_acc);
        a_scalars += n, b_scalars += n, count_scalars -= n;
    }
    sum_dot_f32x16 = nk_flush_f16_to_f32_sapphire_(dot_acc, sum_dot_f32x16);
    sum_a_f32x16 = nk_flush_f16_to_f32_sapphire_(a_norm_acc, sum_a_f32x16);
    sum_b_f32x16 = nk_flush_f16_to_f32_sapphire_(b_norm_acc, sum_b_f32x16);

    nk_f32_t dot_f32 = nk_reduce_add_f32x16_skylake_(sum_dot_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(sum_a_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(sum_b_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_PUBLIC void nk_angular_e3m2_sapphire(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars, nk_size_t count_scalars,
                                        nk_f32_t *result) {
    __m512 sum_dot_f32x16 = _mm512_setzero_ps();
    __m512 sum_a_f32x16 = _mm512_setzero_ps();
    __m512 sum_b_f32x16 = _mm512_setzero_ps();

    // Main loop: 4-way unrolled, 128 elements per flush
    while (count_scalars >= 128) {
        __m512h dot_acc = _mm512_setzero_ph();
        __m512h a_norm_acc = _mm512_setzero_ph();
        __m512h b_norm_acc = _mm512_setzero_ph();
        __m512h a_f16x32, b_f16x32;
        // Iteration 1
        a_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_loadu_epi8(a_scalars));
        b_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_loadu_epi8(b_scalars));
        dot_acc = _mm512_fmadd_ph(a_f16x32, b_f16x32, dot_acc);
        a_norm_acc = _mm512_fmadd_ph(a_f16x32, a_f16x32, a_norm_acc);
        b_norm_acc = _mm512_fmadd_ph(b_f16x32, b_f16x32, b_norm_acc);
        // Iteration 2
        a_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_loadu_epi8(a_scalars + 32));
        b_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_loadu_epi8(b_scalars + 32));
        dot_acc = _mm512_fmadd_ph(a_f16x32, b_f16x32, dot_acc);
        a_norm_acc = _mm512_fmadd_ph(a_f16x32, a_f16x32, a_norm_acc);
        b_norm_acc = _mm512_fmadd_ph(b_f16x32, b_f16x32, b_norm_acc);
        // Iteration 3
        a_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_loadu_epi8(a_scalars + 64));
        b_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_loadu_epi8(b_scalars + 64));
        dot_acc = _mm512_fmadd_ph(a_f16x32, b_f16x32, dot_acc);
        a_norm_acc = _mm512_fmadd_ph(a_f16x32, a_f16x32, a_norm_acc);
        b_norm_acc = _mm512_fmadd_ph(b_f16x32, b_f16x32, b_norm_acc);
        // Iteration 4
        a_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_loadu_epi8(a_scalars + 96));
        b_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_loadu_epi8(b_scalars + 96));
        dot_acc = _mm512_fmadd_ph(a_f16x32, b_f16x32, dot_acc);
        a_norm_acc = _mm512_fmadd_ph(a_f16x32, a_f16x32, a_norm_acc);
        b_norm_acc = _mm512_fmadd_ph(b_f16x32, b_f16x32, b_norm_acc);
        // Flush to F32
        sum_dot_f32x16 = nk_flush_f16_to_f32_sapphire_(dot_acc, sum_dot_f32x16);
        sum_a_f32x16 = nk_flush_f16_to_f32_sapphire_(a_norm_acc, sum_a_f32x16);
        sum_b_f32x16 = nk_flush_f16_to_f32_sapphire_(b_norm_acc, sum_b_f32x16);
        a_scalars += 128, b_scalars += 128, count_scalars -= 128;
    }

    // Tail: remaining 0–127 elements, 32 at a time via masked loads
    __m512h dot_acc = _mm512_setzero_ph();
    __m512h a_norm_acc = _mm512_setzero_ph();
    __m512h b_norm_acc = _mm512_setzero_ph();
    while (count_scalars > 0) {
        nk_size_t const n = count_scalars < 32 ? count_scalars : 32;
        __mmask32 const mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        __m512h a_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_maskz_loadu_epi8(mask, a_scalars));
        __m512h b_f16x32 = nk_e3m2x32_to_f16x32_sapphire_(_mm256_maskz_loadu_epi8(mask, b_scalars));
        dot_acc = _mm512_fmadd_ph(a_f16x32, b_f16x32, dot_acc);
        a_norm_acc = _mm512_fmadd_ph(a_f16x32, a_f16x32, a_norm_acc);
        b_norm_acc = _mm512_fmadd_ph(b_f16x32, b_f16x32, b_norm_acc);
        a_scalars += n, b_scalars += n, count_scalars -= n;
    }
    sum_dot_f32x16 = nk_flush_f16_to_f32_sapphire_(dot_acc, sum_dot_f32x16);
    sum_a_f32x16 = nk_flush_f16_to_f32_sapphire_(a_norm_acc, sum_a_f32x16);
    sum_b_f32x16 = nk_flush_f16_to_f32_sapphire_(b_norm_acc, sum_b_f32x16);

    nk_f32_t dot_f32 = nk_reduce_add_f32x16_skylake_(sum_dot_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(sum_a_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(sum_b_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_f32, a_norm_sq_f32, b_norm_sq_f32);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SAPPHIRE
#endif // NK_TARGET_X86_
#endif // NK_SPATIAL_SAPPHIRE_H
