/**
 *  @brief SIMD-accelerated Elementwise Arithmetic for Skylake.
 *  @file include/numkong/each/skylake.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/each.h
 *
 *  @section skylake_elementwise_instructions Relevant Instructions
 *
 *      Intrinsic                   Instruction                     SKL         ICL         Genoa
 *      _mm512_add_ps               VADDPS (ZMM, ZMM, ZMM)          4cy @ p05   4cy @ p0    3cy @ p01
 *      _mm512_fmadd_ps             VFMADD132PS (ZMM, ZMM, ZMM)     4cy @ p05   4cy @ p0    4cy @ p01
 *      _mm512_mul_ps               VMULPS (ZMM, ZMM, ZMM)          4cy @ p05   4cy @ p0    3cy @ p01
 *      _mm512_cvtph_ps             VCVTPH2PS (ZMM, YMM)            5cy @ p05   7cy @ p0    5cy @ p01
 *      _mm512_maskz_loadu_ps       VMOVUPS (ZMM {K}, M512)         7cy @ p23   7cy @ p23   7cy @ p23
 *      _mm512_mask_storeu_ps       VMOVUPS (M512 {K}, ZMM)         4cy @ p4    4cy @ p4    4cy @ p4
 *
 *  Skylake-X server chips have dual 512-bit FMA units enabling 0.5cy throughput for arithmetic operations.
 *  AVX-512 masked loads and stores eliminate branch misprediction penalties for partial vector processing.
 *  Note that client Skylake chips may throttle frequency when executing 512-bit instructions continuously.
 */
#ifndef NK_EACH_SKYLAKE_H
#define NK_EACH_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE

#include "numkong/types.h"
#include "numkong/cast/skylake.h" // `nk_e4m3x16_to_f32x16_skylake_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC void nk_each_sum_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    __m512d a_vec, b_vec, sum_vec;
    __mmask8 mask = 0xFF;
nk_each_sum_f64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_pd(mask, a);
        b_vec = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_pd(a);
        b_vec = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    sum_vec = _mm512_add_pd(a_vec, b_vec);
    _mm512_mask_storeu_pd(result, mask, sum_vec);
    result += 8;
    if (n) goto nk_each_sum_f64_skylake_cycle;
}

NK_PUBLIC void nk_each_scale_f64_skylake(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                         nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512d a_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
nk_each_scale_f64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        a += 8, n -= 8;
    }
    result_f64x8 = _mm512_fmadd_pd(a_f64x8, alpha_f64x8, beta_f64x8);
    _mm512_mask_storeu_pd(result, mask, result_f64x8);
    result += 8;
    if (n) goto nk_each_scale_f64_skylake_cycle;
}

NK_PUBLIC void nk_each_blend_f64_skylake(              //
    nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // There are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_each_sum_f64_skylake(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f64_t zero = 0;
        if (beta_val == 0) { nk_each_scale_f64_skylake(a, n, alpha, &zero, result); }
        else { nk_each_scale_f64_skylake(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512d a_f64x8, b_f64x8, a_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
nk_each_blend_f64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        b_f64x8 = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    a_scaled_f64x8 = _mm512_mul_pd(a_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(b_f64x8, beta_f64x8, a_scaled_f64x8);
    _mm512_mask_storeu_pd(result, mask, result_f64x8);
    result += 8;
    if (n) goto nk_each_blend_f64_skylake_cycle;
}

NK_PUBLIC void nk_each_sum_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 a_vec, b_vec, sum_vec;
    __mmask16 mask = 0xFFFF;

nk_each_sum_f32_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_ps(mask, a);
        b_vec = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_ps(a);
        b_vec = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    sum_vec = _mm512_add_ps(a_vec, b_vec);
    _mm512_mask_storeu_ps(result, mask, sum_vec);
    result += 16;
    if (n) goto nk_each_sum_f32_skylake_cycle;
}

NK_PUBLIC void nk_each_scale_f32_skylake(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                         nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;

nk_each_scale_f32_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, a);
        n = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps(a);
        a += 16, n -= 16;
    }
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    _mm512_mask_storeu_ps(result, mask, result_f32x16);
    result += 16;
    if (n) goto nk_each_scale_f32_skylake_cycle;
}

NK_PUBLIC void nk_each_blend_f32_skylake(              //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_each_sum_f32_skylake(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_f32_skylake(a, n, alpha, &zero, result); }
        else { nk_each_scale_f32_skylake(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m512 a_f32x16, b_f32x16, a_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_each_blend_f32_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, a);
        b_f32x16 = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps(a);
        b_f32x16 = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    a_scaled_f32x16 = _mm512_mul_ps(a_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(b_f32x16, beta_f32x16, a_scaled_f32x16);
    _mm512_mask_storeu_ps(result, mask, result_f32x16);
    result += 16;
    if (n) goto nk_each_blend_f32_skylake_cycle;
}

NK_PUBLIC void nk_each_sum_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result) {
    __m256i a_bf16_vec, b_bf16_vec, sum_bf16_vec;
    __m512 a_vec, b_vec, sum_vec;
    __mmask16 mask = 0xFFFF;
nk_each_sum_bf16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16_vec = _mm256_maskz_loadu_epi16(mask, a);
        b_bf16_vec = _mm256_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_bf16_vec = _mm256_loadu_epi16(a);
        b_bf16_vec = _mm256_loadu_epi16(b);
        a += 16, b += 16, n -= 16;
    }
    a_vec = nk_bf16x16_to_f32x16_skylake_(a_bf16_vec);
    b_vec = nk_bf16x16_to_f32x16_skylake_(b_bf16_vec);
    sum_vec = _mm512_add_ps(a_vec, b_vec);
    sum_bf16_vec = nk_f32x16_to_bf16x16_skylake_(sum_vec);
    _mm256_mask_storeu_epi16(result, mask, sum_bf16_vec);
    result += 16;
    if (n) goto nk_each_sum_bf16_skylake_cycle;
}

NK_PUBLIC void nk_each_scale_bf16_skylake(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                          nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m256i a_bf16x16, result_bf16x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_each_scale_bf16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x16 = _mm256_maskz_loadu_epi16(mask, a);
        n = 0;
    }
    else {
        a_bf16x16 = _mm256_loadu_epi16(a);
        a += 16, n -= 16;
    }
    a_f32x16 = nk_bf16x16_to_f32x16_skylake_(a_bf16x16);
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_bf16x16 = nk_f32x16_to_bf16x16_skylake_(result_f32x16);
    _mm256_mask_storeu_epi16(result, mask, result_bf16x16);
    result += 16;
    if (n) goto nk_each_scale_bf16_skylake_cycle;
}

NK_PUBLIC void nk_each_blend_bf16_skylake(               //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_each_sum_bf16_skylake(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_bf16_skylake(a, n, alpha, &zero, result); }
        else { nk_each_scale_bf16_skylake(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m256i a_bf16x16, b_bf16x16, result_bf16x16;
    __m512 a_f32x16, b_f32x16, a_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_each_blend_bf16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x16 = _mm256_maskz_loadu_epi16(mask, a);
        b_bf16x16 = _mm256_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_bf16x16 = _mm256_loadu_epi16(a);
        b_bf16x16 = _mm256_loadu_epi16(b);
        a += 16, b += 16, n -= 16;
    }
    a_f32x16 = nk_bf16x16_to_f32x16_skylake_(a_bf16x16);
    b_f32x16 = nk_bf16x16_to_f32x16_skylake_(b_bf16x16);
    a_scaled_f32x16 = _mm512_mul_ps(a_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(b_f32x16, beta_f32x16, a_scaled_f32x16);
    result_bf16x16 = nk_f32x16_to_bf16x16_skylake_(result_f32x16);
    _mm256_mask_storeu_epi16(result, mask, result_bf16x16);
    result += 16;
    if (n) goto nk_each_blend_bf16_skylake_cycle;
}

NK_PUBLIC void nk_each_fma_f64_skylake(                                   //
    nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512d a_f64x8, b_f64x8, c_f64x8, ab_f64x8, ab_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
nk_each_fma_f64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b);
        c_f64x8 = _mm512_maskz_loadu_pd(mask, c);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        b_f64x8 = _mm512_loadu_pd(b);
        c_f64x8 = _mm512_loadu_pd(c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    ab_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    ab_scaled_f64x8 = _mm512_mul_pd(ab_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(c_f64x8, beta_f64x8, ab_scaled_f64x8);
    _mm512_mask_storeu_pd(result, mask, result_f64x8);
    result += 8;
    if (n) goto nk_each_fma_f64_skylake_cycle;
}

NK_PUBLIC void nk_each_fma_f32_skylake(                                   //
    nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_each_fma_f32_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, a);
        b_f32x16 = _mm512_maskz_loadu_ps(mask, b);
        c_f32x16 = _mm512_maskz_loadu_ps(mask, c);
        n = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps(a);
        b_f32x16 = _mm512_loadu_ps(b);
        c_f32x16 = _mm512_loadu_ps(c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    _mm512_mask_storeu_ps(result, mask, result_f32x16);
    result += 16;
    if (n) goto nk_each_fma_f32_skylake_cycle;
}

NK_PUBLIC void nk_each_fma_bf16_skylake(                                     //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m256i a_bf16x16, b_bf16x16, c_bf16x16, result_bf16x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_each_fma_bf16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x16 = _mm256_maskz_loadu_epi16(mask, a);
        b_bf16x16 = _mm256_maskz_loadu_epi16(mask, b);
        c_bf16x16 = _mm256_maskz_loadu_epi16(mask, c);
        n = 0;
    }
    else {
        a_bf16x16 = _mm256_loadu_epi16(a);
        b_bf16x16 = _mm256_loadu_epi16(b);
        c_bf16x16 = _mm256_loadu_epi16(c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = nk_bf16x16_to_f32x16_skylake_(a_bf16x16);
    b_f32x16 = nk_bf16x16_to_f32x16_skylake_(b_bf16x16);
    c_f32x16 = nk_bf16x16_to_f32x16_skylake_(c_bf16x16);
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_bf16x16 = nk_f32x16_to_bf16x16_skylake_(result_f32x16);
    _mm256_mask_storeu_epi16(result, mask, result_bf16x16);
    result += 16;
    if (n) goto nk_each_fma_bf16_skylake_cycle;
}

NK_PUBLIC void nk_each_scale_i8_skylake(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                        nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m128i a_i8x16, result_i8x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_i32x16;
    __m512i min_i32x16 = _mm512_set1_epi32(-128);
    __m512i max_i32x16 = _mm512_set1_epi32(127);

nk_each_scale_i8_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i8x16 = _mm_maskz_loadu_epi8(mask, a);
        n = 0;
    }
    else {
        a_i8x16 = _mm_loadu_si128((__m128i *)a);
        a += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a_i8x16));
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_i32x16 = _mm512_cvtps_epi32(result_f32x16);
    result_i32x16 = _mm512_max_epi32(result_i32x16, min_i32x16);
    result_i32x16 = _mm512_min_epi32(result_i32x16, max_i32x16);
    result_i8x16 = _mm512_cvtepi32_epi8(result_i32x16);
    _mm_mask_storeu_epi8(result, mask, result_i8x16);
    result += 16;
    if (n) goto nk_each_scale_i8_skylake_cycle;
}

NK_PUBLIC void nk_each_fma_i8_skylake(                                 //
    nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m128i a_i8x16, b_i8x16, c_i8x16, result_i8x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_i32x16;
    __m512i min_i32x16 = _mm512_set1_epi32(-128);
    __m512i max_i32x16 = _mm512_set1_epi32(127);

nk_each_fma_i8_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i8x16 = _mm_maskz_loadu_epi8(mask, a);
        b_i8x16 = _mm_maskz_loadu_epi8(mask, b);
        c_i8x16 = _mm_maskz_loadu_epi8(mask, c);
        n = 0;
    }
    else {
        a_i8x16 = _mm_loadu_si128((__m128i *)a);
        b_i8x16 = _mm_loadu_si128((__m128i *)b);
        c_i8x16 = _mm_loadu_si128((__m128i *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a_i8x16));
    b_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b_i8x16));
    c_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(c_i8x16));
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_i32x16 = _mm512_cvtps_epi32(result_f32x16);
    result_i32x16 = _mm512_max_epi32(result_i32x16, min_i32x16);
    result_i32x16 = _mm512_min_epi32(result_i32x16, max_i32x16);
    result_i8x16 = _mm512_cvtepi32_epi8(result_i32x16);
    _mm_mask_storeu_epi8(result, mask, result_i8x16);
    result += 16;
    if (n) goto nk_each_fma_i8_skylake_cycle;
}

NK_PUBLIC void nk_each_scale_u8_skylake(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                        nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m128i a_u8x16, result_u8x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_u32x16;
    __m512i min_u32x16 = _mm512_set1_epi32(0);
    __m512i max_u32x16 = _mm512_set1_epi32(255);

nk_each_scale_u8_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u8x16 = _mm_maskz_loadu_epi8(mask, a);
        n = 0;
    }
    else {
        a_u8x16 = _mm_loadu_si128((__m128i *)a);
        a += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(a_u8x16));
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_u32x16 = _mm512_cvtps_epu32(result_f32x16);
    result_u32x16 = _mm512_max_epu32(result_u32x16, min_u32x16);
    result_u32x16 = _mm512_min_epu32(result_u32x16, max_u32x16);
    result_u8x16 = _mm512_cvtepi32_epi8(result_u32x16);
    _mm_mask_storeu_epi8(result, mask, result_u8x16);
    result += 16;
    if (n) goto nk_each_scale_u8_skylake_cycle;
}

NK_PUBLIC void nk_each_fma_u8_skylake(                                 //
    nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m128i a_u8x16, b_u8x16, c_u8x16, result_u8x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_u32x16;
    __m512i min_u32x16 = _mm512_set1_epi32(0);
    __m512i max_u32x16 = _mm512_set1_epi32(255);

nk_each_fma_u8_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u8x16 = _mm_maskz_loadu_epi8(mask, a);
        b_u8x16 = _mm_maskz_loadu_epi8(mask, b);
        c_u8x16 = _mm_maskz_loadu_epi8(mask, c);
        n = 0;
    }
    else {
        a_u8x16 = _mm_loadu_si128((__m128i *)a);
        b_u8x16 = _mm_loadu_si128((__m128i *)b);
        c_u8x16 = _mm_loadu_si128((__m128i *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(a_u8x16));
    b_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b_u8x16));
    c_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c_u8x16));
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_u32x16 = _mm512_cvtps_epu32(result_f32x16);
    result_u32x16 = _mm512_max_epu32(result_u32x16, min_u32x16);
    result_u32x16 = _mm512_min_epu32(result_u32x16, max_u32x16);
    result_u8x16 = _mm512_cvtepi32_epi8(result_u32x16);
    _mm_mask_storeu_epi8(result, mask, result_u8x16);
    result += 16;
    if (n) goto nk_each_fma_u8_skylake_cycle;
}

NK_PUBLIC void nk_each_scale_i16_skylake(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                         nk_i16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_f32);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_f32);
    __m256i a_i16x16, result_i16x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_i32x16;
    __m512 min_f32x16 = _mm512_set1_ps(-32768.0f);
    __m512 max_f32x16 = _mm512_set1_ps(32767.0f);

nk_each_scale_i16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i16x16 = _mm256_maskz_loadu_epi16(mask, a);
        n = 0;
    }
    else {
        a_i16x16 = _mm256_loadu_si256((__m256i *)a);
        a += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(a_i16x16));
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_f32x16 = _mm512_max_ps(result_f32x16, min_f32x16);
    result_f32x16 = _mm512_min_ps(result_f32x16, max_f32x16);
    result_i32x16 = _mm512_cvtps_epi32(result_f32x16);
    result_i16x16 = _mm512_cvtepi32_epi16(result_i32x16);
    _mm256_mask_storeu_epi16(result, mask, result_i16x16);
    result += 16;
    if (n) goto nk_each_scale_i16_skylake_cycle;
}

NK_PUBLIC void nk_each_fma_i16_skylake(                                   //
    nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_f32);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_f32);
    __m256i a_i16x16, b_i16x16, c_i16x16, result_i16x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_i32x16;
    __m512 min_f32x16 = _mm512_set1_ps(-32768.0f);
    __m512 max_f32x16 = _mm512_set1_ps(32767.0f);

nk_each_fma_i16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i16x16 = _mm256_maskz_loadu_epi16(mask, a);
        b_i16x16 = _mm256_maskz_loadu_epi16(mask, b);
        c_i16x16 = _mm256_maskz_loadu_epi16(mask, c);
        n = 0;
    }
    else {
        a_i16x16 = _mm256_loadu_si256((__m256i *)a);
        b_i16x16 = _mm256_loadu_si256((__m256i *)b);
        c_i16x16 = _mm256_loadu_si256((__m256i *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(a_i16x16));
    b_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(b_i16x16));
    c_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c_i16x16));
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_f32x16 = _mm512_max_ps(result_f32x16, min_f32x16);
    result_f32x16 = _mm512_min_ps(result_f32x16, max_f32x16);
    result_i32x16 = _mm512_cvtps_epi32(result_f32x16);
    result_i16x16 = _mm512_cvtepi32_epi16(result_i32x16);
    _mm256_mask_storeu_epi16(result, mask, result_i16x16);
    result += 16;
    if (n) goto nk_each_fma_i16_skylake_cycle;
}

NK_PUBLIC void nk_each_scale_u16_skylake(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                         nk_u16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_f32);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_f32);
    __m256i a_u16x16, result_u16x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_u32x16;
    __m512 min_f32x16 = _mm512_setzero_ps();
    __m512 max_f32x16 = _mm512_set1_ps(65535.0f);

nk_each_scale_u16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u16x16 = _mm256_maskz_loadu_epi16(mask, a);
        n = 0;
    }
    else {
        a_u16x16 = _mm256_loadu_si256((__m256i *)a);
        a += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(a_u16x16));
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_f32x16 = _mm512_max_ps(result_f32x16, min_f32x16);
    result_f32x16 = _mm512_min_ps(result_f32x16, max_f32x16);
    result_u32x16 = _mm512_cvtps_epu32(result_f32x16);
    result_u16x16 = _mm512_cvtepi32_epi16(result_u32x16);
    _mm256_mask_storeu_epi16(result, mask, result_u16x16);
    result += 16;
    if (n) goto nk_each_scale_u16_skylake_cycle;
}

NK_PUBLIC void nk_each_fma_u16_skylake(                                   //
    nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_f32);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_f32);
    __m256i a_u16x16, b_u16x16, c_u16x16, result_u16x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_u32x16;
    __m512 min_f32x16 = _mm512_setzero_ps();
    __m512 max_f32x16 = _mm512_set1_ps(65535.0f);

nk_each_fma_u16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u16x16 = _mm256_maskz_loadu_epi16(mask, a);
        b_u16x16 = _mm256_maskz_loadu_epi16(mask, b);
        c_u16x16 = _mm256_maskz_loadu_epi16(mask, c);
        n = 0;
    }
    else {
        a_u16x16 = _mm256_loadu_si256((__m256i *)a);
        b_u16x16 = _mm256_loadu_si256((__m256i *)b);
        c_u16x16 = _mm256_loadu_si256((__m256i *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(a_u16x16));
    b_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(b_u16x16));
    c_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(c_u16x16));
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_f32x16 = _mm512_max_ps(result_f32x16, min_f32x16);
    result_f32x16 = _mm512_min_ps(result_f32x16, max_f32x16);
    result_u32x16 = _mm512_cvtps_epu32(result_f32x16);
    result_u16x16 = _mm512_cvtepi32_epi16(result_u32x16);
    _mm256_mask_storeu_epi16(result, mask, result_u16x16);
    result += 16;
    if (n) goto nk_each_fma_u16_skylake_cycle;
}

NK_PUBLIC void nk_each_scale_i32_skylake(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                         nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m256i a_i32x8, result_i32x8;
    __m512d a_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
    __m512d min_f64x8 = _mm512_set1_pd(-2147483648.0);
    __m512d max_f64x8 = _mm512_set1_pd(2147483647.0);

nk_each_scale_i32_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i32x8 = _mm256_maskz_loadu_epi32(mask, a);
        n = 0;
    }
    else {
        a_i32x8 = _mm256_loadu_si256((__m256i *)a);
        a += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepi32_pd(a_i32x8);
    result_f64x8 = _mm512_fmadd_pd(a_f64x8, alpha_f64x8, beta_f64x8);
    result_f64x8 = _mm512_max_pd(result_f64x8, min_f64x8);
    result_f64x8 = _mm512_min_pd(result_f64x8, max_f64x8);
    result_i32x8 = _mm512_cvtpd_epi32(result_f64x8);
    _mm256_mask_storeu_epi32(result, mask, result_i32x8);
    result += 8;
    if (n) goto nk_each_scale_i32_skylake_cycle;
}

NK_PUBLIC void nk_each_fma_i32_skylake(                                   //
    nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m256i a_i32x8, b_i32x8, c_i32x8, result_i32x8;
    __m512d a_f64x8, b_f64x8, c_f64x8, ab_f64x8, ab_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
    __m512d min_f64x8 = _mm512_set1_pd(-2147483648.0);
    __m512d max_f64x8 = _mm512_set1_pd(2147483647.0);

nk_each_fma_i32_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i32x8 = _mm256_maskz_loadu_epi32(mask, a);
        b_i32x8 = _mm256_maskz_loadu_epi32(mask, b);
        c_i32x8 = _mm256_maskz_loadu_epi32(mask, c);
        n = 0;
    }
    else {
        a_i32x8 = _mm256_loadu_si256((__m256i *)a);
        b_i32x8 = _mm256_loadu_si256((__m256i *)b);
        c_i32x8 = _mm256_loadu_si256((__m256i *)c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepi32_pd(a_i32x8);
    b_f64x8 = _mm512_cvtepi32_pd(b_i32x8);
    c_f64x8 = _mm512_cvtepi32_pd(c_i32x8);
    ab_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    ab_scaled_f64x8 = _mm512_mul_pd(ab_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(c_f64x8, beta_f64x8, ab_scaled_f64x8);
    result_f64x8 = _mm512_max_pd(result_f64x8, min_f64x8);
    result_f64x8 = _mm512_min_pd(result_f64x8, max_f64x8);
    result_i32x8 = _mm512_cvtpd_epi32(result_f64x8);
    _mm256_mask_storeu_epi32(result, mask, result_i32x8);
    result += 8;
    if (n) goto nk_each_fma_i32_skylake_cycle;
}

NK_PUBLIC void nk_each_scale_u32_skylake(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                         nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m256i a_u32x8, result_u32x8;
    __m512d a_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
    __m512d min_f64x8 = _mm512_set1_pd(0.0);
    __m512d max_f64x8 = _mm512_set1_pd(4294967295.0);

nk_each_scale_u32_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u32x8 = _mm256_maskz_loadu_epi32(mask, a);
        n = 0;
    }
    else {
        a_u32x8 = _mm256_loadu_si256((__m256i *)a);
        a += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepu32_pd(a_u32x8);
    result_f64x8 = _mm512_fmadd_pd(a_f64x8, alpha_f64x8, beta_f64x8);
    result_f64x8 = _mm512_max_pd(result_f64x8, min_f64x8);
    result_f64x8 = _mm512_min_pd(result_f64x8, max_f64x8);
    result_u32x8 = _mm512_cvtpd_epu32(result_f64x8);
    _mm256_mask_storeu_epi32(result, mask, result_u32x8);
    result += 8;
    if (n) goto nk_each_scale_u32_skylake_cycle;
}

NK_PUBLIC void nk_each_fma_u32_skylake(                                   //
    nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m256i a_u32x8, b_u32x8, c_u32x8, result_u32x8;
    __m512d a_f64x8, b_f64x8, c_f64x8, ab_f64x8, ab_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
    __m512d min_f64x8 = _mm512_set1_pd(0.0);
    __m512d max_f64x8 = _mm512_set1_pd(4294967295.0);

nk_each_fma_u32_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u32x8 = _mm256_maskz_loadu_epi32(mask, a);
        b_u32x8 = _mm256_maskz_loadu_epi32(mask, b);
        c_u32x8 = _mm256_maskz_loadu_epi32(mask, c);
        n = 0;
    }
    else {
        a_u32x8 = _mm256_loadu_si256((__m256i *)a);
        b_u32x8 = _mm256_loadu_si256((__m256i *)b);
        c_u32x8 = _mm256_loadu_si256((__m256i *)c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepu32_pd(a_u32x8);
    b_f64x8 = _mm512_cvtepu32_pd(b_u32x8);
    c_f64x8 = _mm512_cvtepu32_pd(c_u32x8);
    ab_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    ab_scaled_f64x8 = _mm512_mul_pd(ab_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(c_f64x8, beta_f64x8, ab_scaled_f64x8);
    result_f64x8 = _mm512_max_pd(result_f64x8, min_f64x8);
    result_f64x8 = _mm512_min_pd(result_f64x8, max_f64x8);
    result_u32x8 = _mm512_cvtpd_epu32(result_f64x8);
    _mm256_mask_storeu_epi32(result, mask, result_u32x8);
    result += 8;
    if (n) goto nk_each_fma_u32_skylake_cycle;
}

NK_PUBLIC void nk_each_scale_i64_skylake(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                         nk_i64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512i a_i64x8;
    __m512d a_f64x8, result_f64x8;
    __m512i result_i64x8;
    __mmask8 mask = 0xFF;

nk_each_scale_i64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i64x8 = _mm512_maskz_loadu_epi64(mask, a);
        n = 0;
    }
    else {
        a_i64x8 = _mm512_loadu_si512((__m512i *)a);
        a += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepi64_pd(a_i64x8);
    result_f64x8 = _mm512_fmadd_pd(a_f64x8, alpha_f64x8, beta_f64x8);
    result_i64x8 = _mm512_cvtpd_epi64(result_f64x8);
    _mm512_mask_storeu_epi64(result, mask, result_i64x8);
    result += 8;
    if (n) goto nk_each_scale_i64_skylake_cycle;
}

NK_PUBLIC void nk_each_fma_i64_skylake(                                   //
    nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_i64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512i a_i64x8, b_i64x8, c_i64x8, result_i64x8;
    __m512d a_f64x8, b_f64x8, c_f64x8, ab_f64x8, ab_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
nk_each_fma_i64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i64x8 = _mm512_maskz_loadu_epi64(mask, a);
        b_i64x8 = _mm512_maskz_loadu_epi64(mask, b);
        c_i64x8 = _mm512_maskz_loadu_epi64(mask, c);
        n = 0;
    }
    else {
        a_i64x8 = _mm512_loadu_si512((__m512i *)a);
        b_i64x8 = _mm512_loadu_si512((__m512i *)b);
        c_i64x8 = _mm512_loadu_si512((__m512i *)c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepi64_pd(a_i64x8);
    b_f64x8 = _mm512_cvtepi64_pd(b_i64x8);
    c_f64x8 = _mm512_cvtepi64_pd(c_i64x8);
    ab_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    ab_scaled_f64x8 = _mm512_mul_pd(ab_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(c_f64x8, beta_f64x8, ab_scaled_f64x8);
    result_i64x8 = _mm512_cvtpd_epi64(result_f64x8);
    _mm512_mask_storeu_epi64(result, mask, result_i64x8);
    result += 8;
    if (n) goto nk_each_fma_i64_skylake_cycle;
}

NK_PUBLIC void nk_each_scale_u64_skylake(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                         nk_u64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512i a_u64x8;
    __m512d a_f64x8, result_f64x8;
    __m512i result_u64x8;
    __mmask8 mask = 0xFF;

nk_each_scale_u64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u64x8 = _mm512_maskz_loadu_epi64(mask, a);
        n = 0;
    }
    else {
        a_u64x8 = _mm512_loadu_si512((__m512i *)a);
        a += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepu64_pd(a_u64x8);
    result_f64x8 = _mm512_fmadd_pd(a_f64x8, alpha_f64x8, beta_f64x8);
    result_u64x8 = _mm512_cvtpd_epu64(result_f64x8);
    _mm512_mask_storeu_epi64(result, mask, result_u64x8);
    result += 8;
    if (n) goto nk_each_scale_u64_skylake_cycle;
}

NK_PUBLIC void nk_each_fma_u64_skylake(                                   //
    nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_u64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512i a_u64x8, b_u64x8, c_u64x8, result_u64x8;
    __m512d a_f64x8, b_f64x8, c_f64x8, ab_f64x8, ab_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
nk_each_fma_u64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u64x8 = _mm512_maskz_loadu_epi64(mask, a);
        b_u64x8 = _mm512_maskz_loadu_epi64(mask, b);
        c_u64x8 = _mm512_maskz_loadu_epi64(mask, c);
        n = 0;
    }
    else {
        a_u64x8 = _mm512_loadu_si512((__m512i *)a);
        b_u64x8 = _mm512_loadu_si512((__m512i *)b);
        c_u64x8 = _mm512_loadu_si512((__m512i *)c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepu64_pd(a_u64x8);
    b_f64x8 = _mm512_cvtepu64_pd(b_u64x8);
    c_f64x8 = _mm512_cvtepu64_pd(c_u64x8);
    ab_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    ab_scaled_f64x8 = _mm512_mul_pd(ab_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(c_f64x8, beta_f64x8, ab_scaled_f64x8);
    result_u64x8 = _mm512_cvtpd_epu64(result_f64x8);
    _mm512_mask_storeu_epi64(result, mask, result_u64x8);
    result += 8;
    if (n) goto nk_each_fma_u64_skylake_cycle;
}

NK_PUBLIC void nk_each_sum_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_e4m3_t *result) {
    __m128i a_e4m3x16, b_e4m3x16, result_e4m3x16;
    __m512 a_f32x16, b_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_each_sum_e4m3_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)n);
        a_e4m3x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e4m3x16 = _mm_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e4m3x16 = _mm_loadu_si128((__m128i const *)a);
        b_e4m3x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    a_f32x16 = nk_e4m3x16_to_f32x16_skylake_(a_e4m3x16);
    b_f32x16 = nk_e4m3x16_to_f32x16_skylake_(b_e4m3x16);
    result_f32x16 = _mm512_add_ps(a_f32x16, b_f32x16);
    result_e4m3x16 = nk_f32x16_to_e4m3x16_skylake_(result_f32x16);
    _mm_mask_storeu_epi8(result, mask, result_e4m3x16);
    result += 16;
    if (n) goto nk_each_sum_e4m3_skylake_cycle;
}

NK_PUBLIC void nk_each_sum_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_e5m2_t *result) {
    __m128i a_e5m2x16, b_e5m2x16, result_e5m2x16;
    __m512 a_f32x16, b_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_each_sum_e5m2_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)n);
        a_e5m2x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e5m2x16 = _mm_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e5m2x16 = _mm_loadu_si128((__m128i const *)a);
        b_e5m2x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    a_f32x16 = nk_e5m2x16_to_f32x16_skylake_(a_e5m2x16);
    b_f32x16 = nk_e5m2x16_to_f32x16_skylake_(b_e5m2x16);
    result_f32x16 = _mm512_add_ps(a_f32x16, b_f32x16);
    result_e5m2x16 = nk_f32x16_to_e5m2x16_skylake_(result_f32x16);
    _mm_mask_storeu_epi8(result, mask, result_e5m2x16);
    result += 16;
    if (n) goto nk_each_sum_e5m2_skylake_cycle;
}

NK_PUBLIC void nk_each_scale_e4m3_skylake(nk_e4m3_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                          nk_e4m3_t *result) {
    __m512 alpha_f32x16 = _mm512_set1_ps(*alpha);
    __m512 beta_f32x16 = _mm512_set1_ps(*beta);
    __m128i a_e4m3x16, result_e4m3x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_each_scale_e4m3_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)n);
        a_e4m3x16 = _mm_maskz_loadu_epi8(mask, a);
        n = 0;
    }
    else {
        a_e4m3x16 = _mm_loadu_si128((__m128i const *)a);
        a += 16, n -= 16;
    }
    a_f32x16 = nk_e4m3x16_to_f32x16_skylake_(a_e4m3x16);
    // FP8 rounding note: FMA is acceptable here because scale computes (α × a + β),
    // a single multiply-add operation where single-rounding preserves accuracy.
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_e4m3x16 = nk_f32x16_to_e4m3x16_skylake_(result_f32x16);
    _mm_mask_storeu_epi8(result, mask, result_e4m3x16);
    result += 16;
    if (n) goto nk_each_scale_e4m3_skylake_cycle;
}

NK_PUBLIC void nk_each_scale_e5m2_skylake(nk_e5m2_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                          nk_e5m2_t *result) {
    __m512 alpha_f32x16 = _mm512_set1_ps(*alpha);
    __m512 beta_f32x16 = _mm512_set1_ps(*beta);
    __m128i a_e5m2x16, result_e5m2x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_each_scale_e5m2_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)n);
        a_e5m2x16 = _mm_maskz_loadu_epi8(mask, a);
        n = 0;
    }
    else {
        a_e5m2x16 = _mm_loadu_si128((__m128i const *)a);
        a += 16, n -= 16;
    }
    a_f32x16 = nk_e5m2x16_to_f32x16_skylake_(a_e5m2x16);
    // FP8 rounding note: FMA is acceptable here because scale computes (α × a + β),
    // a single multiply-add operation where single-rounding preserves accuracy.
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_e5m2x16 = nk_f32x16_to_e5m2x16_skylake_(result_f32x16);
    _mm_mask_storeu_epi8(result, mask, result_e5m2x16);
    result += 16;
    if (n) goto nk_each_scale_e5m2_skylake_cycle;
}

NK_PUBLIC void nk_each_blend_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                          nk_f32_t const *beta, nk_e4m3_t *result) {
    __m512 alpha_f32x16 = _mm512_set1_ps(*alpha);
    __m512 beta_f32x16 = _mm512_set1_ps(*beta);
    __m128i a_e4m3x16, b_e4m3x16, result_e4m3x16;
    __m512 a_f32x16, b_f32x16, a_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_each_blend_e4m3_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)n);
        a_e4m3x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e4m3x16 = _mm_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e4m3x16 = _mm_loadu_si128((__m128i const *)a);
        b_e4m3x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    a_f32x16 = nk_e4m3x16_to_f32x16_skylake_(a_e4m3x16);
    b_f32x16 = nk_e4m3x16_to_f32x16_skylake_(b_e4m3x16);
    a_scaled_f32x16 = _mm512_mul_ps(a_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(b_f32x16, beta_f32x16, a_scaled_f32x16);
    result_e4m3x16 = nk_f32x16_to_e4m3x16_skylake_(result_f32x16);
    _mm_mask_storeu_epi8(result, mask, result_e4m3x16);
    result += 16;
    if (n) goto nk_each_blend_e4m3_skylake_cycle;
}

NK_PUBLIC void nk_each_blend_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                          nk_f32_t const *beta, nk_e5m2_t *result) {
    __m512 alpha_f32x16 = _mm512_set1_ps(*alpha);
    __m512 beta_f32x16 = _mm512_set1_ps(*beta);
    __m128i a_e5m2x16, b_e5m2x16, result_e5m2x16;
    __m512 a_f32x16, b_f32x16, a_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_each_blend_e5m2_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)n);
        a_e5m2x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e5m2x16 = _mm_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e5m2x16 = _mm_loadu_si128((__m128i const *)a);
        b_e5m2x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    a_f32x16 = nk_e5m2x16_to_f32x16_skylake_(a_e5m2x16);
    b_f32x16 = nk_e5m2x16_to_f32x16_skylake_(b_e5m2x16);
    a_scaled_f32x16 = _mm512_mul_ps(a_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(b_f32x16, beta_f32x16, a_scaled_f32x16);
    result_e5m2x16 = nk_f32x16_to_e5m2x16_skylake_(result_f32x16);
    _mm_mask_storeu_epi8(result, mask, result_e5m2x16);
    result += 16;
    if (n) goto nk_each_blend_e5m2_skylake_cycle;
}

NK_PUBLIC void nk_each_fma_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_e4m3_t const *c, nk_size_t n,
                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_e4m3_t *result) {
    __m512 alpha_f32x16 = _mm512_set1_ps(*alpha);
    __m512 beta_f32x16 = _mm512_set1_ps(*beta);
    __m128i a_e4m3x16, b_e4m3x16, c_e4m3x16, result_e4m3x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_each_fma_e4m3_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)n);
        a_e4m3x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e4m3x16 = _mm_maskz_loadu_epi8(mask, b);
        c_e4m3x16 = _mm_maskz_loadu_epi8(mask, c);
        n = 0;
    }
    else {
        a_e4m3x16 = _mm_loadu_si128((__m128i const *)a);
        b_e4m3x16 = _mm_loadu_si128((__m128i const *)b);
        c_e4m3x16 = _mm_loadu_si128((__m128i const *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = nk_e4m3x16_to_f32x16_skylake_(a_e4m3x16);
    b_f32x16 = nk_e4m3x16_to_f32x16_skylake_(b_e4m3x16);
    c_f32x16 = nk_e4m3x16_to_f32x16_skylake_(c_e4m3x16);
    // FP8 rounding note: Hybrid approach - use separate MUL for (a × b) and (α × a × b) to
    // preserve intermediate rounding, then FMA for final addition since it matches scalar
    // semantics of (α × a × b + β × c) when the multiply term is already computed.
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_e4m3x16 = nk_f32x16_to_e4m3x16_skylake_(result_f32x16);
    _mm_mask_storeu_epi8(result, mask, result_e4m3x16);
    result += 16;
    if (n) goto nk_each_fma_e4m3_skylake_cycle;
}

NK_PUBLIC void nk_each_fma_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_e5m2_t const *c, nk_size_t n,
                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_e5m2_t *result) {
    __m512 alpha_f32x16 = _mm512_set1_ps(*alpha);
    __m512 beta_f32x16 = _mm512_set1_ps(*beta);
    __m128i a_e5m2x16, b_e5m2x16, c_e5m2x16, result_e5m2x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_each_fma_e5m2_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)n);
        a_e5m2x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e5m2x16 = _mm_maskz_loadu_epi8(mask, b);
        c_e5m2x16 = _mm_maskz_loadu_epi8(mask, c);
        n = 0;
    }
    else {
        a_e5m2x16 = _mm_loadu_si128((__m128i const *)a);
        b_e5m2x16 = _mm_loadu_si128((__m128i const *)b);
        c_e5m2x16 = _mm_loadu_si128((__m128i const *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = nk_e5m2x16_to_f32x16_skylake_(a_e5m2x16);
    b_f32x16 = nk_e5m2x16_to_f32x16_skylake_(b_e5m2x16);
    c_f32x16 = nk_e5m2x16_to_f32x16_skylake_(c_e5m2x16);
    // FP8 rounding note: Hybrid approach - use separate MUL for (a × b) and (α × a × b) to
    // preserve intermediate rounding, then FMA for final addition since it matches scalar
    // semantics of (α × a × b + β × c) when the multiply term is already computed.
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_e5m2x16 = nk_f32x16_to_e5m2x16_skylake_(result_f32x16);
    _mm_mask_storeu_epi8(result, mask, result_e5m2x16);
    result += 16;
    if (n) goto nk_each_fma_e5m2_skylake_cycle;
}

NK_PUBLIC void nk_each_scale_f32c_skylake(nk_f32c_t const *a, nk_size_t n, nk_f32c_t const *alpha,
                                          nk_f32c_t const *beta, nk_f32c_t *result) {
    nk_f32_t const *a_f32 = (nk_f32_t const *)a;
    nk_f32_t *result_f32 = (nk_f32_t *)result;
    __m512 alpha_real_f32x16 = _mm512_set1_ps(alpha->real);
    __m512 alpha_imag_f32x16 = _mm512_set1_ps(alpha->imag);
    __m512 beta_f32x16 = _mm512_set_ps(beta->imag, beta->real, beta->imag, beta->real, beta->imag, beta->real,
                                       beta->imag, beta->real, beta->imag, beta->real, beta->imag, beta->real,
                                       beta->imag, beta->real, beta->imag, beta->real);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512 a_f32x16 = _mm512_loadu_ps(a_f32 + 2 * i);
        __m512 a_swapped_f32x16 = _mm512_permute_ps(a_f32x16, 0xB1);
        __m512 temp_f32x16 = _mm512_mul_ps(alpha_imag_f32x16, a_swapped_f32x16);
        __m512 y_f32x16 = _mm512_fmaddsub_ps(alpha_real_f32x16, a_f32x16, temp_f32x16);
        y_f32x16 = _mm512_add_ps(y_f32x16, beta_f32x16);
        _mm512_storeu_ps(result_f32 + 2 * i, y_f32x16);
    }
    for (; i < n; i++) {
        nk_f32_t a_real = a[i].real, a_imag = a[i].imag;
        result[i].real = alpha->real * a_real - alpha->imag * a_imag + beta->real;
        result[i].imag = alpha->real * a_imag + alpha->imag * a_real + beta->imag;
    }
}

NK_PUBLIC void nk_each_scale_f64c_skylake(nk_f64c_t const *a, nk_size_t n, nk_f64c_t const *alpha,
                                          nk_f64c_t const *beta, nk_f64c_t *result) {
    nk_f64_t const *a_f64 = (nk_f64_t const *)a;
    nk_f64_t *result_f64 = (nk_f64_t *)result;
    __m512d alpha_real_f64x8 = _mm512_set1_pd(alpha->real);
    __m512d alpha_imag_f64x8 = _mm512_set1_pd(alpha->imag);
    __m512d beta_f64x8 = _mm512_set_pd(beta->imag, beta->real, beta->imag, beta->real, beta->imag, beta->real,
                                       beta->imag, beta->real);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m512d a_f64x8 = _mm512_loadu_pd(a_f64 + 2 * i);
        __m512d a_swapped_f64x8 = _mm512_permute_pd(a_f64x8, 0x55);
        __m512d temp_f64x8 = _mm512_mul_pd(alpha_imag_f64x8, a_swapped_f64x8);
        __m512d y_f64x8 = _mm512_fmaddsub_pd(alpha_real_f64x8, a_f64x8, temp_f64x8);
        y_f64x8 = _mm512_add_pd(y_f64x8, beta_f64x8);
        _mm512_storeu_pd(result_f64 + 2 * i, y_f64x8);
    }
    for (; i < n; i++) {
        nk_f64_t a_real = a[i].real, a_imag = a[i].imag;
        result[i].real = alpha->real * a_real - alpha->imag * a_imag + beta->real;
        result[i].imag = alpha->real * a_imag + alpha->imag * a_real + beta->imag;
    }
}

NK_PUBLIC void nk_each_blend_f32c_skylake(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t const *alpha,
                                          nk_f32c_t const *beta, nk_f32c_t *result) {
    nk_f32_t const *a_f32 = (nk_f32_t const *)a;
    nk_f32_t const *b_f32 = (nk_f32_t const *)b;
    nk_f32_t *result_f32 = (nk_f32_t *)result;
    __m512 alpha_real_f32x16 = _mm512_set1_ps(alpha->real);
    __m512 alpha_imag_f32x16 = _mm512_set1_ps(alpha->imag);
    __m512 beta_real_f32x16 = _mm512_set1_ps(beta->real);
    __m512 beta_imag_f32x16 = _mm512_set1_ps(beta->imag);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512 a_f32x16 = _mm512_loadu_ps(a_f32 + 2 * i);
        __m512 b_f32x16 = _mm512_loadu_ps(b_f32 + 2 * i);
        __m512 a_swapped_f32x16 = _mm512_permute_ps(a_f32x16, 0xB1);
        __m512 ta_f32x16 = _mm512_mul_ps(alpha_imag_f32x16, a_swapped_f32x16);
        __m512 ya_f32x16 = _mm512_fmaddsub_ps(alpha_real_f32x16, a_f32x16, ta_f32x16);
        __m512 b_swapped_f32x16 = _mm512_permute_ps(b_f32x16, 0xB1);
        __m512 tb_f32x16 = _mm512_mul_ps(beta_imag_f32x16, b_swapped_f32x16);
        __m512 yb_f32x16 = _mm512_fmaddsub_ps(beta_real_f32x16, b_f32x16, tb_f32x16);
        _mm512_storeu_ps(result_f32 + 2 * i, _mm512_add_ps(ya_f32x16, yb_f32x16));
    }
    for (; i < n; i++) {
        nk_f32_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f32_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f32_t ar = alpha->real * a_real - alpha->imag * a_imag;
        nk_f32_t ai = alpha->real * a_imag + alpha->imag * a_real;
        nk_f32_t br = beta->real * b_real - beta->imag * b_imag;
        nk_f32_t bi = beta->real * b_imag + beta->imag * b_real;
        result[i].real = ar + br;
        result[i].imag = ai + bi;
    }
}

NK_PUBLIC void nk_each_blend_f64c_skylake(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t const *alpha,
                                          nk_f64c_t const *beta, nk_f64c_t *result) {
    nk_f64_t const *a_f64 = (nk_f64_t const *)a;
    nk_f64_t const *b_f64 = (nk_f64_t const *)b;
    nk_f64_t *result_f64 = (nk_f64_t *)result;
    __m512d alpha_real_f64x8 = _mm512_set1_pd(alpha->real);
    __m512d alpha_imag_f64x8 = _mm512_set1_pd(alpha->imag);
    __m512d beta_real_f64x8 = _mm512_set1_pd(beta->real);
    __m512d beta_imag_f64x8 = _mm512_set1_pd(beta->imag);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m512d a_f64x8 = _mm512_loadu_pd(a_f64 + 2 * i);
        __m512d b_f64x8 = _mm512_loadu_pd(b_f64 + 2 * i);
        __m512d a_swapped_f64x8 = _mm512_permute_pd(a_f64x8, 0x55);
        __m512d ta_f64x8 = _mm512_mul_pd(alpha_imag_f64x8, a_swapped_f64x8);
        __m512d ya_f64x8 = _mm512_fmaddsub_pd(alpha_real_f64x8, a_f64x8, ta_f64x8);
        __m512d b_swapped_f64x8 = _mm512_permute_pd(b_f64x8, 0x55);
        __m512d tb_f64x8 = _mm512_mul_pd(beta_imag_f64x8, b_swapped_f64x8);
        __m512d yb_f64x8 = _mm512_fmaddsub_pd(beta_real_f64x8, b_f64x8, tb_f64x8);
        _mm512_storeu_pd(result_f64 + 2 * i, _mm512_add_pd(ya_f64x8, yb_f64x8));
    }
    for (; i < n; i++) {
        nk_f64_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f64_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f64_t ar = alpha->real * a_real - alpha->imag * a_imag;
        nk_f64_t ai = alpha->real * a_imag + alpha->imag * a_real;
        nk_f64_t br = beta->real * b_real - beta->imag * b_imag;
        nk_f64_t bi = beta->real * b_imag + beta->imag * b_real;
        result[i].real = ar + br;
        result[i].imag = ai + bi;
    }
}

NK_PUBLIC void nk_each_fma_f32c_skylake(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                        nk_f32c_t const *alpha, nk_f32c_t const *beta, nk_f32c_t *result) {
    nk_f32_t const *a_f32 = (nk_f32_t const *)a;
    nk_f32_t const *b_f32 = (nk_f32_t const *)b;
    nk_f32_t const *c_f32 = (nk_f32_t const *)c;
    nk_f32_t *result_f32 = (nk_f32_t *)result;
    __m512 alpha_real_f32x16 = _mm512_set1_ps(alpha->real);
    __m512 alpha_imag_f32x16 = _mm512_set1_ps(alpha->imag);
    __m512 beta_real_f32x16 = _mm512_set1_ps(beta->real);
    __m512 beta_imag_f32x16 = _mm512_set1_ps(beta->imag);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512 a_f32x16 = _mm512_loadu_ps(a_f32 + 2 * i);
        __m512 b_f32x16 = _mm512_loadu_ps(b_f32 + 2 * i);
        __m512 c_f32x16 = _mm512_loadu_ps(c_f32 + 2 * i);
        __m512 b_swapped_f32x16 = _mm512_permute_ps(b_f32x16, 0xB1);
        __m512 a_real_f32x16 = _mm512_moveldup_ps(a_f32x16);
        __m512 a_imag_f32x16 = _mm512_movehdup_ps(a_f32x16);
        __m512 tab_f32x16 = _mm512_mul_ps(a_imag_f32x16, b_swapped_f32x16);
        __m512 ab_f32x16 = _mm512_fmaddsub_ps(a_real_f32x16, b_f32x16, tab_f32x16);
        __m512 ab_swapped_f32x16 = _mm512_permute_ps(ab_f32x16, 0xB1);
        __m512 taa_f32x16 = _mm512_mul_ps(alpha_imag_f32x16, ab_swapped_f32x16);
        __m512 ya_f32x16 = _mm512_fmaddsub_ps(alpha_real_f32x16, ab_f32x16, taa_f32x16);
        __m512 c_swapped_f32x16 = _mm512_permute_ps(c_f32x16, 0xB1);
        __m512 tbc_f32x16 = _mm512_mul_ps(beta_imag_f32x16, c_swapped_f32x16);
        __m512 yb_f32x16 = _mm512_fmaddsub_ps(beta_real_f32x16, c_f32x16, tbc_f32x16);
        _mm512_storeu_ps(result_f32 + 2 * i, _mm512_add_ps(ya_f32x16, yb_f32x16));
    }
    for (; i < n; i++) {
        nk_f32_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f32_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f32_t c_real = c[i].real, c_imag = c[i].imag;
        nk_f32_t ab_real = a_real * b_real - a_imag * b_imag;
        nk_f32_t ab_imag = a_real * b_imag + a_imag * b_real;
        nk_f32_t aab_real = alpha->real * ab_real - alpha->imag * ab_imag;
        nk_f32_t aab_imag = alpha->real * ab_imag + alpha->imag * ab_real;
        nk_f32_t bc_real = beta->real * c_real - beta->imag * c_imag;
        nk_f32_t bc_imag = beta->real * c_imag + beta->imag * c_real;
        result[i].real = aab_real + bc_real;
        result[i].imag = aab_imag + bc_imag;
    }
}

NK_PUBLIC void nk_each_fma_f64c_skylake(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                        nk_f64c_t const *alpha, nk_f64c_t const *beta, nk_f64c_t *result) {
    nk_f64_t const *a_f64 = (nk_f64_t const *)a;
    nk_f64_t const *b_f64 = (nk_f64_t const *)b;
    nk_f64_t const *c_f64 = (nk_f64_t const *)c;
    nk_f64_t *result_f64 = (nk_f64_t *)result;
    __m512d alpha_real_f64x8 = _mm512_set1_pd(alpha->real);
    __m512d alpha_imag_f64x8 = _mm512_set1_pd(alpha->imag);
    __m512d beta_real_f64x8 = _mm512_set1_pd(beta->real);
    __m512d beta_imag_f64x8 = _mm512_set1_pd(beta->imag);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m512d a_f64x8 = _mm512_loadu_pd(a_f64 + 2 * i);
        __m512d b_f64x8 = _mm512_loadu_pd(b_f64 + 2 * i);
        __m512d c_f64x8 = _mm512_loadu_pd(c_f64 + 2 * i);
        __m512d b_swapped_f64x8 = _mm512_permute_pd(b_f64x8, 0x55);
        __m512d a_real_f64x8 = _mm512_unpacklo_pd(a_f64x8, a_f64x8);
        __m512d a_imag_f64x8 = _mm512_unpackhi_pd(a_f64x8, a_f64x8);
        __m512d tab_f64x8 = _mm512_mul_pd(a_imag_f64x8, b_swapped_f64x8);
        __m512d ab_f64x8 = _mm512_fmaddsub_pd(a_real_f64x8, b_f64x8, tab_f64x8);
        __m512d ab_swapped_f64x8 = _mm512_permute_pd(ab_f64x8, 0x55);
        __m512d taa_f64x8 = _mm512_mul_pd(alpha_imag_f64x8, ab_swapped_f64x8);
        __m512d ya_f64x8 = _mm512_fmaddsub_pd(alpha_real_f64x8, ab_f64x8, taa_f64x8);
        __m512d c_swapped_f64x8 = _mm512_permute_pd(c_f64x8, 0x55);
        __m512d tbc_f64x8 = _mm512_mul_pd(beta_imag_f64x8, c_swapped_f64x8);
        __m512d yb_f64x8 = _mm512_fmaddsub_pd(beta_real_f64x8, c_f64x8, tbc_f64x8);
        _mm512_storeu_pd(result_f64 + 2 * i, _mm512_add_pd(ya_f64x8, yb_f64x8));
    }
    for (; i < n; i++) {
        nk_f64_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f64_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f64_t c_real = c[i].real, c_imag = c[i].imag;
        nk_f64_t ab_real = a_real * b_real - a_imag * b_imag;
        nk_f64_t ab_imag = a_real * b_imag + a_imag * b_real;
        nk_f64_t aab_real = alpha->real * ab_real - alpha->imag * ab_imag;
        nk_f64_t aab_imag = alpha->real * ab_imag + alpha->imag * ab_real;
        nk_f64_t bc_real = beta->real * c_real - beta->imag * c_imag;
        nk_f64_t bc_imag = beta->real * c_imag + beta->imag * c_real;
        result[i].real = aab_real + bc_real;
        result[i].imag = aab_imag + bc_imag;
    }
}

NK_PUBLIC void nk_each_scale_f16_skylake(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                         nk_f16_t *result) {
    __m512 alpha_f32x16 = _mm512_set1_ps(*alpha);
    __m512 beta_f32x16 = _mm512_set1_ps(*beta);
    __m512 a_f32x16;
nk_each_scale_f16_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_f32x16 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a));
        __m512 result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
        _mm256_mask_storeu_epi16(result, mask, _mm512_cvtps_ph(result_f32x16, _MM_FROUND_TO_NEAREST_INT));
        n = 0;
    }
    else {
        a_f32x16 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)a));
        __m512 result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
        _mm256_storeu_si256((__m256i *)result, _mm512_cvtps_ph(result_f32x16, _MM_FROUND_TO_NEAREST_INT));
        a += 16, result += 16, n -= 16;
    }
    if (n) goto nk_each_scale_f16_skylake_cycle;
}

NK_PUBLIC void nk_each_blend_f16_skylake(              //
    nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_each_sum_f16_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_f16_skylake(a, n, alpha, &zero, result); }
        else { nk_each_scale_f16_skylake(b, n, beta, &zero, result); }
        return;
    }

    // The general case: compute in f32 for precision (f16 products overflow at 127x127=16129)
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m512 a_f32x16, b_f32x16;
nk_each_blend_f16_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_f32x16 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a));
        b_f32x16 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, b));
        __m512 a_scaled_f32x16 = _mm512_mul_ps(a_f32x16, alpha_f32x16);
        __m512 result_f32x16 = _mm512_fmadd_ps(b_f32x16, beta_f32x16, a_scaled_f32x16);
        _mm256_mask_storeu_epi16(result, mask, _mm512_cvtps_ph(result_f32x16, _MM_FROUND_TO_NEAREST_INT));
        n = 0;
    }
    else {
        a_f32x16 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)a));
        b_f32x16 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)b));
        __m512 a_scaled_f32x16 = _mm512_mul_ps(a_f32x16, alpha_f32x16);
        __m512 result_f32x16 = _mm512_fmadd_ps(b_f32x16, beta_f32x16, a_scaled_f32x16);
        _mm256_storeu_si256((__m256i *)result, _mm512_cvtps_ph(result_f32x16, _MM_FROUND_TO_NEAREST_INT));
        a += 16, b += 16, result += 16, n -= 16;
    }
    if (n) goto nk_each_blend_f16_skylake_cycle;
}

NK_PUBLIC void nk_each_fma_f16_skylake(                                   //
    nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {

    // Compute in f32 for precision (f16 products overflow at 127x127=16129)
    __m512 alpha_f32x16 = _mm512_set1_ps(*alpha);
    __m512 beta_f32x16 = _mm512_set1_ps(*beta);
    __m512 a_f32x16, b_f32x16, c_f32x16;
nk_each_fma_f16_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_f32x16 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a));
        b_f32x16 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, b));
        c_f32x16 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, c));
        __m512 ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
        __m512 ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
        __m512 result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
        _mm256_mask_storeu_epi16(result, mask, _mm512_cvtps_ph(result_f32x16, _MM_FROUND_TO_NEAREST_INT));
        n = 0;
    }
    else {
        a_f32x16 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)a));
        b_f32x16 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)b));
        c_f32x16 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)c));
        __m512 ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
        __m512 ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
        __m512 result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
        _mm256_storeu_si256((__m256i *)result, _mm512_cvtps_ph(result_f32x16, _MM_FROUND_TO_NEAREST_INT));
        a += 16, b += 16, c += 16, result += 16, n -= 16;
    }
    if (n) goto nk_each_fma_f16_skylake_cycle;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_
#endif // NK_EACH_SKYLAKE_H
