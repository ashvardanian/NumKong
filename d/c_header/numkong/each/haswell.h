/**
 *  @brief SIMD-accelerated Elementwise Arithmetic for Haswell.
 *  @file include/numkong/each/haswell.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/each.h
 *
 *  @section haswell_elementwise_instructions Key AVX2 Elementwise Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      _mm256_fmadd_ps             VFMADD (YMM, YMM, YMM)          5cy         0.5/cy      p01
 *      _mm256_add_ps               VADDPS (YMM, YMM, YMM)          3cy         1/cy        p01
 *      _mm256_mul_ps               VMULPS (YMM, YMM, YMM)          5cy         0.5/cy      p01
 *      _mm256_cvtepi32_ps          VCVTDQ2PS (YMM, YMM)            4cy         1/cy        p01
 *      _mm256_cvtepi8_epi32        VPMOVSXBD (YMM, XMM)            3cy         1/cy        p5
 *
 *  Elementwise operations (sum, scale, blend, fma) are compute-bound on FMA throughput. For mixed-
 *  precision operations, type conversion chains (e.g., i8->i32->f32) add ~7-10 cycles overhead.
 *  The FMA unit handles both multiply-add fusion and standalone multiply/add operations.
 */
#ifndef NK_EACH_HASWELL_H
#define NK_EACH_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL

#include "numkong/types.h"
#include "numkong/cast/serial.h"    // `nk_f32_to_i8_serial`
#include "numkong/reduce/haswell.h" // `nk_e4m3x8_to_f32x8_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC void nk_each_sum_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_loadu_ps(a + i);
        __m256 b_f32x8 = _mm256_loadu_ps(b + i);
        __m256 result_f32x8 = _mm256_add_ps(a_f32x8, b_f32x8);
        _mm256_storeu_ps(result + i, result_f32x8);
    }

    // The tail:
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

NK_PUBLIC void nk_each_scale_f32_haswell(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                         nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_loadu_ps(a + i);
        __m256 result_f32x8 = _mm256_fmadd_ps(a_f32x8, alpha_f32x8, beta_f32x8);
        _mm256_storeu_ps(result + i, result_f32x8);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val;
}

NK_PUBLIC void nk_each_blend_f32_haswell(              //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_each_sum_f32_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_f32_haswell(a, n, alpha, &zero, result); }
        else { nk_each_scale_f32_haswell(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_loadu_ps(a + i);
        __m256 b_f32x8 = _mm256_loadu_ps(b + i);
        __m256 a_scaled_f32x8 = _mm256_mul_ps(a_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(b_f32x8, beta_f32x8, a_scaled_f32x8);
        _mm256_storeu_ps(result + i, result_f32x8);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val * b[i];
}

NK_PUBLIC void nk_each_sum_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_loadu_pd(a + i);
        __m256d b_f64x4 = _mm256_loadu_pd(b + i);
        __m256d result_f64x4 = _mm256_add_pd(a_f64x4, b_f64x4);
        _mm256_storeu_pd(result + i, result_f64x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

NK_PUBLIC void nk_each_scale_f64_haswell(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                         nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m256d alpha_f64x4 = _mm256_set1_pd(alpha_val);
    __m256d beta_f64x4 = _mm256_set1_pd(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_loadu_pd(a + i);
        __m256d result_f64x4 = _mm256_fmadd_pd(a_f64x4, alpha_f64x4, beta_f64x4);
        _mm256_storeu_pd(result + i, result_f64x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val;
}

NK_PUBLIC void nk_each_blend_f64_haswell(              //
    nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // There are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_each_sum_f64_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f64_t zero = 0;
        if (beta_val == 0) { nk_each_scale_f64_haswell(a, n, alpha, &zero, result); }
        else { nk_each_scale_f64_haswell(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m256d alpha_f64x4 = _mm256_set1_pd(alpha_val);
    __m256d beta_f64x4 = _mm256_set1_pd(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_loadu_pd(a + i);
        __m256d b_f64x4 = _mm256_loadu_pd(b + i);
        __m256d a_scaled_f64x4 = _mm256_mul_pd(a_f64x4, alpha_f64x4);
        __m256d result_f64x4 = _mm256_fmadd_pd(b_f64x4, beta_f64x4, a_scaled_f64x4);
        _mm256_storeu_pd(result + i, result_f64x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val * b[i];
}

NK_PUBLIC void nk_each_sum_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result) {

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16x8 = _mm_loadu_si128((__m128i const *)(a + i));
        __m128i b_f16x8 = _mm_loadu_si128((__m128i const *)(b + i));
        __m256 a_f32x8 = _mm256_cvtph_ps(a_f16x8);
        __m256 b_f32x8 = _mm256_cvtph_ps(b_f16x8);
        __m256 result_f32x8 = _mm256_add_ps(a_f32x8, b_f32x8);
        __m128i result_f16x8 = _mm256_cvtps_ph(result_f32x8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(result + i), result_f16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_f16_to_f32_haswell(a + i, &ai);
        nk_f16_to_f32_haswell(b + i, &bi);
        nk_f32_t sum = ai + bi;
        nk_f32_to_f16_haswell(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_scale_f16_haswell(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                         nk_f16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16x8 = _mm_loadu_si128((__m128i const *)(a + i));
        __m256 a_f32x8 = _mm256_cvtph_ps(a_f16x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(a_f32x8, alpha_f32x8, beta_f32x8);
        __m128i result_f16x8 = _mm256_cvtps_ph(result_f32x8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(result + i), result_f16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai;
        nk_f16_to_f32_haswell(a + i, &ai);
        nk_f32_t sum = alpha_val * ai + beta_val;
        nk_f32_to_f16_haswell(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_blend_f16_haswell(              //
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
        if (beta_val == 0) { nk_each_scale_f16_haswell(a, n, alpha, &zero, result); }
        else { nk_each_scale_f16_haswell(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16x8 = _mm_loadu_si128((__m128i const *)(a + i));
        __m128i b_f16x8 = _mm_loadu_si128((__m128i const *)(b + i));
        __m256 a_f32x8 = _mm256_cvtph_ps(a_f16x8);
        __m256 b_f32x8 = _mm256_cvtph_ps(b_f16x8);
        __m256 a_scaled_f32x8 = _mm256_mul_ps(a_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(b_f32x8, beta_f32x8, a_scaled_f32x8);
        __m128i result_f16x8 = _mm256_cvtps_ph(result_f32x8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(result + i), result_f16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_f16_to_f32_haswell(a + i, &ai);
        nk_f16_to_f32_haswell(b + i, &bi);
        nk_f32_t sum = alpha_val * ai + beta_val * bi;
        nk_f32_to_f16_haswell(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_sum_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_bf16x8 = _mm_loadu_si128((__m128i const *)(a + i));
        __m128i b_bf16x8 = _mm_loadu_si128((__m128i const *)(b + i));
        __m256 a_f32x8 = nk_bf16x8_to_f32x8_haswell_(a_bf16x8);
        __m256 b_f32x8 = nk_bf16x8_to_f32x8_haswell_(b_bf16x8);
        __m256 result_f32x8 = _mm256_add_ps(a_f32x8, b_f32x8);
        __m128i result_bf16x8 = nk_f32x8_to_bf16x8_haswell_(result_f32x8);
        _mm_storeu_si128((__m128i *)(result + i), result_bf16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_bf16_to_f32_serial(a + i, &ai);
        nk_bf16_to_f32_serial(b + i, &bi);
        nk_f32_t sum = ai + bi;
        nk_f32_to_bf16_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_scale_bf16_haswell(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                          nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_bf16x8 = _mm_loadu_si128((__m128i const *)(a + i));
        __m256 a_f32x8 = nk_bf16x8_to_f32x8_haswell_(a_bf16x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(a_f32x8, alpha_f32x8, beta_f32x8);
        __m128i result_bf16x8 = nk_f32x8_to_bf16x8_haswell_(result_f32x8);
        _mm_storeu_si128((__m128i *)(result + i), result_bf16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai;
        nk_bf16_to_f32_serial(a + i, &ai);
        nk_f32_t sum = alpha_val * ai + beta_val;
        nk_f32_to_bf16_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_blend_bf16_haswell(               //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_each_sum_bf16_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_bf16_haswell(a, n, alpha, &zero, result); }
        else { nk_each_scale_bf16_haswell(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_bf16x8 = _mm_loadu_si128((__m128i const *)(a + i));
        __m128i b_bf16x8 = _mm_loadu_si128((__m128i const *)(b + i));
        __m256 a_f32x8 = nk_bf16x8_to_f32x8_haswell_(a_bf16x8);
        __m256 b_f32x8 = nk_bf16x8_to_f32x8_haswell_(b_bf16x8);
        __m256 a_scaled_f32x8 = _mm256_mul_ps(a_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(b_f32x8, beta_f32x8, a_scaled_f32x8);
        __m128i result_bf16x8 = nk_f32x8_to_bf16x8_haswell_(result_f32x8);
        _mm_storeu_si128((__m128i *)(result + i), result_bf16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_bf16_to_f32_serial(a + i, &ai);
        nk_bf16_to_f32_serial(b + i, &bi);
        nk_f32_t sum = alpha_val * ai + beta_val * bi;
        nk_f32_to_bf16_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_fma_f32_haswell(                      //
    nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_loadu_ps(a + i);
        __m256 b_f32x8 = _mm256_loadu_ps(b + i);
        __m256 c_f32x8 = _mm256_loadu_ps(c + i);
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, b_f32x8);
        __m256 ab_scaled_f32x8 = _mm256_mul_ps(ab_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(c_f32x8, beta_f32x8, ab_scaled_f32x8);
        _mm256_storeu_ps(result + i, result_f32x8);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] * b[i] + beta_val * c[i];
}

NK_PUBLIC void nk_each_fma_f64_haswell(                      //
    nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, //
    nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m256d alpha_f64x4 = _mm256_set1_pd(alpha_val);
    __m256d beta_f64x4 = _mm256_set1_pd(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_loadu_pd(a + i);
        __m256d b_f64x4 = _mm256_loadu_pd(b + i);
        __m256d c_f64x4 = _mm256_loadu_pd(c + i);
        __m256d ab_f64x4 = _mm256_mul_pd(a_f64x4, b_f64x4);
        __m256d abc_f64x4 = _mm256_mul_pd(ab_f64x4, alpha_f64x4);
        __m256d result_f64x4 = _mm256_fmadd_pd(c_f64x4, beta_f64x4, abc_f64x4);
        _mm256_storeu_pd(result + i, result_f64x4);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_val * a[i] * b[i] + beta_val * c[i];
}

NK_PUBLIC void nk_each_fma_f16_haswell(                      //
    nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16x8 = _mm_loadu_si128((__m128i const *)(a + i));
        __m128i b_f16x8 = _mm_loadu_si128((__m128i const *)(b + i));
        __m128i c_f16x8 = _mm_loadu_si128((__m128i const *)(c + i));
        __m256 a_f32x8 = _mm256_cvtph_ps(a_f16x8);
        __m256 b_f32x8 = _mm256_cvtph_ps(b_f16x8);
        __m256 c_f32x8 = _mm256_cvtph_ps(c_f16x8);
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, b_f32x8);
        __m256 abc_f32x8 = _mm256_mul_ps(ab_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(c_f32x8, beta_f32x8, abc_f32x8);
        __m128i result_f16x8 = _mm256_cvtps_ph(result_f32x8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(result + i), result_f16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi, ci;
        nk_f16_to_f32_haswell(a + i, &ai);
        nk_f16_to_f32_haswell(b + i, &bi);
        nk_f16_to_f32_haswell(c + i, &ci);
        nk_f32_t sum = alpha_val * ai * bi + beta_val * ci;
        nk_f32_to_f16_haswell(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_fma_bf16_haswell(                        //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_bf16x8 = _mm_loadu_si128((__m128i const *)(a + i));
        __m128i b_bf16x8 = _mm_loadu_si128((__m128i const *)(b + i));
        __m128i c_bf16x8 = _mm_loadu_si128((__m128i const *)(c + i));
        __m256 a_f32x8 = nk_bf16x8_to_f32x8_haswell_(a_bf16x8);
        __m256 b_f32x8 = nk_bf16x8_to_f32x8_haswell_(b_bf16x8);
        __m256 c_f32x8 = nk_bf16x8_to_f32x8_haswell_(c_bf16x8);
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, b_f32x8);
        __m256 abc_f32x8 = _mm256_mul_ps(ab_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(c_f32x8, beta_f32x8, abc_f32x8);
        __m128i result_bf16x8 = nk_f32x8_to_bf16x8_haswell_(result_f32x8);
        _mm_storeu_si128((__m128i *)(result + i), result_bf16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi, ci;
        nk_bf16_to_f32_serial(a + i, &ai);
        nk_bf16_to_f32_serial(b + i, &bi);
        nk_bf16_to_f32_serial(c + i, &ci);
        nk_f32_t sum = alpha_val * ai * bi + beta_val * ci;
        nk_f32_to_bf16_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_sum_i8_haswell(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8x32 = _mm256_loadu_si256((__m256i *)(a + i));
        __m256i b_i8x32 = _mm256_loadu_si256((__m256i *)(b + i));
        __m256i result_i8x32 = _mm256_adds_epi8(a_i8x32, b_i8x32);
        _mm256_storeu_si256((__m256i *)(result + i), result_i8x32);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i];
        nk_f32_t sum = ai + bi;
        nk_f32_to_i8_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_scale_i8_haswell(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                        nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);
    int sum_i32s[8], a_i32s[8];

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on `_mm256_cvtepi32_ps`: 4cy (1/cy) @ p01.
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i *)a_i32s));
        // The normal part.
        __m256 result_f32x8 = _mm256_fmadd_ps(a_f32x8, alpha_f32x8, beta_f32x8);
        // Instead of serial calls to expensive `nk_f32_to_u8_serial`, convert and clip with SIMD.
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        result_i32x8 = _mm256_max_epi32(result_i32x8, _mm256_set1_epi32(-128));
        result_i32x8 = _mm256_min_epi32(result_i32x8, _mm256_set1_epi32(127));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, result_i32x8);
        result[i + 0] = (nk_i8_t)sum_i32s[0];
        result[i + 1] = (nk_i8_t)sum_i32s[1];
        result[i + 2] = (nk_i8_t)sum_i32s[2];
        result[i + 3] = (nk_i8_t)sum_i32s[3];
        result[i + 4] = (nk_i8_t)sum_i32s[4];
        result[i + 5] = (nk_i8_t)sum_i32s[5];
        result[i + 6] = (nk_i8_t)sum_i32s[6];
        result[i + 7] = (nk_i8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i];
        nk_f32_t sum = alpha_val * ai + beta_val;
        nk_f32_to_i8_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_blend_i8_haswell(             //
    nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_each_sum_i8_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_i8_haswell(a, n, alpha, &zero, result); }
        else { nk_each_scale_i8_haswell(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);
    int sum_i32s[8], a_i32s[8], b_i32s[8];

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        b_i32s[0] = b[i + 0], b_i32s[1] = b[i + 1], b_i32s[2] = b[i + 2], b_i32s[3] = b[i + 3], //
            b_i32s[4] = b[i + 4], b_i32s[5] = b[i + 5], b_i32s[6] = b[i + 6], b_i32s[7] = b[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on `_mm256_cvtepi32_ps`: 4cy (1/cy) @ p01.
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i *)a_i32s));
        __m256 b_f32x8 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i *)b_i32s));
        // The normal part.
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(b_f32x8, beta_f32x8, ab_f32x8);
        // Instead of serial calls to expensive `nk_f32_to_u8_serial`, convert and clip with SIMD.
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        result_i32x8 = _mm256_max_epi32(result_i32x8, _mm256_set1_epi32(-128));
        result_i32x8 = _mm256_min_epi32(result_i32x8, _mm256_set1_epi32(127));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, result_i32x8);
        result[i + 0] = (nk_i8_t)sum_i32s[0];
        result[i + 1] = (nk_i8_t)sum_i32s[1];
        result[i + 2] = (nk_i8_t)sum_i32s[2];
        result[i + 3] = (nk_i8_t)sum_i32s[3];
        result[i + 4] = (nk_i8_t)sum_i32s[4];
        result[i + 5] = (nk_i8_t)sum_i32s[5];
        result[i + 6] = (nk_i8_t)sum_i32s[6];
        result[i + 7] = (nk_i8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i];
        nk_f32_t sum = alpha_val * ai + beta_val * bi;
        nk_f32_to_i8_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_sum_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_u8x32 = _mm256_loadu_si256((__m256i *)(a + i));
        __m256i b_u8x32 = _mm256_loadu_si256((__m256i *)(b + i));
        __m256i result_u8x32 = _mm256_adds_epu8(a_u8x32, b_u8x32);
        _mm256_storeu_si256((__m256i *)(result + i), result_u8x32);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i];
        nk_f32_t sum = ai + bi;
        nk_f32_to_u8_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_scale_u8_haswell(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                        nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);
    int sum_i32s[8], a_i32s[8];

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on `_mm256_cvtepi32_ps`: 4cy (1/cy) @ p01.
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i *)a_i32s));
        // The normal part.
        __m256 result_f32x8 = _mm256_fmadd_ps(a_f32x8, alpha_f32x8, beta_f32x8);
        // Instead of serial calls to expensive `nk_f32_to_u8_serial`, convert and clip with SIMD.
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        result_i32x8 = _mm256_max_epi32(result_i32x8, _mm256_set1_epi32(0));
        result_i32x8 = _mm256_min_epi32(result_i32x8, _mm256_set1_epi32(255));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, result_i32x8);
        result[i + 0] = (nk_u8_t)sum_i32s[0];
        result[i + 1] = (nk_u8_t)sum_i32s[1];
        result[i + 2] = (nk_u8_t)sum_i32s[2];
        result[i + 3] = (nk_u8_t)sum_i32s[3];
        result[i + 4] = (nk_u8_t)sum_i32s[4];
        result[i + 5] = (nk_u8_t)sum_i32s[5];
        result[i + 6] = (nk_u8_t)sum_i32s[6];
        result[i + 7] = (nk_u8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i];
        nk_f32_t sum = alpha_val * ai + beta_val;
        nk_f32_to_u8_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_blend_u8_haswell(             //
    nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_each_sum_u8_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_u8_haswell(a, n, alpha, &zero, result); }
        else { nk_each_scale_u8_haswell(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);
    int sum_i32s[8], a_i32s[8], b_i32s[8];

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        b_i32s[0] = b[i + 0], b_i32s[1] = b[i + 1], b_i32s[2] = b[i + 2], b_i32s[3] = b[i + 3], //
            b_i32s[4] = b[i + 4], b_i32s[5] = b[i + 5], b_i32s[6] = b[i + 6], b_i32s[7] = b[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on `_mm256_cvtepi32_ps`: 4cy (1/cy) @ p01.
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i *)a_i32s));
        __m256 b_f32x8 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i *)b_i32s));
        // The normal part.
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(b_f32x8, beta_f32x8, ab_f32x8);
        // Instead of serial calls to expensive `nk_f32_to_u8_serial`, convert and clip with SIMD.
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        result_i32x8 = _mm256_max_epi32(result_i32x8, _mm256_set1_epi32(0));
        result_i32x8 = _mm256_min_epi32(result_i32x8, _mm256_set1_epi32(255));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, result_i32x8);
        result[i + 0] = (nk_u8_t)sum_i32s[0];
        result[i + 1] = (nk_u8_t)sum_i32s[1];
        result[i + 2] = (nk_u8_t)sum_i32s[2];
        result[i + 3] = (nk_u8_t)sum_i32s[3];
        result[i + 4] = (nk_u8_t)sum_i32s[4];
        result[i + 5] = (nk_u8_t)sum_i32s[5];
        result[i + 6] = (nk_u8_t)sum_i32s[6];
        result[i + 7] = (nk_u8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i];
        nk_f32_t sum = alpha_val * ai + beta_val * bi;
        nk_f32_to_u8_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_fma_i8_haswell(                                 //
    nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);
    int sum_i32s[8], a_i32s[8], b_i32s[8], c_i32s[8];

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        b_i32s[0] = b[i + 0], b_i32s[1] = b[i + 1], b_i32s[2] = b[i + 2], b_i32s[3] = b[i + 3], //
            b_i32s[4] = b[i + 4], b_i32s[5] = b[i + 5], b_i32s[6] = b[i + 6], b_i32s[7] = b[i + 7];
        c_i32s[0] = c[i + 0], c_i32s[1] = c[i + 1], c_i32s[2] = c[i + 2], c_i32s[3] = c[i + 3], //
            c_i32s[4] = c[i + 4], c_i32s[5] = c[i + 5], c_i32s[6] = c[i + 6], c_i32s[7] = c[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on `_mm256_cvtepi32_ps`: 4cy (1/cy) @ p01.
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i *)a_i32s));
        __m256 b_f32x8 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i *)b_i32s));
        __m256 c_f32x8 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i *)c_i32s));
        // The normal part.
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, b_f32x8);
        __m256 abc_f32x8 = _mm256_mul_ps(ab_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(c_f32x8, beta_f32x8, abc_f32x8);
        // Instead of serial calls to expensive `nk_f32_to_u8_serial`, convert and clip with SIMD.
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        result_i32x8 = _mm256_max_epi32(result_i32x8, _mm256_set1_epi32(-128));
        result_i32x8 = _mm256_min_epi32(result_i32x8, _mm256_set1_epi32(127));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, result_i32x8);
        result[i + 0] = (nk_i8_t)sum_i32s[0];
        result[i + 1] = (nk_i8_t)sum_i32s[1];
        result[i + 2] = (nk_i8_t)sum_i32s[2];
        result[i + 3] = (nk_i8_t)sum_i32s[3];
        result[i + 4] = (nk_i8_t)sum_i32s[4];
        result[i + 5] = (nk_i8_t)sum_i32s[5];
        result[i + 6] = (nk_i8_t)sum_i32s[6];
        result[i + 7] = (nk_i8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i], ci = c[i];
        nk_f32_t sum = alpha_val * ai * bi + beta_val * ci;
        nk_f32_to_i8_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_fma_u8_haswell(                                 //
    nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_val);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_val);
    int sum_i32s[8], a_i32s[8], b_i32s[8], c_i32s[8];

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        b_i32s[0] = b[i + 0], b_i32s[1] = b[i + 1], b_i32s[2] = b[i + 2], b_i32s[3] = b[i + 3], //
            b_i32s[4] = b[i + 4], b_i32s[5] = b[i + 5], b_i32s[6] = b[i + 6], b_i32s[7] = b[i + 7];
        c_i32s[0] = c[i + 0], c_i32s[1] = c[i + 1], c_i32s[2] = c[i + 2], c_i32s[3] = c[i + 3], //
            c_i32s[4] = c[i + 4], c_i32s[5] = c[i + 5], c_i32s[6] = c[i + 6], c_i32s[7] = c[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on `_mm256_cvtepi32_ps`: 4cy (1/cy) @ p01.
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i *)a_i32s));
        __m256 b_f32x8 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i *)b_i32s));
        __m256 c_f32x8 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i *)c_i32s));
        // The normal part.
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, b_f32x8);
        __m256 abc_f32x8 = _mm256_mul_ps(ab_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(c_f32x8, beta_f32x8, abc_f32x8);
        // Instead of serial calls to expensive `nk_f32_to_u8_serial`, convert and clip with SIMD.
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        result_i32x8 = _mm256_max_epi32(result_i32x8, _mm256_set1_epi32(0));
        result_i32x8 = _mm256_min_epi32(result_i32x8, _mm256_set1_epi32(255));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, result_i32x8);
        result[i + 0] = (nk_u8_t)sum_i32s[0];
        result[i + 1] = (nk_u8_t)sum_i32s[1];
        result[i + 2] = (nk_u8_t)sum_i32s[2];
        result[i + 3] = (nk_u8_t)sum_i32s[3];
        result[i + 4] = (nk_u8_t)sum_i32s[4];
        result[i + 5] = (nk_u8_t)sum_i32s[5];
        result[i + 6] = (nk_u8_t)sum_i32s[6];
        result[i + 7] = (nk_u8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i], ci = c[i];
        nk_f32_t sum = alpha_val * ai * bi + beta_val * ci;
        nk_f32_to_u8_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_sum_i16_haswell(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_i16_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i a_vec = _mm256_loadu_si256((__m256i *)(a + i));
        __m256i b_vec = _mm256_loadu_si256((__m256i *)(b + i));
        __m256i sum_vec = _mm256_adds_epi16(a_vec, b_vec);
        _mm256_storeu_si256((__m256i *)(result + i), sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_i64_t ai = a[i], bi = b[i];
        nk_i64_t sum = ai + bi;
        nk_i64_to_i16_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_scale_i16_haswell(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                         nk_i16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_f32);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_f32);
    __m256 min_f32x8 = _mm256_set1_ps(-32768.0f);
    __m256 max_f32x8 = _mm256_set1_ps(32767.0f);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i *)(a + i))));
        __m256 result_f32x8 = _mm256_fmadd_ps(a_f32x8, alpha_f32x8, beta_f32x8);
        result_f32x8 = _mm256_max_ps(result_f32x8, min_f32x8);
        result_f32x8 = _mm256_min_ps(result_f32x8, max_f32x8);
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        // Casting down to 16-bit integers is tricky!
        __m128i result_i16x8 = _mm_packs_epi32(_mm256_castsi256_si128(result_i32x8),
                                               _mm256_extracti128_si256(result_i32x8, 1));
        _mm_storeu_si128((__m128i *)(result + i), result_i16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i];
        nk_f32_t sum = alpha_f32 * ai + beta_f32;
        nk_f32_to_i16_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_fma_i16_haswell(                                   //
    nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_f32);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_f32);
    __m256 min_f32x8 = _mm256_set1_ps(-32768.0f);
    __m256 max_f32x8 = _mm256_set1_ps(32767.0f);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i *)(a + i))));
        __m256 b_f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i *)(b + i))));
        __m256 c_f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i *)(c + i))));
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, b_f32x8);
        __m256 abc_f32x8 = _mm256_mul_ps(ab_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(c_f32x8, beta_f32x8, abc_f32x8);
        result_f32x8 = _mm256_max_ps(result_f32x8, min_f32x8);
        result_f32x8 = _mm256_min_ps(result_f32x8, max_f32x8);
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        // Casting down to 16-bit integers is tricky!
        __m128i result_i16x8 = _mm_packs_epi32(_mm256_castsi256_si128(result_i32x8),
                                               _mm256_extracti128_si256(result_i32x8, 1));
        _mm_storeu_si128((__m128i *)(result + i), result_i16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i], ci = c[i];
        nk_f32_t sum = alpha_f32 * ai * bi + beta_f32 * ci;
        nk_f32_to_i16_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_sum_u16_haswell(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_u16_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i a_vec = _mm256_loadu_si256((__m256i *)(a + i));
        __m256i b_vec = _mm256_loadu_si256((__m256i *)(b + i));
        __m256i sum_vec = _mm256_adds_epu16(a_vec, b_vec);
        _mm256_storeu_si256((__m256i *)(result + i), sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_u64_t ai = a[i], bi = b[i];
        nk_u64_t sum = ai + bi;
        nk_u64_to_u16_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_scale_u16_haswell(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                         nk_u16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_f32);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_f32);
    __m256 min_f32x8 = _mm256_setzero_ps();
    __m256 max_f32x8 = _mm256_set1_ps(65535.0f);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)(a + i))));
        __m256 result_f32x8 = _mm256_fmadd_ps(a_f32x8, alpha_f32x8, beta_f32x8);
        result_f32x8 = _mm256_max_ps(result_f32x8, min_f32x8);
        result_f32x8 = _mm256_min_ps(result_f32x8, max_f32x8);
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        // Casting down to 16-bit integers is tricky!
        __m128i result_u16x8 = _mm_packus_epi32(_mm256_castsi256_si128(result_i32x8),
                                                _mm256_extracti128_si256(result_i32x8, 1));
        _mm_storeu_si128((__m128i *)(result + i), result_u16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i];
        nk_f32_t sum = alpha_f32 * ai + beta_f32;
        nk_f32_to_u16_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_fma_u16_haswell(                                   //
    nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m256 alpha_f32x8 = _mm256_set1_ps(alpha_f32);
    __m256 beta_f32x8 = _mm256_set1_ps(beta_f32);
    __m256 min_f32x8 = _mm256_setzero_ps();
    __m256 max_f32x8 = _mm256_set1_ps(65535.0f);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)(a + i))));
        __m256 b_f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)(b + i))));
        __m256 c_f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i *)(c + i))));
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, b_f32x8);
        __m256 abc_f32x8 = _mm256_mul_ps(ab_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(c_f32x8, beta_f32x8, abc_f32x8);
        result_f32x8 = _mm256_max_ps(result_f32x8, min_f32x8);
        result_f32x8 = _mm256_min_ps(result_f32x8, max_f32x8);
        __m256i result_i32x8 = _mm256_cvtps_epi32(result_f32x8);
        // Casting down to 16-bit integers is tricky!
        __m128i result_u16x8 = _mm_packus_epi32(_mm256_castsi256_si128(result_i32x8),
                                                _mm256_extracti128_si256(result_i32x8, 1));
        _mm_storeu_si128((__m128i *)(result + i), result_u16x8);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai = a[i], bi = b[i], ci = c[i];
        nk_f32_t sum = alpha_f32 * ai * bi + beta_f32 * ci;
        nk_f32_to_u16_serial(&sum, result + i);
    }
}

NK_INTERNAL __m256i _mm256_adds_epi32_haswell(__m256i a, __m256i b) {
    __m256i sum_i32x8 = _mm256_add_epi32(a, b);
    __m256i a_xor_b_i32x8 = _mm256_xor_si256(a, b);
    __m256i sum_xor_a_i32x8 = _mm256_xor_si256(sum_i32x8, a);
    // ~(a^b) & (sum^a): overflow iff same-sign inputs produce different-sign result
    __m256i overflow_i32x8 = _mm256_srai_epi32(_mm256_andnot_si256(a_xor_b_i32x8, sum_xor_a_i32x8), 31);
    // Positive overflow → INT32_MAX, negative overflow → INT32_MIN
    __m256i max_i32x8 = _mm256_set1_epi32(0x7FFFFFFF);
    __m256i min_i32x8 = _mm256_set1_epi32((int)0x80000000);
    __m256i saturated_i32x8 = _mm256_blendv_epi8(max_i32x8, min_i32x8, _mm256_srai_epi32(a, 31));
    return _mm256_blendv_epi8(sum_i32x8, saturated_i32x8, overflow_i32x8);
}

NK_PUBLIC void nk_each_sum_i32_haswell(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_i32_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i a_vec = _mm256_loadu_si256((__m256i *)(a + i));
        __m256i b_vec = _mm256_loadu_si256((__m256i *)(b + i));
        __m256i sum_vec = _mm256_adds_epi32_haswell(a_vec, b_vec);
        _mm256_storeu_si256((__m256i *)(result + i), sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_i64_t ai = a[i], bi = b[i];
        nk_i64_t sum = ai + bi;
        nk_i64_to_i32_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_scale_i32_haswell(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                         nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m256d alpha_f64x4 = _mm256_set1_pd(alpha_val);
    __m256d beta_f64x4 = _mm256_set1_pd(beta_val);
    __m256d min_f64x4 = _mm256_set1_pd(-2147483648.0);
    __m256d max_f64x4 = _mm256_set1_pd(2147483647.0);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i *)(a + i)));
        __m256d result_f64x4 = _mm256_fmadd_pd(a_f64x4, alpha_f64x4, beta_f64x4);
        // Clip to the largest values representable by 32-bit integers.
        result_f64x4 = _mm256_max_pd(result_f64x4, min_f64x4);
        result_f64x4 = _mm256_min_pd(result_f64x4, max_f64x4);
        __m128i result_i32x4 = _mm256_cvtpd_epi32(result_f64x4);
        _mm_storeu_si128((__m128i *)(result + i), result_i32x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t ai = a[i];
        nk_f64_t sum = alpha_val * ai + beta_val;
        nk_f64_to_i32_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_fma_i32_haswell(                                   //
    nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m256d alpha_f64x4 = _mm256_set1_pd(alpha_val);
    __m256d beta_f64x4 = _mm256_set1_pd(beta_val);
    __m256d min_f64x4 = _mm256_set1_pd(-2147483648.0);
    __m256d max_f64x4 = _mm256_set1_pd(2147483647.0);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i *)(a + i)));
        __m256d b_f64x4 = _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i *)(b + i)));
        __m256d c_f64x4 = _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i *)(c + i)));
        __m256d ab_f64x4 = _mm256_mul_pd(a_f64x4, b_f64x4);
        __m256d ab_scaled_f64x4 = _mm256_mul_pd(ab_f64x4, alpha_f64x4);
        __m256d result_f64x4 = _mm256_fmadd_pd(c_f64x4, beta_f64x4, ab_scaled_f64x4);
        // Clip to the largest values representable by 32-bit integers.
        result_f64x4 = _mm256_max_pd(result_f64x4, min_f64x4);
        result_f64x4 = _mm256_min_pd(result_f64x4, max_f64x4);
        __m128i result_i32x4 = _mm256_cvtpd_epi32(result_f64x4);
        _mm_storeu_si128((__m128i *)(result + i), result_i32x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t ai = a[i], bi = b[i], ci = c[i];
        nk_f64_t sum = alpha_val * ai * bi + beta_val * ci;
        nk_f64_to_i32_serial(&sum, result + i);
    }
}

NK_INTERNAL __m256i _mm256_adds_epu32_haswell(__m256i a, __m256i b) {
    __m256i sum_u32x8 = _mm256_add_epi32(a, b);
    __m256i max_u32x8 = _mm256_set1_epi32((int)0xFFFFFFFF);
    // Overflow iff sum < a (unsigned wrapping). max_epu32(sum, a) != sum means overflow.
    __m256i no_overflow_u32x8 = _mm256_cmpeq_epi32(_mm256_max_epu32(sum_u32x8, a), sum_u32x8);
    return _mm256_blendv_epi8(max_u32x8, sum_u32x8, no_overflow_u32x8);
}

NK_INTERNAL __m256d _mm256_cvtepu32_pd_haswell(__m128i a) {
    // TODO: Converting unsigned 32-bit integers to double-precision floats isn't trivial in AVX2.
    // Let's convert the lower 31 bits to a double-precision float.
    // And then conditionally add 2³¹ to the result if the MSB is set.
    //
    //  __m256d result = _mm256_cvtepi32_pd(_mm_and_si128(a, _mm_set1_epi32(0x7FFFFFFF)));
    //  int should_increment = (_mm_movemask_epi8(a) & 0x8888);
    //  should_increment = should_increment / 0x8888; // Transform something like 0b1000100010001000 to 0b1111
    //  __m256d incremented = _mm256_add_pd(result, _mm256_set1_pd(2147483648.0));
    //  result = _mm256_blend_pd(result, incremented, should_increment);
    nk_u32_t from[4];
    nk_f64_t to[4];
    _mm_storeu_si128((__m128i *)from, a);
    to[0] = (nk_f64_t)from[0];
    to[1] = (nk_f64_t)from[1];
    to[2] = (nk_f64_t)from[2];
    to[3] = (nk_f64_t)from[3];
    return _mm256_loadu_pd(to);
}

NK_INTERNAL __m128i _mm256_cvtpd_epu32_haswell(__m256d a) {
    //? For now let's avoid SIMD and just use serial conversion.
    nk_f64_t from[4];
    nk_u32_t to[4];
    _mm256_storeu_pd(from, a);
    to[0] = (nk_u32_t)from[0];
    to[1] = (nk_u32_t)from[1];
    to[2] = (nk_u32_t)from[2];
    to[3] = (nk_u32_t)from[3];
    return _mm_loadu_si128((__m128i *)to);
}

NK_PUBLIC void nk_each_sum_u32_haswell(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_u32_t *result) {
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i a_vec = _mm256_loadu_si256((__m256i *)(a + i));
        __m256i b_vec = _mm256_loadu_si256((__m256i *)(b + i));
        __m256i sum_vec = _mm256_adds_epu32_haswell(a_vec, b_vec);
        _mm256_storeu_si256((__m256i *)(result + i), sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_i64_t ai = a[i], bi = b[i];
        nk_i64_t sum = ai + bi;
        nk_i64_to_u32_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_scale_u32_haswell(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                         nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m256d alpha_f64x4 = _mm256_set1_pd(alpha_val);
    __m256d beta_f64x4 = _mm256_set1_pd(beta_val);
    __m256d min_f64x4 = _mm256_set1_pd(0);
    __m256d max_f64x4 = _mm256_set1_pd(4294967295.0);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_cvtepu32_pd_haswell(_mm_loadu_si128((__m128i *)(a + i)));
        __m256d result_f64x4 = _mm256_fmadd_pd(a_f64x4, alpha_f64x4, beta_f64x4);
        // Clip to the largest values representable by 32-bit integers.
        result_f64x4 = _mm256_max_pd(result_f64x4, min_f64x4);
        result_f64x4 = _mm256_min_pd(result_f64x4, max_f64x4);
        __m128i result_u32x4 = _mm256_cvtpd_epu32_haswell(result_f64x4);
        _mm_storeu_si128((__m128i *)(result + i), result_u32x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t ai = a[i];
        nk_f64_t sum = alpha_val * ai + beta_val;
        nk_f64_to_u32_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_fma_u32_haswell(                                   //
    nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m256d alpha_f64x4 = _mm256_set1_pd(alpha_val);
    __m256d beta_f64x4 = _mm256_set1_pd(beta_val);
    __m256d min_f64x4 = _mm256_set1_pd(0);
    __m256d max_f64x4 = _mm256_set1_pd(4294967295.0);

    // The main loop:
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = _mm256_cvtepu32_pd_haswell(_mm_loadu_si128((__m128i *)(a + i)));
        __m256d b_f64x4 = _mm256_cvtepu32_pd_haswell(_mm_loadu_si128((__m128i *)(b + i)));
        __m256d c_f64x4 = _mm256_cvtepu32_pd_haswell(_mm_loadu_si128((__m128i *)(c + i)));
        __m256d ab_f64x4 = _mm256_mul_pd(a_f64x4, b_f64x4);
        __m256d ab_scaled_f64x4 = _mm256_mul_pd(ab_f64x4, alpha_f64x4);
        __m256d result_f64x4 = _mm256_fmadd_pd(c_f64x4, beta_f64x4, ab_scaled_f64x4);
        // Clip to the largest values representable by 32-bit integers.
        result_f64x4 = _mm256_max_pd(result_f64x4, min_f64x4);
        result_f64x4 = _mm256_min_pd(result_f64x4, max_f64x4);
        __m128i result_u32x4 = _mm256_cvtpd_epu32_haswell(result_f64x4);
        _mm_storeu_si128((__m128i *)(result + i), result_u32x4);
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f64_t ai = a[i], bi = b[i], ci = c[i];
        nk_f64_t sum = alpha_val * ai * bi + beta_val * ci;
        nk_f64_to_u32_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_sum_e4m3_haswell(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_e4m3_t *result) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_e4m3x8 = _mm_loadl_epi64((__m128i const *)(a + i));
        __m128i b_e4m3x8 = _mm_loadl_epi64((__m128i const *)(b + i));
        __m256 a_f32x8 = nk_e4m3x8_to_f32x8_haswell_(a_e4m3x8);
        __m256 b_f32x8 = nk_e4m3x8_to_f32x8_haswell_(b_e4m3x8);
        __m256 result_f32x8 = _mm256_add_ps(a_f32x8, b_f32x8);
        __m128i result_e4m3x8 = nk_f32x8_to_e4m3x8_haswell_(result_f32x8);
        _mm_storel_epi64((__m128i *)(result + i), result_e4m3x8);
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_e4m3_to_f32_serial(a + i, &ai);
        nk_e4m3_to_f32_serial(b + i, &bi);
        nk_f32_t sum = ai + bi;
        nk_f32_to_e4m3_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_sum_e5m2_haswell(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_e5m2_t *result) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_e5m2x8 = _mm_loadl_epi64((__m128i const *)(a + i));
        __m128i b_e5m2x8 = _mm_loadl_epi64((__m128i const *)(b + i));
        __m256 a_f32x8 = nk_e5m2x8_to_f32x8_haswell_(a_e5m2x8);
        __m256 b_f32x8 = nk_e5m2x8_to_f32x8_haswell_(b_e5m2x8);
        __m256 result_f32x8 = _mm256_add_ps(a_f32x8, b_f32x8);
        __m128i result_e5m2x8 = nk_f32x8_to_e5m2x8_haswell_(result_f32x8);
        _mm_storel_epi64((__m128i *)(result + i), result_e5m2x8);
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_e5m2_to_f32_serial(a + i, &ai);
        nk_e5m2_to_f32_serial(b + i, &bi);
        nk_f32_t sum = ai + bi;
        nk_f32_to_e5m2_serial(&sum, result + i);
    }
}

NK_PUBLIC void nk_each_scale_e4m3_haswell(nk_e4m3_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                          nk_e4m3_t *result) {
    __m256 alpha_f32x8 = _mm256_set1_ps(*alpha);
    __m256 beta_f32x8 = _mm256_set1_ps(*beta);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_e4m3x8 = _mm_loadl_epi64((__m128i const *)(a + i));
        __m256 a_f32x8 = nk_e4m3x8_to_f32x8_haswell_(a_e4m3x8);
        // FP8 rounding note: FMA is acceptable here because scale computes (α × a + β),
        // a single multiply-add operation where single-rounding preserves accuracy.
        __m256 result_f32x8 = _mm256_fmadd_ps(a_f32x8, alpha_f32x8, beta_f32x8);
        __m128i result_e4m3x8 = nk_f32x8_to_e4m3x8_haswell_(result_f32x8);
        _mm_storel_epi64((__m128i *)(result + i), result_e4m3x8);
    }
    for (; i < n; ++i) {
        nk_f32_t ai;
        nk_e4m3_to_f32_serial(a + i, &ai);
        nk_f32_t scaled = *alpha * ai + *beta;
        nk_f32_to_e4m3_serial(&scaled, result + i);
    }
}

NK_PUBLIC void nk_each_scale_e5m2_haswell(nk_e5m2_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                          nk_e5m2_t *result) {
    __m256 alpha_f32x8 = _mm256_set1_ps(*alpha);
    __m256 beta_f32x8 = _mm256_set1_ps(*beta);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_e5m2x8 = _mm_loadl_epi64((__m128i const *)(a + i));
        __m256 a_f32x8 = nk_e5m2x8_to_f32x8_haswell_(a_e5m2x8);
        // FP8 rounding note: FMA is acceptable here because scale computes (α × a + β),
        // a single multiply-add operation where single-rounding preserves accuracy.
        __m256 result_f32x8 = _mm256_fmadd_ps(a_f32x8, alpha_f32x8, beta_f32x8);
        __m128i result_e5m2x8 = nk_f32x8_to_e5m2x8_haswell_(result_f32x8);
        _mm_storel_epi64((__m128i *)(result + i), result_e5m2x8);
    }
    for (; i < n; ++i) {
        nk_f32_t ai;
        nk_e5m2_to_f32_serial(a + i, &ai);
        nk_f32_t scaled = *alpha * ai + *beta;
        nk_f32_to_e5m2_serial(&scaled, result + i);
    }
}

NK_PUBLIC void nk_each_blend_e4m3_haswell(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                          nk_f32_t const *beta, nk_e4m3_t *result) {
    __m256 alpha_f32x8 = _mm256_set1_ps(*alpha);
    __m256 beta_f32x8 = _mm256_set1_ps(*beta);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_e4m3x8 = _mm_loadl_epi64((__m128i const *)(a + i));
        __m128i b_e4m3x8 = _mm_loadl_epi64((__m128i const *)(b + i));
        __m256 a_f32x8 = nk_e4m3x8_to_f32x8_haswell_(a_e4m3x8);
        __m256 b_f32x8 = nk_e4m3x8_to_f32x8_haswell_(b_e4m3x8);
        __m256 a_scaled_f32x8 = _mm256_mul_ps(a_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(b_f32x8, beta_f32x8, a_scaled_f32x8);
        __m128i result_e4m3x8 = nk_f32x8_to_e4m3x8_haswell_(result_f32x8);
        _mm_storel_epi64((__m128i *)(result + i), result_e4m3x8);
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_e4m3_to_f32_serial(a + i, &ai);
        nk_e4m3_to_f32_serial(b + i, &bi);
        nk_f32_t blended = *alpha * ai + *beta * bi;
        nk_f32_to_e4m3_serial(&blended, result + i);
    }
}

NK_PUBLIC void nk_each_blend_e5m2_haswell(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                          nk_f32_t const *beta, nk_e5m2_t *result) {
    __m256 alpha_f32x8 = _mm256_set1_ps(*alpha);
    __m256 beta_f32x8 = _mm256_set1_ps(*beta);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_e5m2x8 = _mm_loadl_epi64((__m128i const *)(a + i));
        __m128i b_e5m2x8 = _mm_loadl_epi64((__m128i const *)(b + i));
        __m256 a_f32x8 = nk_e5m2x8_to_f32x8_haswell_(a_e5m2x8);
        __m256 b_f32x8 = nk_e5m2x8_to_f32x8_haswell_(b_e5m2x8);
        __m256 a_scaled_f32x8 = _mm256_mul_ps(a_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(b_f32x8, beta_f32x8, a_scaled_f32x8);
        __m128i result_e5m2x8 = nk_f32x8_to_e5m2x8_haswell_(result_f32x8);
        _mm_storel_epi64((__m128i *)(result + i), result_e5m2x8);
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_e5m2_to_f32_serial(a + i, &ai);
        nk_e5m2_to_f32_serial(b + i, &bi);
        nk_f32_t blended = *alpha * ai + *beta * bi;
        nk_f32_to_e5m2_serial(&blended, result + i);
    }
}

NK_PUBLIC void nk_each_fma_e4m3_haswell(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_e4m3_t const *c, nk_size_t n,
                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_e4m3_t *result) {
    __m256 alpha_f32x8 = _mm256_set1_ps(*alpha);
    __m256 beta_f32x8 = _mm256_set1_ps(*beta);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_e4m3x8 = _mm_loadl_epi64((__m128i const *)(a + i));
        __m128i b_e4m3x8 = _mm_loadl_epi64((__m128i const *)(b + i));
        __m128i c_e4m3x8 = _mm_loadl_epi64((__m128i const *)(c + i));
        __m256 a_f32x8 = nk_e4m3x8_to_f32x8_haswell_(a_e4m3x8);
        __m256 b_f32x8 = nk_e4m3x8_to_f32x8_haswell_(b_e4m3x8);
        __m256 c_f32x8 = nk_e4m3x8_to_f32x8_haswell_(c_e4m3x8);
        // FP8 rounding note: Hybrid approach - use separate MUL for (a × b) and (α × a × b) to
        // preserve intermediate rounding, then FMA for final addition since it matches scalar
        // semantics of (α × a × b + β × c) when the multiply term is already computed.
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, b_f32x8);
        __m256 ab_scaled_f32x8 = _mm256_mul_ps(ab_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(c_f32x8, beta_f32x8, ab_scaled_f32x8);
        __m128i result_e4m3x8 = nk_f32x8_to_e4m3x8_haswell_(result_f32x8);
        _mm_storel_epi64((__m128i *)(result + i), result_e4m3x8);
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi, ci;
        nk_e4m3_to_f32_serial(a + i, &ai);
        nk_e4m3_to_f32_serial(b + i, &bi);
        nk_e4m3_to_f32_serial(c + i, &ci);
        nk_f32_t fma = *alpha * ai * bi + *beta * ci;
        nk_f32_to_e4m3_serial(&fma, result + i);
    }
}

NK_PUBLIC void nk_each_fma_e5m2_haswell(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_e5m2_t const *c, nk_size_t n,
                                        nk_f32_t const *alpha, nk_f32_t const *beta, nk_e5m2_t *result) {
    __m256 alpha_f32x8 = _mm256_set1_ps(*alpha);
    __m256 beta_f32x8 = _mm256_set1_ps(*beta);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_e5m2x8 = _mm_loadl_epi64((__m128i const *)(a + i));
        __m128i b_e5m2x8 = _mm_loadl_epi64((__m128i const *)(b + i));
        __m128i c_e5m2x8 = _mm_loadl_epi64((__m128i const *)(c + i));
        __m256 a_f32x8 = nk_e5m2x8_to_f32x8_haswell_(a_e5m2x8);
        __m256 b_f32x8 = nk_e5m2x8_to_f32x8_haswell_(b_e5m2x8);
        __m256 c_f32x8 = nk_e5m2x8_to_f32x8_haswell_(c_e5m2x8);
        // FP8 rounding note: Hybrid approach - use separate MUL for (a × b) and (α × a × b) to
        // preserve intermediate rounding, then FMA for final addition since it matches scalar
        // semantics of (α × a × b + β × c) when the multiply term is already computed.
        __m256 ab_f32x8 = _mm256_mul_ps(a_f32x8, b_f32x8);
        __m256 ab_scaled_f32x8 = _mm256_mul_ps(ab_f32x8, alpha_f32x8);
        __m256 result_f32x8 = _mm256_fmadd_ps(c_f32x8, beta_f32x8, ab_scaled_f32x8);
        __m128i result_e5m2x8 = nk_f32x8_to_e5m2x8_haswell_(result_f32x8);
        _mm_storel_epi64((__m128i *)(result + i), result_e5m2x8);
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi, ci;
        nk_e5m2_to_f32_serial(a + i, &ai);
        nk_e5m2_to_f32_serial(b + i, &bi);
        nk_e5m2_to_f32_serial(c + i, &ci);
        nk_f32_t fma = *alpha * ai * bi + *beta * ci;
        nk_f32_to_e5m2_serial(&fma, result + i);
    }
}

NK_PUBLIC void nk_each_scale_f32c_haswell(nk_f32c_t const *a, nk_size_t n, nk_f32c_t const *alpha,
                                          nk_f32c_t const *beta, nk_f32c_t *result) {
    nk_f32_t const *a_f32 = (nk_f32_t const *)a;
    nk_f32_t *result_f32 = (nk_f32_t *)result;
    __m256 alpha_real_f32x8 = _mm256_set1_ps(alpha->real);
    __m256 alpha_imag_f32x8 = _mm256_set1_ps(alpha->imag);
    __m256 beta_f32x8 = _mm256_setr_ps(beta->real, beta->imag, beta->real, beta->imag, beta->real, beta->imag,
                                       beta->real, beta->imag);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256 a_f32x8 = _mm256_loadu_ps(a_f32 + 2 * i);
        __m256 a_swapped_f32x8 = _mm256_permute_ps(a_f32x8, 0xB1);
        __m256 temp_f32x8 = _mm256_mul_ps(alpha_imag_f32x8, a_swapped_f32x8);
        __m256 y_f32x8 = _mm256_fmaddsub_ps(alpha_real_f32x8, a_f32x8, temp_f32x8);
        y_f32x8 = _mm256_add_ps(y_f32x8, beta_f32x8);
        _mm256_storeu_ps(result_f32 + 2 * i, y_f32x8);
    }
    for (; i < n; i++) {
        nk_f32_t a_real = a[i].real, a_imag = a[i].imag;
        result[i].real = alpha->real * a_real - alpha->imag * a_imag + beta->real;
        result[i].imag = alpha->real * a_imag + alpha->imag * a_real + beta->imag;
    }
}

NK_PUBLIC void nk_each_scale_f64c_haswell(nk_f64c_t const *a, nk_size_t n, nk_f64c_t const *alpha,
                                          nk_f64c_t const *beta, nk_f64c_t *result) {
    nk_f64_t const *a_f64 = (nk_f64_t const *)a;
    nk_f64_t *result_f64 = (nk_f64_t *)result;
    __m256d alpha_real_f64x4 = _mm256_set1_pd(alpha->real);
    __m256d alpha_imag_f64x4 = _mm256_set1_pd(alpha->imag);
    __m256d beta_f64x4 = _mm256_setr_pd(beta->real, beta->imag, beta->real, beta->imag);
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m256d a_f64x4 = _mm256_loadu_pd(a_f64 + 2 * i);
        __m256d a_swapped_f64x4 = _mm256_permute_pd(a_f64x4, 0x5);
        __m256d temp_f64x4 = _mm256_mul_pd(alpha_imag_f64x4, a_swapped_f64x4);
        __m256d y_f64x4 = _mm256_fmaddsub_pd(alpha_real_f64x4, a_f64x4, temp_f64x4);
        y_f64x4 = _mm256_add_pd(y_f64x4, beta_f64x4);
        _mm256_storeu_pd(result_f64 + 2 * i, y_f64x4);
    }
    for (; i < n; i++) {
        nk_f64_t a_real = a[i].real, a_imag = a[i].imag;
        result[i].real = alpha->real * a_real - alpha->imag * a_imag + beta->real;
        result[i].imag = alpha->real * a_imag + alpha->imag * a_real + beta->imag;
    }
}

NK_PUBLIC void nk_each_blend_f32c_haswell(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t const *alpha,
                                          nk_f32c_t const *beta, nk_f32c_t *result) {
    nk_f32_t const *a_f32 = (nk_f32_t const *)a;
    nk_f32_t const *b_f32 = (nk_f32_t const *)b;
    nk_f32_t *result_f32 = (nk_f32_t *)result;
    __m256 alpha_real_f32x8 = _mm256_set1_ps(alpha->real);
    __m256 alpha_imag_f32x8 = _mm256_set1_ps(alpha->imag);
    __m256 beta_real_f32x8 = _mm256_set1_ps(beta->real);
    __m256 beta_imag_f32x8 = _mm256_set1_ps(beta->imag);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256 a_f32x8 = _mm256_loadu_ps(a_f32 + 2 * i);
        __m256 b_f32x8 = _mm256_loadu_ps(b_f32 + 2 * i);
        __m256 a_swapped_f32x8 = _mm256_permute_ps(a_f32x8, 0xB1);
        __m256 ta_f32x8 = _mm256_mul_ps(alpha_imag_f32x8, a_swapped_f32x8);
        __m256 ya_f32x8 = _mm256_fmaddsub_ps(alpha_real_f32x8, a_f32x8, ta_f32x8);
        __m256 b_swapped_f32x8 = _mm256_permute_ps(b_f32x8, 0xB1);
        __m256 tb_f32x8 = _mm256_mul_ps(beta_imag_f32x8, b_swapped_f32x8);
        __m256 yb_f32x8 = _mm256_fmaddsub_ps(beta_real_f32x8, b_f32x8, tb_f32x8);
        _mm256_storeu_ps(result_f32 + 2 * i, _mm256_add_ps(ya_f32x8, yb_f32x8));
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

NK_PUBLIC void nk_each_blend_f64c_haswell(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t const *alpha,
                                          nk_f64c_t const *beta, nk_f64c_t *result) {
    nk_f64_t const *a_f64 = (nk_f64_t const *)a;
    nk_f64_t const *b_f64 = (nk_f64_t const *)b;
    nk_f64_t *result_f64 = (nk_f64_t *)result;
    __m256d alpha_real_f64x4 = _mm256_set1_pd(alpha->real);
    __m256d alpha_imag_f64x4 = _mm256_set1_pd(alpha->imag);
    __m256d beta_real_f64x4 = _mm256_set1_pd(beta->real);
    __m256d beta_imag_f64x4 = _mm256_set1_pd(beta->imag);
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m256d a_f64x4 = _mm256_loadu_pd(a_f64 + 2 * i);
        __m256d b_f64x4 = _mm256_loadu_pd(b_f64 + 2 * i);
        __m256d a_swapped_f64x4 = _mm256_permute_pd(a_f64x4, 0x5);
        __m256d ta_f64x4 = _mm256_mul_pd(alpha_imag_f64x4, a_swapped_f64x4);
        __m256d ya_f64x4 = _mm256_fmaddsub_pd(alpha_real_f64x4, a_f64x4, ta_f64x4);
        __m256d b_swapped_f64x4 = _mm256_permute_pd(b_f64x4, 0x5);
        __m256d tb_f64x4 = _mm256_mul_pd(beta_imag_f64x4, b_swapped_f64x4);
        __m256d yb_f64x4 = _mm256_fmaddsub_pd(beta_real_f64x4, b_f64x4, tb_f64x4);
        _mm256_storeu_pd(result_f64 + 2 * i, _mm256_add_pd(ya_f64x4, yb_f64x4));
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

NK_PUBLIC void nk_each_fma_f32c_haswell(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                        nk_f32c_t const *alpha, nk_f32c_t const *beta, nk_f32c_t *result) {
    nk_f32_t const *a_f32 = (nk_f32_t const *)a;
    nk_f32_t const *b_f32 = (nk_f32_t const *)b;
    nk_f32_t const *c_f32 = (nk_f32_t const *)c;
    nk_f32_t *result_f32 = (nk_f32_t *)result;
    __m256 alpha_real_f32x8 = _mm256_set1_ps(alpha->real);
    __m256 alpha_imag_f32x8 = _mm256_set1_ps(alpha->imag);
    __m256 beta_real_f32x8 = _mm256_set1_ps(beta->real);
    __m256 beta_imag_f32x8 = _mm256_set1_ps(beta->imag);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256 a_f32x8 = _mm256_loadu_ps(a_f32 + 2 * i);
        __m256 b_f32x8 = _mm256_loadu_ps(b_f32 + 2 * i);
        __m256 c_f32x8 = _mm256_loadu_ps(c_f32 + 2 * i);
        __m256 b_swapped_f32x8 = _mm256_permute_ps(b_f32x8, 0xB1);
        __m256 a_real_f32x8 = _mm256_moveldup_ps(a_f32x8);
        __m256 a_imag_f32x8 = _mm256_movehdup_ps(a_f32x8);
        __m256 tab_f32x8 = _mm256_mul_ps(a_imag_f32x8, b_swapped_f32x8);
        __m256 ab_f32x8 = _mm256_fmaddsub_ps(a_real_f32x8, b_f32x8, tab_f32x8);
        __m256 ab_swapped_f32x8 = _mm256_permute_ps(ab_f32x8, 0xB1);
        __m256 taa_f32x8 = _mm256_mul_ps(alpha_imag_f32x8, ab_swapped_f32x8);
        __m256 ya_f32x8 = _mm256_fmaddsub_ps(alpha_real_f32x8, ab_f32x8, taa_f32x8);
        __m256 c_swapped_f32x8 = _mm256_permute_ps(c_f32x8, 0xB1);
        __m256 tbc_f32x8 = _mm256_mul_ps(beta_imag_f32x8, c_swapped_f32x8);
        __m256 yb_f32x8 = _mm256_fmaddsub_ps(beta_real_f32x8, c_f32x8, tbc_f32x8);
        _mm256_storeu_ps(result_f32 + 2 * i, _mm256_add_ps(ya_f32x8, yb_f32x8));
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

NK_PUBLIC void nk_each_fma_f64c_haswell(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                        nk_f64c_t const *alpha, nk_f64c_t const *beta, nk_f64c_t *result) {
    nk_f64_t const *a_f64 = (nk_f64_t const *)a;
    nk_f64_t const *b_f64 = (nk_f64_t const *)b;
    nk_f64_t const *c_f64 = (nk_f64_t const *)c;
    nk_f64_t *result_f64 = (nk_f64_t *)result;
    __m256d alpha_real_f64x4 = _mm256_set1_pd(alpha->real);
    __m256d alpha_imag_f64x4 = _mm256_set1_pd(alpha->imag);
    __m256d beta_real_f64x4 = _mm256_set1_pd(beta->real);
    __m256d beta_imag_f64x4 = _mm256_set1_pd(beta->imag);
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m256d a_f64x4 = _mm256_loadu_pd(a_f64 + 2 * i);
        __m256d b_f64x4 = _mm256_loadu_pd(b_f64 + 2 * i);
        __m256d c_f64x4 = _mm256_loadu_pd(c_f64 + 2 * i);
        __m256d b_swapped_f64x4 = _mm256_permute_pd(b_f64x4, 0x5);
        __m256d a_real_f64x4 = _mm256_movedup_pd(a_f64x4);
        __m256d a_imag_f64x4 = _mm256_permute_pd(a_f64x4, 0xF);
        __m256d tab_f64x4 = _mm256_mul_pd(a_imag_f64x4, b_swapped_f64x4);
        __m256d ab_f64x4 = _mm256_fmaddsub_pd(a_real_f64x4, b_f64x4, tab_f64x4);
        __m256d ab_swapped_f64x4 = _mm256_permute_pd(ab_f64x4, 0x5);
        __m256d taa_f64x4 = _mm256_mul_pd(alpha_imag_f64x4, ab_swapped_f64x4);
        __m256d ya_f64x4 = _mm256_fmaddsub_pd(alpha_real_f64x4, ab_f64x4, taa_f64x4);
        __m256d c_swapped_f64x4 = _mm256_permute_pd(c_f64x4, 0x5);
        __m256d tbc_f64x4 = _mm256_mul_pd(beta_imag_f64x4, c_swapped_f64x4);
        __m256d yb_f64x4 = _mm256_fmaddsub_pd(beta_real_f64x4, c_f64x4, tbc_f64x4);
        _mm256_storeu_pd(result_f64 + 2 * i, _mm256_add_pd(ya_f64x4, yb_f64x4));
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

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_
#endif // NK_EACH_HASWELL_H
