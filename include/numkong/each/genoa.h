/**
 *  @file include/numkong/each/genoa.h
 *  @author Ash Vardanian
 *  @date September 30, 2026
 *  @brief AVX-512 BF16 implementations of the row-wise RMSNorm for Genoa.
 *
 *  @sa include/numkong/each.h
 *
 *  The mean square of every group comes from the VDPBF16PS moments in @c reduce/genoa.h, which
 *  fold 32 BF16 values into 16 F32 accumulators per instruction. E4M3 inputs widen to BF16 first.
 */
#ifndef NUMKONG_EACH_GENOA_H
#define NUMKONG_EACH_GENOA_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_TARGET_GENOA

#include "numkong/types.h"
#include "numkong/cast/serial.h"  // `nk_bf16_to_f32_`, `nk_f32_to_e4m3_`
#include "numkong/reduce/genoa.h" // `nk_reduce_moments_bf16_genoa_chunked_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512bf16,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512bf16", "f16c", "fma", "bmi", "bmi2")
#endif

NUMKONG_API nk_status_t nk_each_rmsnorm_bf16_genoa(nk_bf16_t const *x, nk_f32_t const *gamma, nk_bf16_t *y,
                                                   nk_size_t rows, nk_size_t groups, nk_size_t cols,
                                                   nk_size_t x_stride_bytes, nk_size_t y_stride_bytes, nk_f32_t eps,
                                                   nk_f32_t input_scale, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t const scale_sq = (nk_f64_t)input_scale * (nk_f64_t)input_scale;
    for (nk_size_t r = 0; r != rows; ++r) {
        nk_bf16_t const *x_row = (nk_bf16_t const *)((unsigned char const *)x + r * x_stride_bytes);
        nk_bf16_t *y_row = (nk_bf16_t *)((unsigned char *)y + r * y_stride_bytes);
        for (nk_size_t group = 0; group != groups; ++group) {
            nk_bf16_t const *group_input = x_row + group * cols;
            nk_bf16_t *group_output = y_row + group * cols;
            nk_f32_t sum, sumsq;
            nk_reduce_moments_bf16_genoa_chunked_(group_input, cols, sizeof(nk_bf16_t), &sum, &sumsq);
            nk_f32_t mean_square = (nk_f32_t)(scale_sq * (nk_f64_t)sumsq / (nk_f64_t)cols) + eps;
            nk_f32_t gain = input_scale *
                            _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(1.0f), _mm_sqrt_ss(_mm_set_ss(mean_square))));
            for (nk_size_t c = 0; c != cols; ++c) {
                nk_f32_t value;
                nk_bf16_to_f32_(group_input + c, &value);
                nk_f32_t result = value * gain * (gamma ? gamma[c] : 1.0f);
                nk_f32_to_bf16_(&result, group_output + c);
            }
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_each_rmsnorm_e4m3_genoa(nk_e4m3_t const *x, nk_f32_t const *gamma, nk_e4m3_t *y,
                                                   nk_size_t rows, nk_size_t groups, nk_size_t cols,
                                                   nk_size_t x_stride_bytes, nk_size_t y_stride_bytes, nk_f32_t eps,
                                                   nk_f32_t input_scale, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t const scale_sq = (nk_f64_t)input_scale * (nk_f64_t)input_scale;
    for (nk_size_t r = 0; r != rows; ++r) {
        nk_e4m3_t const *x_row = (nk_e4m3_t const *)((unsigned char const *)x + r * x_stride_bytes);
        nk_e4m3_t *y_row = (nk_e4m3_t *)((unsigned char *)y + r * y_stride_bytes);
        for (nk_size_t group = 0; group != groups; ++group) {
            nk_e4m3_t const *group_input = x_row + group * cols;
            nk_e4m3_t *group_output = y_row + group * cols;
            nk_f32_t sum, sumsq;
            nk_reduce_moments_e4m3_genoa_chunked_(group_input, cols, sizeof(nk_e4m3_t), &sum, &sumsq);
            nk_f32_t mean_square = (nk_f32_t)(scale_sq * (nk_f64_t)sumsq / (nk_f64_t)cols) + eps;
            nk_f32_t gain = input_scale *
                            _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(1.0f), _mm_sqrt_ss(_mm_set_ss(mean_square))));
            for (nk_size_t c = 0; c != cols; ++c) {
                nk_f32_t value;
                nk_e4m3_to_f32_(group_input + c, &value);
                nk_f32_t result = value * gain * (gamma ? gamma[c] : 1.0f);
                nk_f32_to_e4m3_(&result, group_output + c);
            }
        }
    }
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_GENOA
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_EACH_GENOA_H
