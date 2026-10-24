/**
 *  @brief AVX-512 BF16 implementations for the redesigned reduction API (moments).
 *  @file include/numkong/reduce/genoa.h
 *  @author Ash Vardanian
 *  @date February 12, 2026
 *
 *  @sa include/numkong/reduce.h
 *
 *  @section bf16_moments BF16 Moments
 *
 *  `_mm512_dpbf16_ps(acc, a, b)` (VDPBF16PS) computes paired bf16→f32 dot products:
 *  `acc[i] += bf16_to_f32(a[2i]) * bf16_to_f32(b[2i]) + bf16_to_f32(a[2i+1]) * bf16_to_f32(b[2i+1])`
 *  Processing 32 bf16 values into 16 f32 accumulators per instruction.
 *
 *  For sum: use ones vector (bf16 1.0 = 0x3F80).
 *  For sumsq: dot product of data with itself.
 */
#ifndef NK_REDUCE_GENOA_H
#define NK_REDUCE_GENOA_H

#if NK_TARGET_X86_
#if NK_TARGET_GENOA

#include "numkong/reduce/serial.h"
#include "numkong/cast/icelake.h" // `nk_e4m3x32_to_bf16x32_icelake_` etc.

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

NK_INTERNAL void nk_reduce_moments_bf16_genoa_contiguous_( //
    nk_bf16_t const *data_ptr, nk_size_t count,            //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    // bf16(1.0) = 0x3F80. Pack 32 of them as __m512bh.
    __m512bh ones_bf16x32 = nk_m512bh_from_m512i_(_mm512_set1_epi16(0x3F80));
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;

    for (; idx + 32 <= count; idx += 32) {
        __m512bh data_bf16x32 = nk_m512bh_from_m512i_(_mm512_loadu_si512(data_ptr + idx));
        sum_f32x16 = _mm512_dpbf16_ps(sum_f32x16, data_bf16x32, ones_bf16x32);
        sumsq_f32x16 = _mm512_dpbf16_ps(sumsq_f32x16, data_bf16x32, data_bf16x32);
    }

    // Tail: masked load for remaining elements (< 32)
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m512bh data_bf16x32 = nk_m512bh_from_m512i_(_mm512_maskz_loadu_epi16(tail_mask, data_ptr + idx));
        sum_f32x16 = _mm512_dpbf16_ps(sum_f32x16, data_bf16x32, ones_bf16x32);
        sumsq_f32x16 = _mm512_dpbf16_ps(sumsq_f32x16, data_bf16x32, data_bf16x32);
    }

    *sum_ptr = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq_ptr = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_PUBLIC void nk_reduce_moments_bf16_genoa(                            //
    nk_bf16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_bf16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_bf16_genoa(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_bf16_genoa(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                     &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum;
        *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_bf16_genoa_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_bf16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_e4m3_genoa_contiguous_( //
    nk_e4m3_t const *data_ptr, nk_size_t count,            //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    __m512bh ones_bf16x32 = nk_m512bh_from_m512i_(_mm512_set1_epi16(0x3F80)); // bf16(1.0)
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;

    for (; idx + 32 <= count; idx += 32) {
        __m256i raw_u8x32 = _mm256_loadu_si256((__m256i const *)(data_ptr + idx));
        __m512bh data_bf16x32 = nk_m512bh_from_m512i_(nk_e4m3x32_to_bf16x32_icelake_(raw_u8x32));
        sum_f32x16 = _mm512_dpbf16_ps(sum_f32x16, data_bf16x32, ones_bf16x32);
        sumsq_f32x16 = _mm512_dpbf16_ps(sumsq_f32x16, data_bf16x32, data_bf16x32);
    }

    // Tail: masked load for remaining elements (< 32)
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m256i raw_u8x32 = _mm256_maskz_loadu_epi8(tail_mask, data_ptr + idx);
        __m512bh data_bf16x32 = nk_m512bh_from_m512i_(nk_e4m3x32_to_bf16x32_icelake_(raw_u8x32));
        sum_f32x16 = _mm512_dpbf16_ps(sum_f32x16, data_bf16x32, ones_bf16x32);
        sumsq_f32x16 = _mm512_dpbf16_ps(sumsq_f32x16, data_bf16x32, data_bf16x32);
    }

    *sum_ptr = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq_ptr = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_PUBLIC void nk_reduce_moments_e4m3_genoa(                            //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e4m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e4m3_genoa(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e4m3_genoa(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                     &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum;
        *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e4m3_genoa_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e4m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_e5m2_genoa_contiguous_( //
    nk_e5m2_t const *data_ptr, nk_size_t count,            //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    __m512bh ones_bf16x32 = nk_m512bh_from_m512i_(_mm512_set1_epi16(0x3F80)); // bf16(1.0)
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m512 sumsq_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;

    for (; idx + 32 <= count; idx += 32) {
        __m256i raw_u8x32 = _mm256_loadu_si256((__m256i const *)(data_ptr + idx));
        __m512bh data_bf16x32 = nk_m512bh_from_m512i_(nk_e5m2x32_to_bf16x32_icelake_(raw_u8x32));
        sum_f32x16 = _mm512_dpbf16_ps(sum_f32x16, data_bf16x32, ones_bf16x32);
        sumsq_f32x16 = _mm512_dpbf16_ps(sumsq_f32x16, data_bf16x32, data_bf16x32);
    }

    // Tail: masked load for remaining elements (< 32)
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m256i raw_u8x32 = _mm256_maskz_loadu_epi8(tail_mask, data_ptr + idx);
        __m512bh data_bf16x32 = nk_m512bh_from_m512i_(nk_e5m2x32_to_bf16x32_icelake_(raw_u8x32));
        sum_f32x16 = _mm512_dpbf16_ps(sum_f32x16, data_bf16x32, ones_bf16x32);
        sumsq_f32x16 = _mm512_dpbf16_ps(sumsq_f32x16, data_bf16x32, data_bf16x32);
    }

    *sum_ptr = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    *sumsq_ptr = nk_reduce_add_f32x16_skylake_(sumsq_f32x16);
}

NK_PUBLIC void nk_reduce_moments_e5m2_genoa(                            //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e5m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e5m2_genoa(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e5m2_genoa(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                     &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum;
        *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e5m2_genoa_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e5m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_GENOA
#endif // NK_TARGET_X86_
#endif // NK_REDUCE_GENOA_H
