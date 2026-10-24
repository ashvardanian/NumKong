/**
 *  @brief AVX-512 VNNI implementations for the redesigned reduction API (moments).
 *  @file include/numkong/reduce/icelake.h
 *  @author Ash Vardanian
 *  @date February 12, 2026
 *
 *  @sa include/numkong/reduce.h
 *
 *  @section vnni_advantage VNNI Advantage
 *
 *  `_mm512_dpwssd_epi32(acc, a, b)` (VPDPWSSD) fuses `acc + _mm512_madd_epi16(a, b)`
 *  into one instruction (5cy @ p0 on Ice Lake, 4cy @ p01 on Genoa), saving one
 *  `_mm512_add_epi32` per call vs the Skylake `madd + add` pair.
 */
#ifndef NK_REDUCE_ICELAKE_H
#define NK_REDUCE_ICELAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_ICELAKE

#include "numkong/reduce/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                   \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,avx512vbmi,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vnni", "avx512vbmi", "f16c", "fma", \
                   "bmi", "bmi2")
#endif

NK_INTERNAL void nk_reduce_moments_i8_icelake_contiguous_( //
    nk_i8_t const *data_ptr, nk_size_t count,              //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __m512i bias_i8x64 = _mm512_set1_epi8((char)0x80);
    __m512i zero_i8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_low_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_high_i32x16 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data_ptr + idx);
        __m512i unsigned_i8x64 = _mm512_xor_si512(data_i8x64, bias_i8x64);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(unsigned_i8x64, zero_i8x64));
        __m512i low_i16x32 = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(data_i8x64));
        __m512i high_i16x32 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(data_i8x64, 1));
        sumsq_low_i32x16 = _mm512_dpwssd_epi32(sumsq_low_i32x16, low_i16x32, low_i16x32);
        sumsq_high_i32x16 = _mm512_dpwssd_epi32(sumsq_high_i32x16, high_i16x32, high_i16x32);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i data_i8x64 = _mm512_maskz_loadu_epi8(tail_mask, data_ptr + idx);
        __m512i unsigned_i8x64 = _mm512_xor_si512(data_i8x64, _mm512_maskz_mov_epi8(tail_mask, bias_i8x64));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(unsigned_i8x64, zero_i8x64));
        __m512i low_i16x32 = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(data_i8x64));
        __m512i high_i16x32 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(data_i8x64, 1));
        sumsq_low_i32x16 = _mm512_dpwssd_epi32(sumsq_low_i32x16, low_i16x32, low_i16x32);
        sumsq_high_i32x16 = _mm512_dpwssd_epi32(sumsq_high_i32x16, high_i16x32, high_i16x32);
    }
    sumsq_low_i32x16 = _mm512_add_epi32(sumsq_low_i32x16, sumsq_high_i32x16);
    __m512i sumsq_i64x8 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sumsq_low_i32x16));
    sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sumsq_low_i32x16, 1)));
    *sum_ptr = (nk_i64_t)nk_reduce_add_u64x8_skylake_(sum_u64x8) - (nk_i64_t)128 * (nk_i64_t)count;
    *sumsq_ptr = (nk_u64_t)nk_reduce_add_i64x8_skylake_(sumsq_i64x8);
}

NK_INTERNAL void nk_reduce_moments_i8_icelake_strided_(                  //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __mmask64 stride_mask_m64 = nk_stride_mask_u1x64_(stride_elements);
    __m512i masked_bias_i8x64 = _mm512_maskz_mov_epi8(stride_mask_m64, _mm512_set1_epi8((char)0x80));
    __m512i zero_i8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_low_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_high_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = (nk_size_t)_mm_popcnt_u64((nk_u64_t)stride_mask_m64) * stride_elements;
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m512i data_i8x64 = _mm512_maskz_loadu_epi8(stride_mask_m64, data_ptr + idx_scalars);
        __m512i unsigned_i8x64 = _mm512_xor_si512(data_i8x64, masked_bias_i8x64);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(unsigned_i8x64, zero_i8x64));
        __m512i low_i16x32 = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(data_i8x64));
        __m512i high_i16x32 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(data_i8x64, 1));
        sumsq_low_i32x16 = _mm512_dpwssd_epi32(sumsq_low_i32x16, low_i16x32, low_i16x32);
        sumsq_high_i32x16 = _mm512_dpwssd_epi32(sumsq_high_i32x16, high_i16x32, high_i16x32);
    }
    nk_size_t remaining_scalars = total_scalars - idx_scalars;
    if (remaining_scalars > 0) {
        __mmask64 tail_mask = stride_mask_m64 & _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining_scalars);
        __m512i data_i8x64 = _mm512_maskz_loadu_epi8(tail_mask, data_ptr + idx_scalars);
        __m512i unsigned_i8x64 = _mm512_xor_si512(data_i8x64,
                                                  _mm512_maskz_mov_epi8(tail_mask, _mm512_set1_epi8((char)0x80)));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(unsigned_i8x64, zero_i8x64));
        __m512i low_i16x32 = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(data_i8x64));
        __m512i high_i16x32 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(data_i8x64, 1));
        sumsq_low_i32x16 = _mm512_dpwssd_epi32(sumsq_low_i32x16, low_i16x32, low_i16x32);
        sumsq_high_i32x16 = _mm512_dpwssd_epi32(sumsq_high_i32x16, high_i16x32, high_i16x32);
    }
    sumsq_low_i32x16 = _mm512_add_epi32(sumsq_low_i32x16, sumsq_high_i32x16);
    __m512i sumsq_i64x8 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sumsq_low_i32x16));
    sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sumsq_low_i32x16, 1)));
    *sum_ptr = (nk_i64_t)nk_reduce_add_u64x8_skylake_(sum_u64x8) - (nk_i64_t)128 * (nk_i64_t)count;
    *sumsq_ptr = (nk_u64_t)nk_reduce_add_i64x8_skylake_(sumsq_i64x8);
}

NK_PUBLIC void nk_reduce_moments_i8_icelake(                          //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_i8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum, right_sum;
        nk_u64_t left_sumsq, right_sumsq;
        nk_reduce_moments_i8_icelake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i8_icelake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                     &right_sum, &right_sumsq);
        *sum_ptr = nk_i64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i8_icelake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 16)
        nk_reduce_moments_i8_icelake_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_i8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_u8_icelake_contiguous_( //
    nk_u8_t const *data_ptr, nk_size_t count,              //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __m512i zero_u8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_low_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_high_i32x16 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_u8x64 = _mm512_loadu_si512(data_ptr + idx);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(data_u8x64, zero_u8x64));
        __m512i low_i16x32 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(data_u8x64));
        __m512i high_i16x32 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(data_u8x64, 1));
        sumsq_low_i32x16 = _mm512_dpwssd_epi32(sumsq_low_i32x16, low_i16x32, low_i16x32);
        sumsq_high_i32x16 = _mm512_dpwssd_epi32(sumsq_high_i32x16, high_i16x32, high_i16x32);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i data_u8x64 = _mm512_maskz_loadu_epi8(tail_mask, data_ptr + idx);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(data_u8x64, zero_u8x64));
        __m512i low_i16x32 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(data_u8x64));
        __m512i high_i16x32 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(data_u8x64, 1));
        sumsq_low_i32x16 = _mm512_dpwssd_epi32(sumsq_low_i32x16, low_i16x32, low_i16x32);
        sumsq_high_i32x16 = _mm512_dpwssd_epi32(sumsq_high_i32x16, high_i16x32, high_i16x32);
    }
    sumsq_low_i32x16 = _mm512_add_epi32(sumsq_low_i32x16, sumsq_high_i32x16);
    __m512i sumsq_u64x8 = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(sumsq_low_i32x16));
    sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(sumsq_low_i32x16, 1)));
    *sum_ptr = nk_reduce_add_u64x8_skylake_(sum_u64x8);
    *sumsq_ptr = nk_reduce_add_u64x8_skylake_(sumsq_u64x8);
}

NK_INTERNAL void nk_reduce_moments_u8_icelake_strided_(                  //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __mmask64 stride_mask_m64 = nk_stride_mask_u1x64_(stride_elements);
    __m512i zero_u8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i sumsq_low_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_high_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = (nk_size_t)_mm_popcnt_u64((nk_u64_t)stride_mask_m64) * stride_elements;
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m512i data_u8x64 = _mm512_maskz_loadu_epi8(stride_mask_m64, data_ptr + idx_scalars);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(data_u8x64, zero_u8x64));
        __m512i low_i16x32 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(data_u8x64));
        __m512i high_i16x32 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(data_u8x64, 1));
        sumsq_low_i32x16 = _mm512_dpwssd_epi32(sumsq_low_i32x16, low_i16x32, low_i16x32);
        sumsq_high_i32x16 = _mm512_dpwssd_epi32(sumsq_high_i32x16, high_i16x32, high_i16x32);
    }
    nk_size_t remaining_scalars = total_scalars - idx_scalars;
    if (remaining_scalars > 0) {
        __mmask64 tail_mask = stride_mask_m64 & _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining_scalars);
        __m512i data_u8x64 = _mm512_maskz_loadu_epi8(tail_mask, data_ptr + idx_scalars);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(data_u8x64, zero_u8x64));
        __m512i low_i16x32 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(data_u8x64));
        __m512i high_i16x32 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(data_u8x64, 1));
        sumsq_low_i32x16 = _mm512_dpwssd_epi32(sumsq_low_i32x16, low_i16x32, low_i16x32);
        sumsq_high_i32x16 = _mm512_dpwssd_epi32(sumsq_high_i32x16, high_i16x32, high_i16x32);
    }
    sumsq_low_i32x16 = _mm512_add_epi32(sumsq_low_i32x16, sumsq_high_i32x16);
    __m512i sumsq_u64x8 = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(sumsq_low_i32x16));
    sumsq_u64x8 = _mm512_add_epi64(sumsq_u64x8, _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(sumsq_low_i32x16, 1)));
    *sum_ptr = nk_reduce_add_u64x8_skylake_(sum_u64x8);
    *sumsq_ptr = nk_reduce_add_u64x8_skylake_(sumsq_u64x8);
}

NK_PUBLIC void nk_reduce_moments_u8_icelake(                          //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_u8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U8_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u8_icelake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u8_icelake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                     &right_sum, &right_sumsq);
        *sum_ptr = nk_u64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u8_icelake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 16)
        nk_reduce_moments_u8_icelake_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_u8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_i16_icelake_contiguous_( //
    nk_i16_t const *data_ptr, nk_size_t count,              //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    // Sum: VPDPWSSD(acc, data, ones) accumulates in i32 — safe for (NK_I16_MAX+1)*32 elements.
    // Sumsq: VPDPWSSD(zero, data, data) → fresh i32, widen to i64 each iteration.
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_i64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_i16x32 = _mm512_loadu_si512(data_ptr + idx);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_i16x32, ones_i16x32);
        __m512i sq_i32x16 = _mm512_dpwssd_epi32(_mm512_setzero_si512(), data_i16x32, data_i16x32);
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sq_i32x16)));
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sq_i32x16, 1)));
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m512i data_i16x32 = _mm512_maskz_loadu_epi16(tail_mask, data_ptr + idx);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_i16x32, ones_i16x32);
        __m512i sq_i32x16 = _mm512_dpwssd_epi32(_mm512_setzero_si512(), data_i16x32, data_i16x32);
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sq_i32x16)));
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sq_i32x16, 1)));
    }
    __m512i sum_i64x8 = _mm512_add_epi64(                                 //
        _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sum_i32x16)),        //
        _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sum_i32x16, 1))); //
    *sum_ptr = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    *sumsq_ptr = (nk_u64_t)nk_reduce_add_i64x8_skylake_(sumsq_i64x8);
}

NK_INTERNAL void nk_reduce_moments_i16_icelake_strided_(                  //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __mmask32 stride_mask_m32 = nk_stride_mask_b16x32_(stride_elements);
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_i64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = (nk_size_t)_mm_popcnt_u64((nk_u64_t)stride_mask_m32) * stride_elements;
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m512i data_i16x32 = _mm512_maskz_loadu_epi16(stride_mask_m32, data_ptr + idx_scalars);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_i16x32, ones_i16x32);
        __m512i sq_i32x16 = _mm512_dpwssd_epi32(_mm512_setzero_si512(), data_i16x32, data_i16x32);
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sq_i32x16)));
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sq_i32x16, 1)));
    }
    nk_size_t remaining_scalars = total_scalars - idx_scalars;
    if (remaining_scalars > 0) {
        __mmask32 tail_mask = stride_mask_m32 & (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)(remaining_scalars));
        __m512i data_i16x32 = _mm512_maskz_loadu_epi16(tail_mask, data_ptr + idx_scalars);
        sum_i32x16 = _mm512_dpwssd_epi32(sum_i32x16, data_i16x32, ones_i16x32);
        __m512i sq_i32x16 = _mm512_dpwssd_epi32(_mm512_setzero_si512(), data_i16x32, data_i16x32);
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sq_i32x16)));
        sumsq_i64x8 = _mm512_add_epi64(sumsq_i64x8, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sq_i32x16, 1)));
    }
    __m512i sum_i64x8 = _mm512_add_epi64(                                 //
        _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sum_i32x16)),        //
        _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(sum_i32x16, 1))); //
    *sum_ptr = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    *sumsq_ptr = (nk_u64_t)nk_reduce_add_i64x8_skylake_(sumsq_i64x8);
}

NK_PUBLIC void nk_reduce_moments_i16_icelake(                          //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_i16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_I16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum, right_sum;
        nk_u64_t left_sumsq, right_sumsq;
        nk_reduce_moments_i16_icelake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i16_icelake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                      &right_sum, &right_sumsq);
        *sum_ptr = nk_i64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i16_icelake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 16)
        nk_reduce_moments_i16_icelake_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_i16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_e2m3_icelake_contiguous_( //
    nk_e2m3_t const *data_ptr, nk_size_t count,              //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    // 64-byte LUT: maps 5-bit unsigned magnitude -> value*16 as u8 (0..120)
    // Entries 0-31 replicated in upper 32 bytes (VPERMB indexes mod 64)
    __m512i const lut_magnitude_u8x64 = _mm512_set_epi8(120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36,
                                                        32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0,
                                                        120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36,
                                                        32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m512i const magnitude_mask_u8x64 = _mm512_set1_epi8(0x1F);
    __m512i const sign_mask_u8x64 = _mm512_set1_epi8(0x20);
    __m512i const ones_u8x64 = _mm512_set1_epi8(1);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_i32x16 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_u8x64 = _mm512_loadu_si512(data_ptr + idx);
        // Extract 5-bit magnitude, LUT lookup
        __m512i magnitude_u8x64 = _mm512_and_si512(data_u8x64, magnitude_mask_u8x64);
        __m512i unsigned_mag_u8x64 = _mm512_permutexvar_epi8(magnitude_u8x64, lut_magnitude_u8x64);
        // Apply sign for sum: negate where bit 5 is set
        __mmask64 sign_mask = _mm512_test_epi8_mask(data_u8x64, sign_mask_u8x64);
        __m512i signed_mag_i8x64 = _mm512_mask_sub_epi8(unsigned_mag_u8x64, sign_mask, _mm512_setzero_si512(),
                                                        unsigned_mag_u8x64);
        // Sum: VPDPBUSD(acc, ones_u8, signed_i8) = acc + sum(1 * signed_val) per 4-byte group
        sum_i32x16 = _mm512_dpbusd_epi32(sum_i32x16, ones_u8x64, signed_mag_i8x64);
        // Sumsq: VPDPBUSD(acc, unsigned_mag, unsigned_mag) = acc + sum(mag^2) per 4-byte group
        // magnitude is 0-120, fits in both u8 and i8 interpretations
        sumsq_i32x16 = _mm512_dpbusd_epi32(sumsq_i32x16, unsigned_mag_u8x64, unsigned_mag_u8x64);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i data_u8x64 = _mm512_maskz_loadu_epi8(tail_mask, data_ptr + idx);
        __m512i magnitude_u8x64 = _mm512_and_si512(data_u8x64, magnitude_mask_u8x64);
        __m512i unsigned_mag_u8x64 = _mm512_permutexvar_epi8(magnitude_u8x64, lut_magnitude_u8x64);
        __mmask64 sign_mask = _mm512_test_epi8_mask(data_u8x64, sign_mask_u8x64);
        __m512i signed_mag_i8x64 = _mm512_mask_sub_epi8(unsigned_mag_u8x64, sign_mask, _mm512_setzero_si512(),
                                                        unsigned_mag_u8x64);
        sum_i32x16 = _mm512_dpbusd_epi32(sum_i32x16, ones_u8x64, signed_mag_i8x64);
        sumsq_i32x16 = _mm512_dpbusd_epi32(sumsq_i32x16, unsigned_mag_u8x64, unsigned_mag_u8x64);
    }
    *sum_ptr = (nk_f32_t)_mm512_reduce_add_epi32(sum_i32x16) / 16.0f;
    *sumsq_ptr = (nk_f32_t)_mm512_reduce_add_epi32(sumsq_i32x16) / 256.0f;
}

NK_INTERNAL void nk_reduce_moments_e2m3_icelake_strided_(                  //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    __mmask64 stride_mask_m64 = nk_stride_mask_u1x64_(stride_elements);
    __m512i const lut_magnitude_u8x64 = _mm512_set_epi8(120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36,
                                                        32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0,
                                                        120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36,
                                                        32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m512i const magnitude_mask_u8x64 = _mm512_set1_epi8(0x1F);
    __m512i const sign_mask_u8x64 = _mm512_set1_epi8(0x20);
    __m512i const ones_u8x64 = _mm512_set1_epi8(1);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = (nk_size_t)_mm_popcnt_u64((nk_u64_t)stride_mask_m64) * stride_elements;
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m512i data_u8x64 = _mm512_maskz_loadu_epi8(stride_mask_m64, data_ptr + idx_scalars);
        __m512i magnitude_u8x64 = _mm512_and_si512(data_u8x64, magnitude_mask_u8x64);
        __m512i unsigned_mag_u8x64 = _mm512_permutexvar_epi8(magnitude_u8x64, lut_magnitude_u8x64);
        __mmask64 sign_mask = _mm512_test_epi8_mask(data_u8x64, sign_mask_u8x64);
        __m512i signed_mag_i8x64 = _mm512_mask_sub_epi8(unsigned_mag_u8x64, sign_mask, _mm512_setzero_si512(),
                                                        unsigned_mag_u8x64);
        sum_i32x16 = _mm512_dpbusd_epi32(sum_i32x16, ones_u8x64, signed_mag_i8x64);
        sumsq_i32x16 = _mm512_dpbusd_epi32(sumsq_i32x16, unsigned_mag_u8x64, unsigned_mag_u8x64);
    }
    nk_size_t remaining_scalars = total_scalars - idx_scalars;
    if (remaining_scalars > 0) {
        __mmask64 tail_mask = stride_mask_m64 & _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining_scalars);
        __m512i data_u8x64 = _mm512_maskz_loadu_epi8(tail_mask, data_ptr + idx_scalars);
        __m512i magnitude_u8x64 = _mm512_and_si512(data_u8x64, magnitude_mask_u8x64);
        __m512i unsigned_mag_u8x64 = _mm512_permutexvar_epi8(magnitude_u8x64, lut_magnitude_u8x64);
        __mmask64 sign_mask = _mm512_test_epi8_mask(data_u8x64, sign_mask_u8x64);
        __m512i signed_mag_i8x64 = _mm512_mask_sub_epi8(unsigned_mag_u8x64, sign_mask, _mm512_setzero_si512(),
                                                        unsigned_mag_u8x64);
        sum_i32x16 = _mm512_dpbusd_epi32(sum_i32x16, ones_u8x64, signed_mag_i8x64);
        sumsq_i32x16 = _mm512_dpbusd_epi32(sumsq_i32x16, unsigned_mag_u8x64, unsigned_mag_u8x64);
    }
    *sum_ptr = (nk_f32_t)_mm512_reduce_add_epi32(sum_i32x16) / 16.0f;
    *sumsq_ptr = (nk_f32_t)_mm512_reduce_add_epi32(sumsq_i32x16) / 256.0f;
}

NK_PUBLIC void nk_reduce_moments_e2m3_icelake(                          //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    int aligned = (stride_bytes % sizeof(nk_e2m3_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e2m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_I16_MAX + 1) * 64) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e2m3_icelake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e2m3_icelake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum;
        *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e2m3_icelake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 16)
        nk_reduce_moments_e2m3_icelake_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e2m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_e3m2_icelake_contiguous_( //
    nk_e3m2_t const *data_ptr, nk_size_t count,              //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    // 32-entry i16 LUT: maps 5-bit unsigned magnitude -> value*16 as i16 (0..448)
    __m512i const lut_magnitude_i16x32 = _mm512_set_epi16(448, 384, 320, 256, 224, 192, 160, 128, 112, 96, 80, 64, 56,
                                                          48, 40, 32, 28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2,
                                                          1, 0);
    __m512i const magnitude_mask_i16x32 = _mm512_set1_epi16(0x1F);
    __m512i const sign_mask_i16x32 = _mm512_set1_epi16(0x20);
    __m512i const ones_i16x32 = _mm512_set1_epi16(1);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_i32x16 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        // Load 32 bytes, widen u8->u16
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data_ptr + idx));
        __m512i data_u16x32 = _mm512_cvtepu8_epi16(data_u8x32);
        // Extract 5-bit magnitude, VPERMW LUT lookup
        __m512i magnitude_u16x32 = _mm512_and_si512(data_u16x32, magnitude_mask_i16x32);
        __m512i unsigned_mag_i16x32 = _mm512_permutexvar_epi16(magnitude_u16x32, lut_magnitude_i16x32);
        // Apply sign for sum: negate where bit 5 is set
        __mmask32 sign_mask = _mm512_test_epi16_mask(data_u16x32, sign_mask_i16x32);
        __m512i signed_mag_i16x32 = _mm512_mask_sub_epi16(unsigned_mag_i16x32, sign_mask, _mm512_setzero_si512(),
                                                          unsigned_mag_i16x32);
        // Sum: VPMADDWD(signed_i16, ones) = sum of pairs -> i32
        sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(signed_mag_i16x32, ones_i16x32));
        // Sumsq: VPMADDWD(unsigned_mag, unsigned_mag) = sum of pairs of squares -> i32
        // max per i32: 2 * 448^2 = 401408, fits in i32
        sumsq_i32x16 = _mm512_add_epi32(sumsq_i32x16, _mm512_madd_epi16(unsigned_mag_i16x32, unsigned_mag_i16x32));
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m256i data_u8x32 = _mm256_maskz_loadu_epi8(tail_mask, data_ptr + idx);
        __m512i data_u16x32 = _mm512_cvtepu8_epi16(data_u8x32);
        __m512i magnitude_u16x32 = _mm512_and_si512(data_u16x32, magnitude_mask_i16x32);
        __m512i unsigned_mag_i16x32 = _mm512_permutexvar_epi16(magnitude_u16x32, lut_magnitude_i16x32);
        __mmask32 sign_mask = _mm512_test_epi16_mask(data_u16x32, sign_mask_i16x32);
        __m512i signed_mag_i16x32 = _mm512_mask_sub_epi16(unsigned_mag_i16x32, sign_mask, _mm512_setzero_si512(),
                                                          unsigned_mag_i16x32);
        sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(signed_mag_i16x32, ones_i16x32));
        sumsq_i32x16 = _mm512_add_epi32(sumsq_i32x16, _mm512_madd_epi16(unsigned_mag_i16x32, unsigned_mag_i16x32));
    }
    *sum_ptr = (nk_f32_t)_mm512_reduce_add_epi32(sum_i32x16) / 16.0f;
    *sumsq_ptr = (nk_f32_t)_mm512_reduce_add_epi32(sumsq_i32x16) / 256.0f;
}

NK_INTERNAL void nk_reduce_moments_e3m2_icelake_strided_(                  //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    __mmask32 stride_mask_m32 = (__mmask32)nk_stride_mask_u1x64_(stride_elements);
    __m512i const lut_magnitude_i16x32 = _mm512_set_epi16(448, 384, 320, 256, 224, 192, 160, 128, 112, 96, 80, 64, 56,
                                                          48, 40, 32, 28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2,
                                                          1, 0);
    __m512i const magnitude_mask_i16x32 = _mm512_set1_epi16(0x1F);
    __m512i const sign_mask_i16x32 = _mm512_set1_epi16(0x20);
    __m512i const ones_i16x32 = _mm512_set1_epi16(1);
    __m512i sum_i32x16 = _mm512_setzero_si512();
    __m512i sumsq_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = (nk_size_t)_mm_popcnt_u64((nk_u64_t)stride_mask_m32) * stride_elements;
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m256i data_u8x32 = _mm256_maskz_loadu_epi8(stride_mask_m32, data_ptr + idx_scalars);
        __m512i data_u16x32 = _mm512_cvtepu8_epi16(data_u8x32);
        __m512i magnitude_u16x32 = _mm512_and_si512(data_u16x32, magnitude_mask_i16x32);
        __m512i unsigned_mag_i16x32 = _mm512_permutexvar_epi16(magnitude_u16x32, lut_magnitude_i16x32);
        __mmask32 sign_mask = _mm512_test_epi16_mask(data_u16x32, sign_mask_i16x32);
        __m512i signed_mag_i16x32 = _mm512_mask_sub_epi16(unsigned_mag_i16x32, sign_mask, _mm512_setzero_si512(),
                                                          unsigned_mag_i16x32);
        sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(signed_mag_i16x32, ones_i16x32));
        sumsq_i32x16 = _mm512_add_epi32(sumsq_i32x16, _mm512_madd_epi16(unsigned_mag_i16x32, unsigned_mag_i16x32));
    }
    nk_size_t remaining_scalars = total_scalars - idx_scalars;
    if (remaining_scalars > 0) {
        __mmask32 tail_mask = stride_mask_m32 & (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining_scalars);
        __m256i data_u8x32 = _mm256_maskz_loadu_epi8(tail_mask, data_ptr + idx_scalars);
        __m512i data_u16x32 = _mm512_cvtepu8_epi16(data_u8x32);
        __m512i magnitude_u16x32 = _mm512_and_si512(data_u16x32, magnitude_mask_i16x32);
        __m512i unsigned_mag_i16x32 = _mm512_permutexvar_epi16(magnitude_u16x32, lut_magnitude_i16x32);
        __mmask32 sign_mask = _mm512_test_epi16_mask(data_u16x32, sign_mask_i16x32);
        __m512i signed_mag_i16x32 = _mm512_mask_sub_epi16(unsigned_mag_i16x32, sign_mask, _mm512_setzero_si512(),
                                                          unsigned_mag_i16x32);
        sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(signed_mag_i16x32, ones_i16x32));
        sumsq_i32x16 = _mm512_add_epi32(sumsq_i32x16, _mm512_madd_epi16(unsigned_mag_i16x32, unsigned_mag_i16x32));
    }
    *sum_ptr = (nk_f32_t)_mm512_reduce_add_epi32(sum_i32x16) / 16.0f;
    *sumsq_ptr = (nk_f32_t)_mm512_reduce_add_epi32(sumsq_i32x16) / 256.0f;
}

NK_PUBLIC void nk_reduce_moments_e3m2_icelake(                          //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    int aligned = (stride_bytes % sizeof(nk_e3m2_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e3m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)2048 * 64) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e3m2_icelake(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e3m2_icelake(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum;
        *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e3m2_icelake_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 16)
        nk_reduce_moments_e3m2_icelake_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e3m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
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
#endif // NK_REDUCE_ICELAKE_H
