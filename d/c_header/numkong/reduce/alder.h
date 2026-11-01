/**
 *  @brief Alder Lake (AVX2+VNNI) implementations for the redesigned reduction API (moments).
 *  @file include/numkong/reduce/alder.h
 *  @author Ash Vardanian
 *  @date March 4, 2026
 *
 *  @sa include/numkong/reduce.h
 *
 *  Uses AVX-VNNI (256-bit) for efficient widening dot-products:
 *  - `_mm256_dpwssd_epi32`: i16 x i16 -> i32 accumulation (AVXVNNI, used for i16 and e3m2)
 *  - `_mm256_sad_epu8` + `_mm256_madd_epi16`: pure AVX2 SAD/MADD (used for u8)
 *  - `_mm256_cvtepu16_epi32` + `_mm256_mullo_epi32`: pure AVX2 (used for u16)
 */
#ifndef NK_REDUCE_ALDER_H
#define NK_REDUCE_ALDER_H

#if NK_TARGET_X86_
#if NK_TARGET_ALDER

#include "numkong/types.h"
#include "numkong/dot/alder.h" // VEX compat macros (_mm256_dpwssd_avx_epi32)
#include "numkong/reduce/serial.h"
#include "numkong/reduce/haswell.h" // `nk_reduce_add_i32x8_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni")
#endif

NK_INTERNAL void nk_reduce_moments_u8_alder_contiguous_( //
    nk_u8_t const *data, nk_size_t count,                //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __m256i zero_u8x32 = _mm256_setzero_si256();
    __m256i sum_u64x4 = _mm256_setzero_si256();
    __m256i sumsq_low_i32x8 = _mm256_setzero_si256();
    __m256i sumsq_high_i32x8 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_sad_epu8(data_u8x32, zero_u8x32));
        __m256i low_u16x16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(data_u8x32));
        __m256i high_u16x16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data_u8x32, 1));
        sumsq_low_i32x8 = _mm256_add_epi32(sumsq_low_i32x8, _mm256_madd_epi16(low_u16x16, low_u16x16));
        sumsq_high_i32x8 = _mm256_add_epi32(sumsq_high_i32x8, _mm256_madd_epi16(high_u16x16, high_u16x16));
    }

    // Handle tail with partial load
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b8x32_serial_(data + idx, &tail_vec, remaining);
        __m256i data_u8x32 = tail_vec.ymm;
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_sad_epu8(data_u8x32, zero_u8x32));
        __m256i low_u16x16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(data_u8x32));
        __m256i high_u16x16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data_u8x32, 1));
        sumsq_low_i32x8 = _mm256_add_epi32(sumsq_low_i32x8, _mm256_madd_epi16(low_u16x16, low_u16x16));
        sumsq_high_i32x8 = _mm256_add_epi32(sumsq_high_i32x8, _mm256_madd_epi16(high_u16x16, high_u16x16));
    }

    sumsq_low_i32x8 = _mm256_add_epi32(sumsq_low_i32x8, sumsq_high_i32x8);
    __m256i sumsq_u64x4 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sumsq_low_i32x8));
    sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sumsq_low_i32x8, 1)));
    *sum_ptr = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    *sumsq_ptr = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sumsq_u64x4);
}

NK_INTERNAL void nk_reduce_moments_u8_alder_strided_(                //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __m256i stride_mask_u8x32 = nk_stride_blend_u1x32_(stride_elements);
    __m256i zero_u8x32 = _mm256_setzero_si256();
    __m256i sum_u64x4 = _mm256_setzero_si256();
    __m256i sumsq_low_i32x8 = _mm256_setzero_si256();
    __m256i sumsq_high_i32x8 = _mm256_setzero_si256();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = nk_size_round_up_to_multiple_(32, stride_elements);
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
        data_u8x32 = _mm256_and_si256(data_u8x32, stride_mask_u8x32);
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_sad_epu8(data_u8x32, zero_u8x32));
        __m256i low_u16x16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(data_u8x32));
        __m256i high_u16x16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data_u8x32, 1));
        sumsq_low_i32x8 = _mm256_add_epi32(sumsq_low_i32x8, _mm256_madd_epi16(low_u16x16, low_u16x16));
        sumsq_high_i32x8 = _mm256_add_epi32(sumsq_high_i32x8, _mm256_madd_epi16(high_u16x16, high_u16x16));
    }

    sumsq_low_i32x8 = _mm256_add_epi32(sumsq_low_i32x8, sumsq_high_i32x8);
    __m256i sumsq_u64x4 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sumsq_low_i32x8));
    sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sumsq_low_i32x8, 1)));
    nk_u64_t sum = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    nk_u64_t sumsq = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sumsq_u64x4);

    // Scalar tail for remaining elements
    nk_u8_t const *ptr = data + idx_scalars;
    nk_size_t remaining_elements = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining_elements; ++i, ptr += stride_elements) {
        nk_u64_t val = (nk_u64_t)*ptr;
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u8_alder(                        //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_u8_serial(data, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)16384 * 32) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u8_alder(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u8_alder(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                   &right_sumsq);
        *sum_ptr = nk_u64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u8_alder_contiguous_(data, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 8)
        nk_reduce_moments_u8_alder_strided_(data, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_u8_serial(data, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_i16_alder_contiguous_( //
    nk_i16_t const *data, nk_size_t count,                //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __m256i ones_i16x16 = _mm256_set1_epi16(1);
    __m256i sum_i64x4 = _mm256_setzero_si256();
    __m256i sumsq_i64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_i16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m256i sum_i32x8 = _mm256_dpwssd_avx_epi32(_mm256_setzero_si256(), data_i16x16, ones_i16x16);
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum_i32x8)));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum_i32x8, 1)));
        __m256i sq_i32x8 = _mm256_dpwssd_avx_epi32(_mm256_setzero_si256(), data_i16x16, data_i16x16);
        sumsq_i64x4 = _mm256_add_epi64(sumsq_i64x4, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sq_i32x8)));
        sumsq_i64x4 = _mm256_add_epi64(sumsq_i64x4, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sq_i32x8, 1)));
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b16x16_serial_(data + idx, &tail_vec, remaining);
        __m256i data_i16x16 = tail_vec.ymm;
        __m256i sum_i32x8 = _mm256_dpwssd_avx_epi32(_mm256_setzero_si256(), data_i16x16, ones_i16x16);
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum_i32x8)));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum_i32x8, 1)));
        __m256i sq_i32x8 = _mm256_dpwssd_avx_epi32(_mm256_setzero_si256(), data_i16x16, data_i16x16);
        sumsq_i64x4 = _mm256_add_epi64(sumsq_i64x4, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sq_i32x8)));
        sumsq_i64x4 = _mm256_add_epi64(sumsq_i64x4, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sq_i32x8, 1)));
    }
    *sum_ptr = (nk_i64_t)nk_reduce_add_i64x4_haswell_(sum_i64x4);
    *sumsq_ptr = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sumsq_i64x4);
}

NK_INTERNAL void nk_reduce_moments_i16_alder_strided_(                //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __m256i stride_mask_i16x16 = nk_stride_blend_b16x16_(stride_elements);
    __m256i ones_i16x16 = _mm256_set1_epi16(1);
    __m256i sum_i64x4 = _mm256_setzero_si256();
    __m256i sumsq_i64x4 = _mm256_setzero_si256();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = nk_size_round_up_to_multiple_(16, stride_elements);
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m256i data_i16x16 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
        data_i16x16 = _mm256_and_si256(data_i16x16, stride_mask_i16x16);
        __m256i sum_i32x8 = _mm256_dpwssd_avx_epi32(_mm256_setzero_si256(), data_i16x16, ones_i16x16);
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum_i32x8)));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum_i32x8, 1)));
        __m256i sq_i32x8 = _mm256_dpwssd_avx_epi32(_mm256_setzero_si256(), data_i16x16, data_i16x16);
        sumsq_i64x4 = _mm256_add_epi64(sumsq_i64x4, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sq_i32x8)));
        sumsq_i64x4 = _mm256_add_epi64(sumsq_i64x4, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sq_i32x8, 1)));
    }
    nk_i64_t sum = (nk_i64_t)nk_reduce_add_i64x4_haswell_(sum_i64x4);
    nk_u64_t sumsq = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sumsq_i64x4);
    nk_i16_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_i64_t val = (nk_i64_t)*ptr;
        sum += val, sumsq += (nk_u64_t)(val * val);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i16_alder(                        //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_i16_serial(data, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum, right_sum;
        nk_u64_t left_sumsq, right_sumsq;
        nk_reduce_moments_i16_alder(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i16_alder(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                    &right_sumsq);
        *sum_ptr = nk_i64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i16_alder_contiguous_(data, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 8)
        nk_reduce_moments_i16_alder_strided_(data, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_i16_serial(data, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_u16_alder_contiguous_( //
    nk_u16_t const *data, nk_size_t count,                //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __m256i sum_u32x8 = _mm256_setzero_si256();
    __m256i sumsq_u64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_u32x8 = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i const *)(data + idx)));
        sum_u32x8 = _mm256_add_epi32(sum_u32x8, data_u32x8);
        __m256i sq_u32x8 = _mm256_mullo_epi32(data_u32x8, data_u32x8);
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sq_u32x8)));
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sq_u32x8, 1)));
    }

    // Handle tail with partial load
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b16x16_serial_(data + idx, &tail_vec, remaining);
        __m256i data_u32x8 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(tail_vec.ymm));
        sum_u32x8 = _mm256_add_epi32(sum_u32x8, data_u32x8);
        __m256i sq_u32x8 = _mm256_mullo_epi32(data_u32x8, data_u32x8);
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sq_u32x8)));
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sq_u32x8, 1)));
    }

    __m256i sum_u64x4 = _mm256_add_epi64(                               //
        _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sum_u32x8)),       //
        _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sum_u32x8, 1))); //
    *sum_ptr = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    *sumsq_ptr = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sumsq_u64x4);
}

NK_INTERNAL void nk_reduce_moments_u16_alder_strided_(                //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __m256i stride_mask_i16x16 = nk_stride_blend_b16x16_(stride_elements);
    __m256i sum_u32x8 = _mm256_setzero_si256();
    __m256i sumsq_u64x4 = _mm256_setzero_si256();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = nk_size_round_up_to_multiple_(16, stride_elements);
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m256i data_u16x16 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
        data_u16x16 = _mm256_and_si256(data_u16x16, stride_mask_i16x16);
        __m256i low_u32x8 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(data_u16x16));
        __m256i high_u32x8 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(data_u16x16, 1));
        sum_u32x8 = _mm256_add_epi32(sum_u32x8, low_u32x8);
        sum_u32x8 = _mm256_add_epi32(sum_u32x8, high_u32x8);
        __m256i low_sq_u32x8 = _mm256_mullo_epi32(low_u32x8, low_u32x8);
        __m256i high_sq_u32x8 = _mm256_mullo_epi32(high_u32x8, high_u32x8);
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(low_sq_u32x8)));
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(low_sq_u32x8, 1)));
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(high_sq_u32x8)));
        sumsq_u64x4 = _mm256_add_epi64(sumsq_u64x4, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(high_sq_u32x8, 1)));
    }

    __m256i sum_u64x4 = _mm256_add_epi64(                               //
        _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sum_u32x8)),       //
        _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sum_u32x8, 1))); //
    nk_u64_t sum = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    nk_u64_t sumsq = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sumsq_u64x4);

    // Scalar tail for remaining elements
    nk_u16_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_u64_t val = (nk_u64_t)*ptr;
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u16_alder(                        //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_u16_serial(data, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u16_alder(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u16_alder(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                    &right_sumsq);
        *sum_ptr = nk_u64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u16_alder_contiguous_(data, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 8)
        nk_reduce_moments_u16_alder_strided_(data, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_u16_serial(data, count, stride_bytes, sum_ptr, sumsq_ptr);
}

/**
 *  @section e3m2 moments via integer VNNI (dpwssd)
 *
 *  Every e3m2 value x 16 is an exact integer in [-448, +448] (i16 range).
 *  We use dual-VPSHUFB for the low byte + threshold compare for the high byte,
 *  then UNPACKLO/HI to form unsigned i16, apply sign via `_mm256_sign_epi16`,
 *  and accumulate with `_mm256_dpwssd_epi32` (signed i16 x signed i16 -> i32).
 *  Final: sum = i32_sum / 16, sumsq = i32_sumsq / 256.
 */
NK_INTERNAL void nk_reduce_moments_e3m2_alder_contiguous_( //
    nk_e3m2_t const *data, nk_size_t count,                //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    __m256i const lut_lo_lower_u8x32 = _mm256_set_epi8(                                                            //
        28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0,                                                     //
        28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0);                                                    //
    __m256i const lut_lo_upper_u8x32 = _mm256_set_epi8(                                                            //
        (char)192, (char)128, 64, 0, (char)224, (char)192, (char)160, (char)128, 112, 96, 80, 64, 56, 48, 40, 32,  //
        (char)192, (char)128, 64, 0, (char)224, (char)192, (char)160, (char)128, 112, 96, 80, 64, 56, 48, 40, 32); //
    __m256i const nibble_mask_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const magnitude_mask_u8x32 = _mm256_set1_epi8(0x1F);
    __m256i const half_select_u8x32 = _mm256_set1_epi8(0x10);
    __m256i const sign_mask_u8x32 = _mm256_set1_epi8(0x20);
    __m256i const high_threshold_u8x32 = _mm256_set1_epi8(27);
    __m256i const ones_u8x32 = _mm256_set1_epi8(1);
    __m256i const ones_i16x16 = _mm256_set1_epi16(1);
    __m256i sum_i32x8 = _mm256_setzero_si256();
    __m256i sumsq_i32x8 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m256i magnitude_u8x32 = _mm256_and_si256(data_u8x32, magnitude_mask_u8x32);
        __m256i shuffle_idx_u8x32 = _mm256_and_si256(magnitude_u8x32, nibble_mask_u8x32);
        __m256i upper_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(magnitude_u8x32, half_select_u8x32),
                                                       half_select_u8x32);
        __m256i lo_bytes_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lo_lower_u8x32, shuffle_idx_u8x32),
                                                    _mm256_shuffle_epi8(lut_lo_upper_u8x32, shuffle_idx_u8x32),
                                                    upper_select_u8x32);
        __m256i hi_bytes_u8x32 = _mm256_and_si256(_mm256_cmpgt_epi8(magnitude_u8x32, high_threshold_u8x32), ones_u8x32);
        __m256i unsigned_lo_i16x16 = _mm256_unpacklo_epi8(lo_bytes_u8x32, hi_bytes_u8x32);
        __m256i unsigned_hi_i16x16 = _mm256_unpackhi_epi8(lo_bytes_u8x32, hi_bytes_u8x32);
        // Sign handling: extract sign bit, widen to i16, create +1/-1, apply via VPSIGNW
        __m256i negate_mask_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(data_u8x32, sign_mask_u8x32), sign_mask_u8x32);
        __m256i negate_lo_i16x16 = _mm256_unpacklo_epi8(negate_mask_u8x32, negate_mask_u8x32);
        __m256i negate_hi_i16x16 = _mm256_unpackhi_epi8(negate_mask_u8x32, negate_mask_u8x32);
        __m256i signed_lo_i16x16 = _mm256_sign_epi16(unsigned_lo_i16x16,
                                                     _mm256_or_si256(negate_lo_i16x16, ones_i16x16));
        __m256i signed_hi_i16x16 = _mm256_sign_epi16(unsigned_hi_i16x16,
                                                     _mm256_or_si256(negate_hi_i16x16, ones_i16x16));
        // VNNI accumulation: dpwssd (signed i16 x signed i16 -> i32)
        sum_i32x8 = _mm256_dpwssd_avx_epi32(sum_i32x8, signed_lo_i16x16, ones_i16x16);
        sum_i32x8 = _mm256_dpwssd_avx_epi32(sum_i32x8, signed_hi_i16x16, ones_i16x16);
        sumsq_i32x8 = _mm256_dpwssd_avx_epi32(sumsq_i32x8, signed_lo_i16x16, signed_lo_i16x16);
        sumsq_i32x8 = _mm256_dpwssd_avx_epi32(sumsq_i32x8, signed_hi_i16x16, signed_hi_i16x16);
    }
    nk_i32_t sum = nk_reduce_add_i32x8_haswell_(sum_i32x8);
    nk_i32_t sumsq = nk_reduce_add_i32x8_haswell_(sumsq_i32x8);
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b8x32_serial_(data + idx, &tail_vec, remaining);
        __m256i data_u8x32 = tail_vec.ymm;
        __m256i magnitude_u8x32 = _mm256_and_si256(data_u8x32, magnitude_mask_u8x32);
        __m256i shuffle_idx_u8x32 = _mm256_and_si256(magnitude_u8x32, nibble_mask_u8x32);
        __m256i upper_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(magnitude_u8x32, half_select_u8x32),
                                                       half_select_u8x32);
        __m256i lo_bytes_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lo_lower_u8x32, shuffle_idx_u8x32),
                                                    _mm256_shuffle_epi8(lut_lo_upper_u8x32, shuffle_idx_u8x32),
                                                    upper_select_u8x32);
        __m256i hi_bytes_u8x32 = _mm256_and_si256(_mm256_cmpgt_epi8(magnitude_u8x32, high_threshold_u8x32), ones_u8x32);
        __m256i unsigned_lo_i16x16 = _mm256_unpacklo_epi8(lo_bytes_u8x32, hi_bytes_u8x32);
        __m256i unsigned_hi_i16x16 = _mm256_unpackhi_epi8(lo_bytes_u8x32, hi_bytes_u8x32);
        __m256i negate_mask_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(data_u8x32, sign_mask_u8x32), sign_mask_u8x32);
        __m256i negate_lo_i16x16 = _mm256_unpacklo_epi8(negate_mask_u8x32, negate_mask_u8x32);
        __m256i negate_hi_i16x16 = _mm256_unpackhi_epi8(negate_mask_u8x32, negate_mask_u8x32);
        __m256i signed_lo_i16x16 = _mm256_sign_epi16(unsigned_lo_i16x16,
                                                     _mm256_or_si256(negate_lo_i16x16, ones_i16x16));
        __m256i signed_hi_i16x16 = _mm256_sign_epi16(unsigned_hi_i16x16,
                                                     _mm256_or_si256(negate_hi_i16x16, ones_i16x16));
        __m256i tail_sum = _mm256_dpwssd_avx_epi32(_mm256_setzero_si256(), signed_lo_i16x16, ones_i16x16);
        tail_sum = _mm256_dpwssd_avx_epi32(tail_sum, signed_hi_i16x16, ones_i16x16);
        __m256i tail_sumsq = _mm256_dpwssd_avx_epi32(_mm256_setzero_si256(), signed_lo_i16x16, signed_lo_i16x16);
        tail_sumsq = _mm256_dpwssd_avx_epi32(tail_sumsq, signed_hi_i16x16, signed_hi_i16x16);
        sum += nk_reduce_add_i32x8_haswell_(tail_sum);
        sumsq += nk_reduce_add_i32x8_haswell_(tail_sumsq);
    }
    *sum_ptr = (nk_f32_t)sum / 16.0f;
    *sumsq_ptr = (nk_f32_t)sumsq / 256.0f;
}

NK_INTERNAL void nk_reduce_moments_e3m2_alder_strided_(                //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    __m256i stride_mask_u8x32 = nk_stride_blend_u1x32_(stride_elements);
    __m256i const lut_lo_lower_u8x32 = _mm256_set_epi8(                                                            //
        28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0,                                                     //
        28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0);                                                    //
    __m256i const lut_lo_upper_u8x32 = _mm256_set_epi8(                                                            //
        (char)192, (char)128, 64, 0, (char)224, (char)192, (char)160, (char)128, 112, 96, 80, 64, 56, 48, 40, 32,  //
        (char)192, (char)128, 64, 0, (char)224, (char)192, (char)160, (char)128, 112, 96, 80, 64, 56, 48, 40, 32); //
    __m256i const nibble_mask_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const magnitude_mask_u8x32 = _mm256_set1_epi8(0x1F);
    __m256i const half_select_u8x32 = _mm256_set1_epi8(0x10);
    __m256i const sign_mask_u8x32 = _mm256_set1_epi8(0x20);
    __m256i const high_threshold_u8x32 = _mm256_set1_epi8(27);
    __m256i const ones_u8x32 = _mm256_set1_epi8(1);
    __m256i const ones_i16x16 = _mm256_set1_epi16(1);
    __m256i sum_i32x8 = _mm256_setzero_si256();
    __m256i sumsq_i32x8 = _mm256_setzero_si256();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = nk_size_round_up_to_multiple_(32, stride_elements);
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
        data_u8x32 = _mm256_and_si256(data_u8x32, stride_mask_u8x32);
        __m256i magnitude_u8x32 = _mm256_and_si256(data_u8x32, magnitude_mask_u8x32);
        __m256i shuffle_idx_u8x32 = _mm256_and_si256(magnitude_u8x32, nibble_mask_u8x32);
        __m256i upper_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(magnitude_u8x32, half_select_u8x32),
                                                       half_select_u8x32);
        __m256i lo_bytes_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lo_lower_u8x32, shuffle_idx_u8x32),
                                                    _mm256_shuffle_epi8(lut_lo_upper_u8x32, shuffle_idx_u8x32),
                                                    upper_select_u8x32);
        __m256i hi_bytes_u8x32 = _mm256_and_si256(_mm256_cmpgt_epi8(magnitude_u8x32, high_threshold_u8x32), ones_u8x32);
        __m256i unsigned_lo_i16x16 = _mm256_unpacklo_epi8(lo_bytes_u8x32, hi_bytes_u8x32);
        __m256i unsigned_hi_i16x16 = _mm256_unpackhi_epi8(lo_bytes_u8x32, hi_bytes_u8x32);
        __m256i negate_mask_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(data_u8x32, sign_mask_u8x32), sign_mask_u8x32);
        __m256i negate_lo_i16x16 = _mm256_unpacklo_epi8(negate_mask_u8x32, negate_mask_u8x32);
        __m256i negate_hi_i16x16 = _mm256_unpackhi_epi8(negate_mask_u8x32, negate_mask_u8x32);
        __m256i signed_lo_i16x16 = _mm256_sign_epi16(unsigned_lo_i16x16,
                                                     _mm256_or_si256(negate_lo_i16x16, ones_i16x16));
        __m256i signed_hi_i16x16 = _mm256_sign_epi16(unsigned_hi_i16x16,
                                                     _mm256_or_si256(negate_hi_i16x16, ones_i16x16));
        sum_i32x8 = _mm256_dpwssd_avx_epi32(sum_i32x8, signed_lo_i16x16, ones_i16x16);
        sum_i32x8 = _mm256_dpwssd_avx_epi32(sum_i32x8, signed_hi_i16x16, ones_i16x16);
        sumsq_i32x8 = _mm256_dpwssd_avx_epi32(sumsq_i32x8, signed_lo_i16x16, signed_lo_i16x16);
        sumsq_i32x8 = _mm256_dpwssd_avx_epi32(sumsq_i32x8, signed_hi_i16x16, signed_hi_i16x16);
    }
    nk_i32_t sum = nk_reduce_add_i32x8_haswell_(sum_i32x8);
    nk_i32_t sumsq = nk_reduce_add_i32x8_haswell_(sumsq_i32x8);
    nk_e3m2_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_f32_t val;
        nk_e3m2_to_f32_serial(ptr, &val);
        nk_i32_t ival = (nk_i32_t)(val * 16.0f);
        sum += ival;
        sumsq += ival * ival;
    }
    *sum_ptr = (nk_f32_t)sum / 16.0f;
    *sumsq_ptr = (nk_f32_t)sumsq / 256.0f;
}

NK_PUBLIC void nk_reduce_moments_e3m2_alder(                        //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    int aligned = (stride_bytes % sizeof(nk_e3m2_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e3m2_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)2048 * 32) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e3m2_alder(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e3m2_alder(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                     &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e3m2_alder_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 8) nk_reduce_moments_e3m2_alder_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_e3m2_serial(data, count, stride_bytes, sum, sumsq);
}

/**
 *  @section e2m3 moments via integer VNNI (dpbusd)
 *
 *  Every e2m3 value × 16 is an exact integer in [-120, +120] (i8 range).
 *  We use a dual-VPSHUFB LUT to map 5-bit magnitude → unsigned i8, create a
 *  sign vector (+1/-1), then accumulate with `_mm256_dpbusd_avx_epi32` (u8 × i8 → i32).
 *  For sumsq, magnitude ≤ 120 < 128 so it's safe as both u8 and i8.
 *  Final: sum = i32_sum / 16, sumsq = i32_sumsq / 256.
 */
NK_INTERNAL void nk_reduce_moments_e2m3_alder_contiguous_( //
    nk_e2m3_t const *data, nk_size_t count,                //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    __m256i const lut_lower_u8x32 = _mm256_set_epi8(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, //
                                                    30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m256i const lut_upper_u8x32 = _mm256_set_epi8(120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32,
                                                    120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32);
    __m256i const nibble_mask_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const magnitude_mask_u8x32 = _mm256_set1_epi8(0x1F);
    __m256i const half_select_u8x32 = _mm256_set1_epi8(0x10);
    __m256i const sign_mask_u8x32 = _mm256_set1_epi8(0x20);
    __m256i const ones_i8x32 = _mm256_set1_epi8(1);
    __m256i const neg_ones_i8x32 = _mm256_set1_epi8(-1);
    __m256i sum_i32x8 = _mm256_setzero_si256();
    __m256i sumsq_i32x8 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m256i magnitude_u8x32 = _mm256_and_si256(data_u8x32, magnitude_mask_u8x32);
        __m256i shuffle_idx_u8x32 = _mm256_and_si256(magnitude_u8x32, nibble_mask_u8x32);
        __m256i upper_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(magnitude_u8x32, half_select_u8x32),
                                                       half_select_u8x32);
        __m256i unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, shuffle_idx_u8x32),
                                                    _mm256_shuffle_epi8(lut_upper_u8x32, shuffle_idx_u8x32),
                                                    upper_select_u8x32);
        // Sign vector: +1 for positive, -1 for negative (i8)
        __m256i negate_mask_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(data_u8x32, sign_mask_u8x32), sign_mask_u8x32);
        __m256i sign_i8x32 = _mm256_blendv_epi8(ones_i8x32, neg_ones_i8x32, negate_mask_u8x32);
        // DPBUSD: unsigned_magnitude (u8) × sign (+1/-1 i8) → signed sum
        sum_i32x8 = _mm256_dpbusd_avx_epi32(sum_i32x8, unsigned_u8x32, sign_i8x32);
        // DPBUSD: unsigned_magnitude (u8) × unsigned_magnitude (as i8, safe since ≤120) → sumsq
        sumsq_i32x8 = _mm256_dpbusd_avx_epi32(sumsq_i32x8, unsigned_u8x32, unsigned_u8x32);
    }
    nk_i32_t sum = nk_reduce_add_i32x8_haswell_(sum_i32x8);
    nk_i32_t sumsq = nk_reduce_add_i32x8_haswell_(sumsq_i32x8);
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b8x32_serial_(data + idx, &tail_vec, remaining);
        __m256i data_u8x32 = tail_vec.ymm;
        __m256i magnitude_u8x32 = _mm256_and_si256(data_u8x32, magnitude_mask_u8x32);
        __m256i shuffle_idx_u8x32 = _mm256_and_si256(magnitude_u8x32, nibble_mask_u8x32);
        __m256i upper_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(magnitude_u8x32, half_select_u8x32),
                                                       half_select_u8x32);
        __m256i unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, shuffle_idx_u8x32),
                                                    _mm256_shuffle_epi8(lut_upper_u8x32, shuffle_idx_u8x32),
                                                    upper_select_u8x32);
        __m256i negate_mask_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(data_u8x32, sign_mask_u8x32), sign_mask_u8x32);
        __m256i sign_i8x32 = _mm256_blendv_epi8(ones_i8x32, neg_ones_i8x32, negate_mask_u8x32);
        sum += nk_reduce_add_i32x8_haswell_(
            _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), unsigned_u8x32, sign_i8x32));
        sumsq += nk_reduce_add_i32x8_haswell_(
            _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), unsigned_u8x32, unsigned_u8x32));
    }
    *sum_ptr = (nk_f32_t)sum / 16.0f;
    *sumsq_ptr = (nk_f32_t)sumsq / 256.0f;
}

NK_INTERNAL void nk_reduce_moments_e2m3_alder_strided_(                //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    __m256i stride_mask_u8x32 = nk_stride_blend_u1x32_(stride_elements);
    __m256i const lut_lower_u8x32 = _mm256_set_epi8(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, //
                                                    30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m256i const lut_upper_u8x32 = _mm256_set_epi8(120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32,
                                                    120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32);
    __m256i const nibble_mask_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const magnitude_mask_u8x32 = _mm256_set1_epi8(0x1F);
    __m256i const half_select_u8x32 = _mm256_set1_epi8(0x10);
    __m256i const sign_mask_u8x32 = _mm256_set1_epi8(0x20);
    __m256i const ones_i8x32 = _mm256_set1_epi8(1);
    __m256i const neg_ones_i8x32 = _mm256_set1_epi8(-1);
    __m256i sum_i32x8 = _mm256_setzero_si256();
    __m256i sumsq_i32x8 = _mm256_setzero_si256();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = nk_size_round_up_to_multiple_(32, stride_elements);
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
        data_u8x32 = _mm256_and_si256(data_u8x32, stride_mask_u8x32);
        __m256i magnitude_u8x32 = _mm256_and_si256(data_u8x32, magnitude_mask_u8x32);
        __m256i shuffle_idx_u8x32 = _mm256_and_si256(magnitude_u8x32, nibble_mask_u8x32);
        __m256i upper_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(magnitude_u8x32, half_select_u8x32),
                                                       half_select_u8x32);
        __m256i unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, shuffle_idx_u8x32),
                                                    _mm256_shuffle_epi8(lut_upper_u8x32, shuffle_idx_u8x32),
                                                    upper_select_u8x32);
        __m256i negate_mask_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(data_u8x32, sign_mask_u8x32), sign_mask_u8x32);
        __m256i sign_i8x32 = _mm256_blendv_epi8(ones_i8x32, neg_ones_i8x32, negate_mask_u8x32);
        sum_i32x8 = _mm256_dpbusd_avx_epi32(sum_i32x8, unsigned_u8x32, sign_i8x32);
        sumsq_i32x8 = _mm256_dpbusd_avx_epi32(sumsq_i32x8, unsigned_u8x32, unsigned_u8x32);
    }
    nk_i32_t sum = nk_reduce_add_i32x8_haswell_(sum_i32x8);
    nk_i32_t sumsq = nk_reduce_add_i32x8_haswell_(sumsq_i32x8);
    nk_e2m3_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_f32_t val;
        nk_e2m3_to_f32_serial(ptr, &val);
        nk_i32_t ival = (nk_i32_t)(val * 16.0f);
        sum += ival;
        sumsq += ival * ival;
    }
    *sum_ptr = (nk_f32_t)sum / 16.0f;
    *sumsq_ptr = (nk_f32_t)sumsq / 256.0f;
}

NK_PUBLIC void nk_reduce_moments_e2m3_alder(                        //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    int aligned = (stride_bytes % sizeof(nk_e2m3_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e2m3_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_I16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e2m3_alder(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e2m3_alder(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                     &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e2m3_alder_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 8) nk_reduce_moments_e2m3_alder_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_e2m3_serial(data, count, stride_bytes, sum, sumsq);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_ALDER
#endif // NK_TARGET_X86_
#endif // NK_REDUCE_ALDER_H
