/**
 *  @brief Sierra Forest (AVX-VNNI-INT8) implementations for the redesigned reduction API (moments).
 *  @file include/numkong/reduce/sierra.h
 *  @author Ash Vardanian
 *  @date February 13, 2026
 *
 *  @sa include/numkong/reduce.h
 *
 *  Uses AVX-VNNI-INT8 (256-bit) for efficient widening dot-products on i8, u8, and e2m3:
 *  - `_mm256_dpbssd_epi32`: i8 x i8 -> i32 signed dot product (AVXVNNIINT8)
 *  - `_mm256_dpbuud_epi32`: u8 x u8 -> u32 unsigned dot product (AVXVNNIINT8)
 */
#ifndef NK_REDUCE_SIERRA_H
#define NK_REDUCE_SIERRA_H

#if NK_TARGET_X86_
#if NK_TARGET_SIERRA

#include "numkong/types.h"
#include "numkong/reduce/serial.h"
#include "numkong/reduce/haswell.h" // `nk_reduce_add_i32x8_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni,avxvnniint8"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni", "avxvnniint8")
#endif

NK_INTERNAL void nk_reduce_moments_i8_sierra_contiguous_( //
    nk_i8_t const *data, nk_size_t count,                 //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __m256i ones_i8x32 = _mm256_set1_epi8(1);
    __m256i sum_i32x8 = _mm256_setzero_si256();
    __m256i sumsq_i32x8 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        sum_i32x8 = _mm256_dpbssd_epi32(sum_i32x8, data_i8x32, ones_i8x32);
        sumsq_i32x8 = _mm256_dpbssd_epi32(sumsq_i32x8, data_i8x32, data_i8x32);
    }
    nk_i64_t sum = (nk_i64_t)nk_reduce_add_i32x8_haswell_(sum_i32x8);
    nk_u64_t sumsq = (nk_u64_t)(nk_u32_t)nk_reduce_add_i32x8_haswell_(sumsq_i32x8);
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b8x32_serial_(data + idx, &tail_vec, remaining);
        __m256i data_i8x32 = tail_vec.ymm;
        __m256i tail_sum_i32x8 = _mm256_dpbssd_epi32(_mm256_setzero_si256(), data_i8x32, ones_i8x32);
        __m256i tail_sumsq_i32x8 = _mm256_dpbssd_epi32(_mm256_setzero_si256(), data_i8x32, data_i8x32);
        sum += (nk_i64_t)nk_reduce_add_i32x8_haswell_(tail_sum_i32x8);
        sumsq += (nk_u64_t)(nk_u32_t)nk_reduce_add_i32x8_haswell_(tail_sumsq_i32x8);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_i8_sierra_strided_(               //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __m256i stride_mask_i8x32 = nk_stride_blend_u1x32_(stride_elements);
    __m256i ones_i8x32 = _mm256_set1_epi8(1);
    __m256i sum_i32x8 = _mm256_setzero_si256();
    __m256i sumsq_i32x8 = _mm256_setzero_si256();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = nk_size_round_up_to_multiple_(32, stride_elements);
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
        data_i8x32 = _mm256_and_si256(data_i8x32, stride_mask_i8x32);
        sum_i32x8 = _mm256_dpbssd_epi32(sum_i32x8, data_i8x32, ones_i8x32);
        sumsq_i32x8 = _mm256_dpbssd_epi32(sumsq_i32x8, data_i8x32, data_i8x32);
    }
    nk_i64_t sum = (nk_i64_t)nk_reduce_add_i32x8_haswell_(sum_i32x8);
    nk_u64_t sumsq = (nk_u64_t)(nk_u32_t)nk_reduce_add_i32x8_haswell_(sumsq_i32x8);
    nk_i8_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_i64_t val = (nk_i64_t)*ptr;
        sum += val, sumsq += (nk_u64_t)(val * val);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i8_sierra(                       //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_i8_serial(data, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)32768 * 32) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum, right_sum;
        nk_u64_t left_sumsq, right_sumsq;
        nk_reduce_moments_i8_sierra(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i8_sierra(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                    &right_sumsq);
        *sum_ptr = nk_i64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i8_sierra_contiguous_(data, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 8)
        nk_reduce_moments_i8_sierra_strided_(data, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_i8_serial(data, count, stride_bytes, sum_ptr, sumsq_ptr);
}

/**
 *  @section u8 moments via VPDPBUUD (unsigned u8 x u8 -> u32)
 *
 *  Sierra's `_mm256_dpbuud_epi32` provides native u8×u8→u32 dot product, replacing
 *  Haswell's 8-instruction SAD+widen+MADD sequence with 3 instructions per 32 elements.
 *  - sum:   dot(data, ones) via DPBUUD — each group of 4 bytes sums into a u32 lane
 *  - sumsq: dot(data, data) via DPBUUD — native u8×u8 squaring and accumulation
 */
NK_INTERNAL void nk_reduce_moments_u8_sierra_contiguous_( //
    nk_u8_t const *data, nk_size_t count,                 //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __m256i ones_u8x32 = _mm256_set1_epi8(1);
    __m256i sum_i32x8 = _mm256_setzero_si256();
    __m256i sumsq_i32x8 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        sum_i32x8 = _mm256_dpbuud_epi32(sum_i32x8, data_u8x32, ones_u8x32);
        sumsq_i32x8 = _mm256_dpbuud_epi32(sumsq_i32x8, data_u8x32, data_u8x32);
    }
    nk_u64_t sum = (nk_u64_t)(nk_u32_t)nk_reduce_add_i32x8_haswell_(sum_i32x8);
    nk_u64_t sumsq = (nk_u64_t)(nk_u32_t)nk_reduce_add_i32x8_haswell_(sumsq_i32x8);
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b256_vec_t tail_vec;
        nk_partial_load_b8x32_serial_(data + idx, &tail_vec, remaining);
        __m256i data_u8x32 = tail_vec.ymm;
        __m256i tail_sum_i32x8 = _mm256_dpbuud_epi32(_mm256_setzero_si256(), data_u8x32, ones_u8x32);
        __m256i tail_sumsq_i32x8 = _mm256_dpbuud_epi32(_mm256_setzero_si256(), data_u8x32, data_u8x32);
        sum += (nk_u64_t)(nk_u32_t)nk_reduce_add_i32x8_haswell_(tail_sum_i32x8);
        sumsq += (nk_u64_t)(nk_u32_t)nk_reduce_add_i32x8_haswell_(tail_sumsq_i32x8);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_u8_sierra_strided_(               //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    __m256i stride_mask_u8x32 = nk_stride_blend_u1x32_(stride_elements);
    __m256i ones_u8x32 = _mm256_set1_epi8(1);
    __m256i sum_i32x8 = _mm256_setzero_si256();
    __m256i sumsq_i32x8 = _mm256_setzero_si256();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t step = nk_size_round_up_to_multiple_(32, stride_elements);
    for (; idx_scalars + step <= total_scalars; idx_scalars += step) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx_scalars));
        data_u8x32 = _mm256_and_si256(data_u8x32, stride_mask_u8x32);
        sum_i32x8 = _mm256_dpbuud_epi32(sum_i32x8, data_u8x32, ones_u8x32);
        sumsq_i32x8 = _mm256_dpbuud_epi32(sumsq_i32x8, data_u8x32, data_u8x32);
    }
    nk_u64_t sum = (nk_u64_t)(nk_u32_t)nk_reduce_add_i32x8_haswell_(sum_i32x8);
    nk_u64_t sumsq = (nk_u64_t)(nk_u32_t)nk_reduce_add_i32x8_haswell_(sumsq_i32x8);
    nk_u8_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_u64_t val = (nk_u64_t)*ptr;
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u8_sierra(                       //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_u8_serial(data, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)16384 * 32) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u8_sierra(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u8_sierra(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                    &right_sumsq);
        *sum_ptr = nk_u64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u8_sierra_contiguous_(data, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 8)
        nk_reduce_moments_u8_sierra_strided_(data, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_u8_serial(data, count, stride_bytes, sum_ptr, sumsq_ptr);
}

/**
 *  @section e2m3 moments via integer VNNI (dpbssd)
 *
 *  Every e2m3 value × 16 is an exact integer in [-120, +120] (i8 range).
 *  We use a dual-VPSHUFB LUT to map 5-bit magnitude → unsigned i8, apply the sign,
 *  then accumulate with `_mm256_dpbssd_epi32` (signed i8 × signed i8 → i32).
 *  Final: sum = i32_sum / 16, sumsq = i32_sumsq / 256.
 */
NK_INTERNAL void nk_reduce_moments_e2m3_sierra_contiguous_( //
    nk_e2m3_t const *data, nk_size_t count,                 //
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
        __m256i negate_mask_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(data_u8x32, sign_mask_u8x32), sign_mask_u8x32);
        __m256i signed_i8x32 = _mm256_blendv_epi8(
            unsigned_u8x32, _mm256_sub_epi8(_mm256_setzero_si256(), unsigned_u8x32), negate_mask_u8x32);
        sum_i32x8 = _mm256_dpbssd_epi32(sum_i32x8, signed_i8x32, ones_i8x32);
        sumsq_i32x8 = _mm256_dpbssd_epi32(sumsq_i32x8, signed_i8x32, signed_i8x32);
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
        __m256i signed_i8x32 = _mm256_blendv_epi8(
            unsigned_u8x32, _mm256_sub_epi8(_mm256_setzero_si256(), unsigned_u8x32), negate_mask_u8x32);
        sum += nk_reduce_add_i32x8_haswell_(_mm256_dpbssd_epi32(_mm256_setzero_si256(), signed_i8x32, ones_i8x32));
        sumsq += nk_reduce_add_i32x8_haswell_(_mm256_dpbssd_epi32(_mm256_setzero_si256(), signed_i8x32, signed_i8x32));
    }
    *sum_ptr = (nk_f32_t)sum / 16.0f;
    *sumsq_ptr = (nk_f32_t)sumsq / 256.0f;
}

NK_INTERNAL void nk_reduce_moments_e2m3_sierra_strided_(               //
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
        __m256i signed_i8x32 = _mm256_blendv_epi8(
            unsigned_u8x32, _mm256_sub_epi8(_mm256_setzero_si256(), unsigned_u8x32), negate_mask_u8x32);
        sum_i32x8 = _mm256_dpbssd_epi32(sum_i32x8, signed_i8x32, ones_i8x32);
        sumsq_i32x8 = _mm256_dpbssd_epi32(sumsq_i32x8, signed_i8x32, signed_i8x32);
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

NK_PUBLIC void nk_reduce_moments_e2m3_sierra(                       //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    int aligned = (stride_bytes % sizeof(nk_e2m3_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e2m3_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_I16_MAX + 1) * 32) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e2m3_sierra(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e2m3_sierra(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                      &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e2m3_sierra_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 8) nk_reduce_moments_e2m3_sierra_strided_(data, count, stride_elements, sum, sumsq);
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

#endif // NK_TARGET_SIERRA
#endif // NK_TARGET_X86_
#endif // NK_REDUCE_SIERRA_H
