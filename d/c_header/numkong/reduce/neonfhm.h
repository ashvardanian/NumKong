/**
 *  @brief ARMv8.4-FHM implementations for the redesigned reduction API (moments + minmax).
 *  @file include/numkong/reduce/neonfhm.h
 *  @author Ash Vardanian
 *  @date February 13, 2026
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_NEONFHM_H
#define NK_REDUCE_NEONFHM_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONFHM

#include "numkong/types.h"         // `nk_e4m3_t`
#include "numkong/cast/serial.h"   // `nk_f16_to_f32_serial`
#include "numkong/cast/neon.h"     // `nk_e4m3x8_to_f16x8_neon_`
#include "numkong/reduce/serial.h" // `nk_reduce_moments_e4m3_serial`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

NK_INTERNAL void nk_reduce_moments_e4m3_neonfhm_contiguous_( //
    nk_e4m3_t const *data_ptr, nk_size_t count,              //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(0x3C00));
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        uint8x8_t data_u8x8 = vld1_u8((uint8_t const *)(data_ptr + idx));
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }

    // Tail: partial load for remaining elements (< 8)
    if (idx < count) {
        nk_b64_vec_t tail_vec;
        nk_partial_load_b8x8_serial_(data_ptr + idx, &tail_vec, count - idx);
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(tail_vec.u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }

    *sum_ptr = vaddvq_f32(sum_f32x4);
    *sumsq_ptr = vaddvq_f32(sumsq_f32x4);
}

NK_INTERNAL void nk_reduce_moments_e4m3_neonfhm_strided_(                  //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(0x3C00));
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint8x8x2_t loaded = vld2_u8((nk_u8_t const *)(data_ptr + idx * 2));
            float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(loaded.val[0]);
            sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
            sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint8x8x3_t loaded = vld3_u8((nk_u8_t const *)(data_ptr + idx * 3));
            float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(loaded.val[0]);
            sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
            sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 <= count; idx += 8) {
            uint8x8x4_t loaded = vld4_u8((nk_u8_t const *)(data_ptr + idx * 4));
            float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(loaded.val[0]);
            sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
            sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        }
    }
    else {
        nk_e4m3_t const *ptr = data_ptr;
        for (; idx + 8 <= count; idx += 8) {
            nk_b64_vec_t data_vec = {0};
            for (nk_size_t i = 0; i < 8; ++i) {
                data_vec.u8s[i] = *ptr;
                ptr += stride_elements;
            }
            float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_vec.u8x8);
            sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
            sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        }
    }

    if (idx < count) {
        nk_b64_vec_t data_vec = {0};
        nk_e4m3_t const *ptr = data_ptr + idx * stride_elements;
        for (nk_size_t i = 0; idx + i < count; ++i) {
            data_vec.u8s[i] = *ptr;
            ptr += stride_elements;
        }
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_vec.u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }

    *sum_ptr = vaddvq_f32(sum_f32x4);
    *sumsq_ptr = vaddvq_f32(sumsq_f32x4);
}

NK_PUBLIC void nk_reduce_moments_e4m3_neonfhm(                          //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e4m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)5000 * 8) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum_value, left_sumsq_value, right_sum_value, right_sumsq_value;
        nk_reduce_moments_e4m3_neonfhm(data_ptr, left_count, stride_bytes, &left_sum_value, &left_sumsq_value);
        nk_reduce_moments_e4m3_neonfhm(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum_value, &right_sumsq_value);
        *sum_ptr = left_sum_value + right_sum_value, *sumsq_ptr = left_sumsq_value + right_sumsq_value;
    }
    else if (stride_elements == 1) nk_reduce_moments_e4m3_neonfhm_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e4m3_neonfhm_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_e5m2_neonfhm_contiguous_( //
    nk_e5m2_t const *data_ptr, nk_size_t count,              //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(0x3C00));
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        uint8x8_t data_u8x8 = vld1_u8((uint8_t const *)(data_ptr + idx));
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }

    // Tail: partial load for remaining elements (< 8)
    if (idx < count) {
        nk_b64_vec_t tail_vec;
        nk_partial_load_b8x8_serial_(data_ptr + idx, &tail_vec, count - idx);
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(tail_vec.u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }

    *sum_ptr = vaddvq_f32(sum_f32x4);
    *sumsq_ptr = vaddvq_f32(sumsq_f32x4);
}

NK_INTERNAL void nk_reduce_moments_e5m2_neonfhm_strided_(                  //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(0x3C00));
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint8x8x2_t loaded = vld2_u8((nk_u8_t const *)(data_ptr + idx * 2));
            float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(loaded.val[0]);
            sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
            sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint8x8x3_t loaded = vld3_u8((nk_u8_t const *)(data_ptr + idx * 3));
            float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(loaded.val[0]);
            sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
            sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 <= count; idx += 8) {
            uint8x8x4_t loaded = vld4_u8((nk_u8_t const *)(data_ptr + idx * 4));
            float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(loaded.val[0]);
            sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
            sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        }
    }
    else {
        nk_e5m2_t const *ptr = data_ptr;
        for (; idx + 8 <= count; idx += 8) {
            nk_b64_vec_t data_vec = {0};
            for (nk_size_t i = 0; i < 8; ++i) {
                data_vec.u8s[i] = *ptr;
                ptr += stride_elements;
            }
            float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_vec.u8x8);
            sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
            sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        }
    }

    if (idx < count) {
        nk_b64_vec_t data_vec = {0};
        nk_e5m2_t const *ptr = data_ptr + idx * stride_elements;
        for (nk_size_t i = 0; idx + i < count; ++i) {
            data_vec.u8s[i] = *ptr;
            ptr += stride_elements;
        }
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_vec.u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }

    *sum_ptr = vaddvq_f32(sum_f32x4);
    *sumsq_ptr = vaddvq_f32(sumsq_f32x4);
}

NK_PUBLIC void nk_reduce_moments_e5m2_neonfhm(                          //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e5m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)5000 * 8) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum_value, left_sumsq_value, right_sum_value, right_sumsq_value;
        nk_reduce_moments_e5m2_neonfhm(data_ptr, left_count, stride_bytes, &left_sum_value, &left_sumsq_value);
        nk_reduce_moments_e5m2_neonfhm(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum_value, &right_sumsq_value);
        *sum_ptr = left_sum_value + right_sum_value, *sumsq_ptr = left_sumsq_value + right_sumsq_value;
    }
    else if (stride_elements == 1) nk_reduce_moments_e5m2_neonfhm_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e5m2_neonfhm_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e4m3_neonfhm_contiguous_( //
    nk_e4m3_t const *data_ptr, nk_size_t count,             //
    nk_e4m3_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_e4m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    uint8x16_t min_u8x16 = vdupq_n_u8(0xFF);
    uint8x16_t max_u8x16 = vdupq_n_u8(0x00);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data_ptr + idx));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(raw_u8x16);
        uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
        iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b8x16_serial_(data_ptr + idx, &tail_vec, remaining);
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(tail_vec.u8x16);
        uint8x16_t lane_indices_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                    vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
        uint8x16_t valid_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((uint8_t)remaining));
        uint8x16_t data_for_min_u8x16 = vbslq_u8(valid_u8x16, comparable_u8x16, vdupq_n_u8(0xFF));
        uint8x16_t data_for_max_u8x16 = vbslq_u8(valid_u8x16, comparable_u8x16, vdupq_n_u8(0));
        uint8x16_t less_u8x16 = vcltq_u8(data_for_min_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(data_for_max_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, data_for_min_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, data_for_max_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
    }
    nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
    // All-NaN early return: both sentinels unchanged means no valid data was found
    if (min_comparable == 0xFF && max_comparable == 0x00) {
        *min_value_ptr = nk_comparable_to_fp8_(min_comparable), *min_index_ptr = NK_SIZE_MAX;
        *max_value_ptr = nk_comparable_to_fp8_(max_comparable), *max_index_ptr = NK_SIZE_MAX;
        return;
    }
    uint8x16_t min_value_match_u8x16 = vceqq_u8(min_u8x16, vdupq_n_u8(min_comparable));
    uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
    uint8x16_t max_value_match_u8x16 = vceqq_u8(max_u8x16, vdupq_n_u8(max_comparable));
    uint8x16_t masked_max_iter_u8x16 = vbslq_u8(max_value_match_u8x16, max_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_max_cycle = vminvq_u8(masked_max_iter_u8x16);
    uint8x16_t lane_indices_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
    uint8x16_t min_cycle_match_u8x16 = vceqq_u8(min_iter_u8x16, vdupq_n_u8(earliest_min_cycle));
    uint8x16_t min_both_match_u8x16 = vandq_u8(min_value_match_u8x16, min_cycle_match_u8x16);
    uint8x16_t min_masked_lanes_u8x16 = vbslq_u8(min_both_match_u8x16, lane_indices_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t min_lane_offset = vminvq_u8(min_masked_lanes_u8x16);
    nk_size_t min_idx = (nk_size_t)earliest_min_cycle * 16 + (nk_size_t)min_lane_offset;
    uint8x16_t max_cycle_match_u8x16 = vceqq_u8(max_iter_u8x16, vdupq_n_u8(earliest_max_cycle));
    uint8x16_t max_both_match_u8x16 = vandq_u8(max_value_match_u8x16, max_cycle_match_u8x16);
    uint8x16_t max_masked_lanes_u8x16 = vbslq_u8(max_both_match_u8x16, lane_indices_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t max_lane_offset = vminvq_u8(max_masked_lanes_u8x16);
    nk_size_t max_idx = (nk_size_t)earliest_max_cycle * 16 + (nk_size_t)max_lane_offset;
    *min_value_ptr = nk_comparable_to_fp8_(min_comparable), *min_index_ptr = min_idx;
    *max_value_ptr = nk_comparable_to_fp8_(max_comparable), *max_index_ptr = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_e4m3_neonfhm_strided_(                   //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_e4m3_t *min_value_ptr, nk_size_t *min_index_ptr,                    //
    nk_e4m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    uint8x16_t min_u8x16 = vdupq_n_u8(0xFF), max_u8x16 = vdupq_n_u8(0);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    uint8x16_t lane_indices_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
    nk_size_t idx = 0;
    uint8x16_t data_for_min_u8x16, data_for_max_u8x16;

nk_reduce_minmax_e4m3_neonfhm_cycle:
    if (stride_elements == 2 && idx + 16 <= count) {
        uint8x16x2_t loaded = vld2q_u8((nk_u8_t const *)(data_ptr + idx * 2));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded.val[0]);
        data_for_min_u8x16 = comparable_u8x16;
        data_for_max_u8x16 = comparable_u8x16;
        idx += 16;
    }
    else if (stride_elements == 3 && idx + 16 <= count) {
        uint8x16x3_t loaded = vld3q_u8((nk_u8_t const *)(data_ptr + idx * 3));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded.val[0]);
        data_for_min_u8x16 = comparable_u8x16;
        data_for_max_u8x16 = comparable_u8x16;
        idx += 16;
    }
    else if (stride_elements == 4 && idx + 16 <= count) {
        uint8x16x4_t loaded = vld4q_u8((nk_u8_t const *)(data_ptr + idx * 4));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded.val[0]);
        data_for_min_u8x16 = comparable_u8x16;
        data_for_max_u8x16 = comparable_u8x16;
        idx += 16;
    }
    else if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_strided_load_b8x16_serial_(data_ptr + idx * stride_elements, stride_elements, &tail_vec, count - idx);
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(tail_vec.u8x16);
        uint8x16_t valid_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((uint8_t)(count - idx)));
        data_for_min_u8x16 = vbslq_u8(valid_u8x16, comparable_u8x16, vdupq_n_u8(0xFF));
        data_for_max_u8x16 = vbslq_u8(valid_u8x16, comparable_u8x16, vdupq_n_u8(0x00));
        idx = count;
    }
    else {
        nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
        if (min_comparable == 0xFF && max_comparable == 0x00) {
            *min_value_ptr = nk_comparable_to_fp8_(min_comparable), *min_index_ptr = NK_SIZE_MAX;
            *max_value_ptr = nk_comparable_to_fp8_(max_comparable), *max_index_ptr = NK_SIZE_MAX;
            return;
        }
        uint8x16_t min_value_match_u8x16 = vceqq_u8(min_u8x16, vdupq_n_u8(min_comparable));
        uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
        nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
        uint8x16_t max_value_match_u8x16 = vceqq_u8(max_u8x16, vdupq_n_u8(max_comparable));
        uint8x16_t masked_max_iter_u8x16 = vbslq_u8(max_value_match_u8x16, max_iter_u8x16, vdupq_n_u8(0xFF));
        nk_u8_t earliest_max_cycle = vminvq_u8(masked_max_iter_u8x16);
        uint8x16_t min_cycle_match_u8x16 = vceqq_u8(min_iter_u8x16, vdupq_n_u8(earliest_min_cycle));
        uint8x16_t min_both_match_u8x16 = vandq_u8(min_value_match_u8x16, min_cycle_match_u8x16);
        uint8x16_t min_masked_lanes_u8x16 = vbslq_u8(min_both_match_u8x16, lane_indices_u8x16, vdupq_n_u8(0xFF));
        nk_u8_t min_lane_offset = vminvq_u8(min_masked_lanes_u8x16);
        nk_size_t min_idx = (nk_size_t)earliest_min_cycle * 16 + (nk_size_t)min_lane_offset;
        uint8x16_t max_cycle_match_u8x16 = vceqq_u8(max_iter_u8x16, vdupq_n_u8(earliest_max_cycle));
        uint8x16_t max_both_match_u8x16 = vandq_u8(max_value_match_u8x16, max_cycle_match_u8x16);
        uint8x16_t max_masked_lanes_u8x16 = vbslq_u8(max_both_match_u8x16, lane_indices_u8x16, vdupq_n_u8(0xFF));
        nk_u8_t max_lane_offset = vminvq_u8(max_masked_lanes_u8x16);
        nk_size_t max_idx = (nk_size_t)earliest_max_cycle * 16 + (nk_size_t)max_lane_offset;
        *min_value_ptr = nk_comparable_to_fp8_(min_comparable), *min_index_ptr = min_idx;
        *max_value_ptr = nk_comparable_to_fp8_(max_comparable), *max_index_ptr = max_idx;
        return;
    }

    // Shared update body
    uint8x16_t less_u8x16 = vcltq_u8(data_for_min_u8x16, min_u8x16);
    uint8x16_t greater_u8x16 = vcgtq_u8(data_for_max_u8x16, max_u8x16);
    min_u8x16 = vbslq_u8(less_u8x16, data_for_min_u8x16, min_u8x16);
    max_u8x16 = vbslq_u8(greater_u8x16, data_for_max_u8x16, max_u8x16);
    min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
    max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
    iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
    goto nk_reduce_minmax_e4m3_neonfhm_cycle;
}

NK_PUBLIC void nk_reduce_minmax_e4m3_neonfhm(                           //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e4m3_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e4m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_E4M3_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_E4M3_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_e4m3_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (count > (nk_size_t)256 * 16) {
        nk_size_t left_count = count / 2;
        nk_e4m3_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_e4m3_neonfhm(data_ptr, left_count, stride_bytes, &left_min_value, &left_min_index,
                                      &left_max_value, &left_max_index);
        nk_reduce_minmax_e4m3_neonfhm(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                      &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (nk_e4m3_order_serial(right_min_value, left_min_value) < 0)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (nk_e4m3_order_serial(right_max_value, left_max_value) > 0)
            *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e4m3_neonfhm_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                  max_index_ptr);
    else if (stride_elements <= 4)
        nk_reduce_minmax_e4m3_neonfhm_strided_(data_ptr, count, stride_elements, min_value_ptr, min_index_ptr,
                                               max_value_ptr, max_index_ptr);
    else
        nk_reduce_minmax_e4m3_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e5m2_neonfhm_contiguous_( //
    nk_e5m2_t const *data_ptr, nk_size_t count,             //
    nk_e5m2_t *min_value_ptr, nk_size_t *min_index_ptr,     //
    nk_e5m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    uint8x16_t min_u8x16 = vdupq_n_u8(0xFF);
    uint8x16_t max_u8x16 = vdupq_n_u8(0x00);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data_ptr + idx));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(raw_u8x16);
        uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
        iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b8x16_serial_(data_ptr + idx, &tail_vec, remaining);
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(tail_vec.u8x16);
        uint8x16_t lane_indices_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                    vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
        uint8x16_t valid_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((uint8_t)remaining));
        uint8x16_t data_for_min_u8x16 = vbslq_u8(valid_u8x16, comparable_u8x16, vdupq_n_u8(0xFF));
        uint8x16_t data_for_max_u8x16 = vbslq_u8(valid_u8x16, comparable_u8x16, vdupq_n_u8(0));
        uint8x16_t less_u8x16 = vcltq_u8(data_for_min_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(data_for_max_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, data_for_min_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, data_for_max_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
    }
    nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
    // All-NaN early return: both sentinels unchanged means no valid data was found
    if (min_comparable == 0xFF && max_comparable == 0x00) {
        *min_value_ptr = nk_comparable_to_fp8_(min_comparable), *min_index_ptr = NK_SIZE_MAX;
        *max_value_ptr = nk_comparable_to_fp8_(max_comparable), *max_index_ptr = NK_SIZE_MAX;
        return;
    }
    uint8x16_t min_value_match_u8x16 = vceqq_u8(min_u8x16, vdupq_n_u8(min_comparable));
    uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
    uint8x16_t max_value_match_u8x16 = vceqq_u8(max_u8x16, vdupq_n_u8(max_comparable));
    uint8x16_t masked_max_iter_u8x16 = vbslq_u8(max_value_match_u8x16, max_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_max_cycle = vminvq_u8(masked_max_iter_u8x16);
    uint8x16_t lane_indices_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
    uint8x16_t min_cycle_match_u8x16 = vceqq_u8(min_iter_u8x16, vdupq_n_u8(earliest_min_cycle));
    uint8x16_t min_both_match_u8x16 = vandq_u8(min_value_match_u8x16, min_cycle_match_u8x16);
    uint8x16_t min_masked_lanes_u8x16 = vbslq_u8(min_both_match_u8x16, lane_indices_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t min_lane_offset = vminvq_u8(min_masked_lanes_u8x16);
    nk_size_t min_idx = (nk_size_t)earliest_min_cycle * 16 + (nk_size_t)min_lane_offset;
    uint8x16_t max_cycle_match_u8x16 = vceqq_u8(max_iter_u8x16, vdupq_n_u8(earliest_max_cycle));
    uint8x16_t max_both_match_u8x16 = vandq_u8(max_value_match_u8x16, max_cycle_match_u8x16);
    uint8x16_t max_masked_lanes_u8x16 = vbslq_u8(max_both_match_u8x16, lane_indices_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t max_lane_offset = vminvq_u8(max_masked_lanes_u8x16);
    nk_size_t max_idx = (nk_size_t)earliest_max_cycle * 16 + (nk_size_t)max_lane_offset;
    *min_value_ptr = nk_comparable_to_fp8_(min_comparable), *min_index_ptr = min_idx;
    *max_value_ptr = nk_comparable_to_fp8_(max_comparable), *max_index_ptr = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_e5m2_neonfhm_strided_(                   //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_e5m2_t *min_value_ptr, nk_size_t *min_index_ptr,                    //
    nk_e5m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    uint8x16_t min_u8x16 = vdupq_n_u8(0xFF), max_u8x16 = vdupq_n_u8(0);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    uint8x16_t lane_indices_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
    nk_size_t idx = 0;
    uint8x16_t data_for_min_u8x16, data_for_max_u8x16;

nk_reduce_minmax_e5m2_neonfhm_cycle:
    if (stride_elements == 2 && idx + 16 <= count) {
        uint8x16x2_t loaded = vld2q_u8((nk_u8_t const *)(data_ptr + idx * 2));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded.val[0]);
        data_for_min_u8x16 = comparable_u8x16;
        data_for_max_u8x16 = comparable_u8x16;
        idx += 16;
    }
    else if (stride_elements == 3 && idx + 16 <= count) {
        uint8x16x3_t loaded = vld3q_u8((nk_u8_t const *)(data_ptr + idx * 3));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded.val[0]);
        data_for_min_u8x16 = comparable_u8x16;
        data_for_max_u8x16 = comparable_u8x16;
        idx += 16;
    }
    else if (stride_elements == 4 && idx + 16 <= count) {
        uint8x16x4_t loaded = vld4q_u8((nk_u8_t const *)(data_ptr + idx * 4));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded.val[0]);
        data_for_min_u8x16 = comparable_u8x16;
        data_for_max_u8x16 = comparable_u8x16;
        idx += 16;
    }
    else if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_strided_load_b8x16_serial_(data_ptr + idx * stride_elements, stride_elements, &tail_vec, count - idx);
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(tail_vec.u8x16);
        uint8x16_t valid_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((uint8_t)(count - idx)));
        data_for_min_u8x16 = vbslq_u8(valid_u8x16, comparable_u8x16, vdupq_n_u8(0xFF));
        data_for_max_u8x16 = vbslq_u8(valid_u8x16, comparable_u8x16, vdupq_n_u8(0x00));
        idx = count;
    }
    else {
        nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
        if (min_comparable == 0xFF && max_comparable == 0x00) {
            *min_value_ptr = nk_comparable_to_fp8_(min_comparable), *min_index_ptr = NK_SIZE_MAX;
            *max_value_ptr = nk_comparable_to_fp8_(max_comparable), *max_index_ptr = NK_SIZE_MAX;
            return;
        }
        uint8x16_t min_value_match_u8x16 = vceqq_u8(min_u8x16, vdupq_n_u8(min_comparable));
        uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
        nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
        uint8x16_t max_value_match_u8x16 = vceqq_u8(max_u8x16, vdupq_n_u8(max_comparable));
        uint8x16_t masked_max_iter_u8x16 = vbslq_u8(max_value_match_u8x16, max_iter_u8x16, vdupq_n_u8(0xFF));
        nk_u8_t earliest_max_cycle = vminvq_u8(masked_max_iter_u8x16);
        uint8x16_t min_cycle_match_u8x16 = vceqq_u8(min_iter_u8x16, vdupq_n_u8(earliest_min_cycle));
        uint8x16_t min_both_match_u8x16 = vandq_u8(min_value_match_u8x16, min_cycle_match_u8x16);
        uint8x16_t min_masked_lanes_u8x16 = vbslq_u8(min_both_match_u8x16, lane_indices_u8x16, vdupq_n_u8(0xFF));
        nk_u8_t min_lane_offset = vminvq_u8(min_masked_lanes_u8x16);
        nk_size_t min_idx = (nk_size_t)earliest_min_cycle * 16 + (nk_size_t)min_lane_offset;
        uint8x16_t max_cycle_match_u8x16 = vceqq_u8(max_iter_u8x16, vdupq_n_u8(earliest_max_cycle));
        uint8x16_t max_both_match_u8x16 = vandq_u8(max_value_match_u8x16, max_cycle_match_u8x16);
        uint8x16_t max_masked_lanes_u8x16 = vbslq_u8(max_both_match_u8x16, lane_indices_u8x16, vdupq_n_u8(0xFF));
        nk_u8_t max_lane_offset = vminvq_u8(max_masked_lanes_u8x16);
        nk_size_t max_idx = (nk_size_t)earliest_max_cycle * 16 + (nk_size_t)max_lane_offset;
        *min_value_ptr = nk_comparable_to_fp8_(min_comparable), *min_index_ptr = min_idx;
        *max_value_ptr = nk_comparable_to_fp8_(max_comparable), *max_index_ptr = max_idx;
        return;
    }

    // Shared update body
    uint8x16_t less_u8x16 = vcltq_u8(data_for_min_u8x16, min_u8x16);
    uint8x16_t greater_u8x16 = vcgtq_u8(data_for_max_u8x16, max_u8x16);
    min_u8x16 = vbslq_u8(less_u8x16, data_for_min_u8x16, min_u8x16);
    max_u8x16 = vbslq_u8(greater_u8x16, data_for_max_u8x16, max_u8x16);
    min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
    max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
    iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
    goto nk_reduce_minmax_e5m2_neonfhm_cycle;
}

NK_PUBLIC void nk_reduce_minmax_e5m2_neonfhm(                           //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e5m2_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e5m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_E5M2_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_E5M2_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_e5m2_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (count > (nk_size_t)256 * 16) {
        nk_size_t left_count = count / 2;
        nk_e5m2_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_e5m2_neonfhm(data_ptr, left_count, stride_bytes, &left_min_value, &left_min_index,
                                      &left_max_value, &left_max_index);
        nk_reduce_minmax_e5m2_neonfhm(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                      &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (nk_e5m2_order_serial(right_min_value, left_min_value) < 0)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (nk_e5m2_order_serial(right_max_value, left_max_value) > 0)
            *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e5m2_neonfhm_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                  max_index_ptr);
    else if (stride_elements <= 4)
        nk_reduce_minmax_e5m2_neonfhm_strided_(data_ptr, count, stride_elements, min_value_ptr, min_index_ptr,
                                               max_value_ptr, max_index_ptr);
    else
        nk_reduce_minmax_e5m2_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONFHM
#endif // NK_TARGET_ARM_
#endif // NK_REDUCE_NEONFHM_H
