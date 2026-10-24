/**
 *  @brief ARMv8.4-DotProd implementations for the redesigned reduction API (moments).
 *  @file include/numkong/reduce/neonsdot.h
 *  @author Ash Vardanian
 *  @date February 13, 2026
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_NEONSDOT_H
#define NK_REDUCE_NEONSDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONSDOT

#include "numkong/types.h"
#include "numkong/cast/serial.h"
#include "numkong/reduce/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod")
#endif

NK_INTERNAL void nk_reduce_moments_i8_neonsdot_contiguous_( //
    nk_i8_t const *data_ptr, nk_size_t count,               //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    int8x16_t ones_i8x16 = vdupq_n_s8(1);
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    int32x4_t sumsq_i32x4 = vdupq_n_s32(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        int8x16_t data_i8x16 = vld1q_s8(data_ptr + idx);
        sum_i32x4 = vdotq_s32(sum_i32x4, data_i8x16, ones_i8x16);
        sumsq_i32x4 = vdotq_s32(sumsq_i32x4, data_i8x16, data_i8x16);
    }
    // Widen i32 -> i64 and horizontal reduce
    int64x2_t sum_i64x2 = vpaddlq_s32(sum_i32x4);
    nk_i64_t sum = vgetq_lane_s64(sum_i64x2, 0) + vgetq_lane_s64(sum_i64x2, 1);
    uint64x2_t sumsq_u64x2 = vpaddlq_u32(vreinterpretq_u32_s32(sumsq_i32x4));
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_i64_t value = (nk_i64_t)data_ptr[idx];
        sum += value, sumsq += (nk_u64_t)(value * value);
    }
    *sum_ptr = sum;
    *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_i8_neonsdot_strided_(                 //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    int8x16_t ones_i8x16 = vdupq_n_s8(1);
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    int32x4_t sumsq_i32x4 = vdupq_n_s32(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            int8x16x2_t loaded = vld2q_s8(data_ptr + idx * 2);
            int8x16_t data_i8x16 = loaded.val[0];
            sum_i32x4 = vdotq_s32(sum_i32x4, data_i8x16, ones_i8x16);
            sumsq_i32x4 = vdotq_s32(sumsq_i32x4, data_i8x16, data_i8x16);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            int8x16x3_t loaded = vld3q_s8(data_ptr + idx * 3);
            int8x16_t data_i8x16 = loaded.val[0];
            sum_i32x4 = vdotq_s32(sum_i32x4, data_i8x16, ones_i8x16);
            sumsq_i32x4 = vdotq_s32(sumsq_i32x4, data_i8x16, data_i8x16);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 16 <= count; idx += 16) {
            int8x16x4_t loaded = vld4q_s8(data_ptr + idx * 4);
            int8x16_t data_i8x16 = loaded.val[0];
            sum_i32x4 = vdotq_s32(sum_i32x4, data_i8x16, ones_i8x16);
            sumsq_i32x4 = vdotq_s32(sumsq_i32x4, data_i8x16, data_i8x16);
        }
    }
    // Widen i32 -> i64 and horizontal reduce
    int64x2_t sum_i64x2 = vpaddlq_s32(sum_i32x4);
    nk_i64_t sum = vgetq_lane_s64(sum_i64x2, 0) + vgetq_lane_s64(sum_i64x2, 1);
    uint64x2_t sumsq_u64x2 = vpaddlq_u32(vreinterpretq_u32_s32(sumsq_i32x4));
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_i64_t value = (nk_i64_t)data_ptr[idx * stride_elements];
        sum += value, sumsq += (nk_u64_t)(value * value);
    }
    *sum_ptr = sum;
    *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i8_neonsdot(                         //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_i8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)32768 * 16) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum_value, right_sum_value;
        nk_u64_t left_sumsq_value, right_sumsq_value;
        nk_reduce_moments_i8_neonsdot(data_ptr, left_count, stride_bytes, &left_sum_value, &left_sumsq_value);
        nk_reduce_moments_i8_neonsdot(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                      &right_sum_value, &right_sumsq_value);
        *sum_ptr = nk_i64_saturating_add_serial(left_sum_value, right_sum_value);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq_value, right_sumsq_value);
    }
    else if (stride_elements == 1) nk_reduce_moments_i8_neonsdot_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_i8_neonsdot_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_i8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_u8_neonsdot_contiguous_( //
    nk_u8_t const *data_ptr, nk_size_t count,               //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    uint8x16_t ones_u8x16 = vdupq_n_u8(1);
    uint32x4_t sum_u32x4 = vdupq_n_u32(0);
    uint32x4_t sumsq_u32x4 = vdupq_n_u32(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t data_u8x16 = vld1q_u8(data_ptr + idx);
        sum_u32x4 = vdotq_u32(sum_u32x4, data_u8x16, ones_u8x16);
        sumsq_u32x4 = vdotq_u32(sumsq_u32x4, data_u8x16, data_u8x16);
    }
    uint64x2_t sum_u64x2 = vpaddlq_u32(sum_u32x4);
    nk_u64_t sum = vgetq_lane_u64(sum_u64x2, 0) + vgetq_lane_u64(sum_u64x2, 1);
    uint64x2_t sumsq_u64x2 = vpaddlq_u32(sumsq_u32x4);
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_u64_t value = (nk_u64_t)data_ptr[idx];
        sum += value, sumsq += value * value;
    }
    *sum_ptr = sum;
    *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_u8_neonsdot_strided_(                 //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    uint8x16_t ones_u8x16 = vdupq_n_u8(1);
    uint32x4_t sum_u32x4 = vdupq_n_u32(0);
    uint32x4_t sumsq_u32x4 = vdupq_n_u32(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x2_t loaded = vld2q_u8(data_ptr + idx * 2);
            uint8x16_t data_u8x16 = loaded.val[0];
            sum_u32x4 = vdotq_u32(sum_u32x4, data_u8x16, ones_u8x16);
            sumsq_u32x4 = vdotq_u32(sumsq_u32x4, data_u8x16, data_u8x16);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x3_t loaded = vld3q_u8(data_ptr + idx * 3);
            uint8x16_t data_u8x16 = loaded.val[0];
            sum_u32x4 = vdotq_u32(sum_u32x4, data_u8x16, ones_u8x16);
            sumsq_u32x4 = vdotq_u32(sumsq_u32x4, data_u8x16, data_u8x16);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x4_t loaded = vld4q_u8(data_ptr + idx * 4);
            uint8x16_t data_u8x16 = loaded.val[0];
            sum_u32x4 = vdotq_u32(sum_u32x4, data_u8x16, ones_u8x16);
            sumsq_u32x4 = vdotq_u32(sumsq_u32x4, data_u8x16, data_u8x16);
        }
    }
    uint64x2_t sum_u64x2 = vpaddlq_u32(sum_u32x4);
    nk_u64_t sum = vgetq_lane_u64(sum_u64x2, 0) + vgetq_lane_u64(sum_u64x2, 1);
    uint64x2_t sumsq_u64x2 = vpaddlq_u32(sumsq_u32x4);
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_u64_t value = (nk_u64_t)data_ptr[idx * stride_elements];
        sum += value, sumsq += value * value;
    }
    *sum_ptr = sum;
    *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u8_neonsdot(                         //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_u8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)16384 * 16) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum_value, left_sumsq_value, right_sum_value, right_sumsq_value;
        nk_reduce_moments_u8_neonsdot(data_ptr, left_count, stride_bytes, &left_sum_value, &left_sumsq_value);
        nk_reduce_moments_u8_neonsdot(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                      &right_sum_value, &right_sumsq_value);
        *sum_ptr = nk_u64_saturating_add_serial(left_sum_value, right_sum_value);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq_value, right_sumsq_value);
    }
    else if (stride_elements == 1) nk_reduce_moments_u8_neonsdot_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_u8_neonsdot_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_u8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_e2m3_neonsdot_contiguous_( //
    nk_e2m3_t const *data_ptr, nk_size_t count,               //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    uint8x16x2_t lut_e2m3_x16;
    // table[0]: values for magnitudes 0..15
    // 0x0E0C0A0806040200 → bytes [0..7]  = 0,2,4,6,8,10,12,14
    // 0x1E1C1A1816141210 → bytes [8..15] = 16,18,20,22,24,26,28,30
    lut_e2m3_x16.val[0] = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0E0C0A0806040200ULL)),
                                      vreinterpret_u8_u64(vcreate_u64(0x1E1C1A1816141210ULL)));
    // table[1]: values for magnitudes 16..31
    // 0x3C3834302C282420 → bytes [0..7]  = 32,36,40,44,48,52,56,60
    // 0x7870686058504840 → bytes [8..15] = 64,72,80,88,96,104,112,120
    lut_e2m3_x16.val[1] = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x3C3834302C282420ULL)),
                                      vreinterpret_u8_u64(vcreate_u64(0x7870686058504840ULL)));
    int8x16_t ones_i8x16 = vdupq_n_s8(1);
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    int32x4_t sumsq_i32x4 = vdupq_n_s32(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data_ptr + idx));
        uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
        uint8x16_t unsigned_u8x16 = vqtbl2q_u8(lut_e2m3_x16, magnitude_u8x16);
        uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
        int8x16_t positive_i8x16 = vreinterpretq_s8_u8(unsigned_u8x16);
        int8x16_t negative_i8x16 = vnegq_s8(positive_i8x16);
        int8x16_t scaled_i8x16 = vbslq_s8(is_negative_u8x16, negative_i8x16, positive_i8x16);
        sum_i32x4 = vdotq_s32(sum_i32x4, scaled_i8x16, ones_i8x16);
        sumsq_i32x4 = vdotq_s32(sumsq_i32x4, scaled_i8x16, scaled_i8x16);
    }
    int64x2_t sum_i64x2 = vpaddlq_s32(sum_i32x4);
    nk_i64_t sum = vgetq_lane_s64(sum_i64x2, 0) + vgetq_lane_s64(sum_i64x2, 1);
    uint64x2_t sumsq_u64x2 = vpaddlq_u32(vreinterpretq_u32_s32(sumsq_i32x4));
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_f32_t value;
        nk_e2m3_to_f32_serial(&data_ptr[idx], &value);
        sum += (nk_i64_t)(value * 16.0f), sumsq += (nk_u64_t)(nk_i64_t)(value * value * 256.0f);
    }
    *sum_ptr = (nk_f32_t)sum / 16.0f, *sumsq_ptr = (nk_f32_t)sumsq / 256.0f;
}

NK_INTERNAL void nk_reduce_moments_e2m3_neonsdot_strided_(                 //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    uint8x16x2_t lut_e2m3_x16;
    // table[0]: values for magnitudes 0..15
    // 0x0E0C0A0806040200 → bytes [0..7]  = 0,2,4,6,8,10,12,14
    // 0x1E1C1A1816141210 → bytes [8..15] = 16,18,20,22,24,26,28,30
    lut_e2m3_x16.val[0] = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0E0C0A0806040200ULL)),
                                      vreinterpret_u8_u64(vcreate_u64(0x1E1C1A1816141210ULL)));
    // table[1]: values for magnitudes 16..31
    // 0x3C3834302C282420 → bytes [0..7]  = 32,36,40,44,48,52,56,60
    // 0x7870686058504840 → bytes [8..15] = 64,72,80,88,96,104,112,120
    lut_e2m3_x16.val[1] = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x3C3834302C282420ULL)),
                                      vreinterpret_u8_u64(vcreate_u64(0x7870686058504840ULL)));
    int8x16_t ones_i8x16 = vdupq_n_s8(1);
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    int32x4_t sumsq_i32x4 = vdupq_n_s32(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x2_t loaded_u8x16x2 = vld2q_u8((nk_u8_t const *)(data_ptr + idx * 2));
            uint8x16_t raw_u8x16 = loaded_u8x16x2.val[0];
            uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
            uint8x16_t unsigned_u8x16 = vqtbl2q_u8(lut_e2m3_x16, magnitude_u8x16);
            uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
            int8x16_t positive_i8x16 = vreinterpretq_s8_u8(unsigned_u8x16);
            int8x16_t negative_i8x16 = vnegq_s8(positive_i8x16);
            int8x16_t scaled_i8x16 = vbslq_s8(is_negative_u8x16, negative_i8x16, positive_i8x16);
            sum_i32x4 = vdotq_s32(sum_i32x4, scaled_i8x16, ones_i8x16);
            sumsq_i32x4 = vdotq_s32(sumsq_i32x4, scaled_i8x16, scaled_i8x16);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x3_t loaded_u8x16x3 = vld3q_u8((nk_u8_t const *)(data_ptr + idx * 3));
            uint8x16_t raw_u8x16 = loaded_u8x16x3.val[0];
            uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
            uint8x16_t unsigned_u8x16 = vqtbl2q_u8(lut_e2m3_x16, magnitude_u8x16);
            uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
            int8x16_t positive_i8x16 = vreinterpretq_s8_u8(unsigned_u8x16);
            int8x16_t negative_i8x16 = vnegq_s8(positive_i8x16);
            int8x16_t scaled_i8x16 = vbslq_s8(is_negative_u8x16, negative_i8x16, positive_i8x16);
            sum_i32x4 = vdotq_s32(sum_i32x4, scaled_i8x16, ones_i8x16);
            sumsq_i32x4 = vdotq_s32(sumsq_i32x4, scaled_i8x16, scaled_i8x16);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x4_t loaded_u8x16x4 = vld4q_u8((nk_u8_t const *)(data_ptr + idx * 4));
            uint8x16_t raw_u8x16 = loaded_u8x16x4.val[0];
            uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
            uint8x16_t unsigned_u8x16 = vqtbl2q_u8(lut_e2m3_x16, magnitude_u8x16);
            uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
            int8x16_t positive_i8x16 = vreinterpretq_s8_u8(unsigned_u8x16);
            int8x16_t negative_i8x16 = vnegq_s8(positive_i8x16);
            int8x16_t scaled_i8x16 = vbslq_s8(is_negative_u8x16, negative_i8x16, positive_i8x16);
            sum_i32x4 = vdotq_s32(sum_i32x4, scaled_i8x16, ones_i8x16);
            sumsq_i32x4 = vdotq_s32(sumsq_i32x4, scaled_i8x16, scaled_i8x16);
        }
    }
    int64x2_t sum_i64x2 = vpaddlq_s32(sum_i32x4);
    nk_i64_t sum = vgetq_lane_s64(sum_i64x2, 0) + vgetq_lane_s64(sum_i64x2, 1);
    uint64x2_t sumsq_u64x2 = vpaddlq_u32(vreinterpretq_u32_s32(sumsq_i32x4));
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_f32_t value;
        nk_e2m3_to_f32_serial(data_ptr + idx * stride_elements, &value);
        sum += (nk_i64_t)(value * 16.0f), sumsq += (nk_u64_t)(nk_i64_t)(value * value * 256.0f);
    }
    *sum_ptr = (nk_f32_t)sum / 16.0f, *sumsq_ptr = (nk_f32_t)sumsq / 256.0f;
}

NK_PUBLIC void nk_reduce_moments_e2m3_neonsdot(                         //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    int aligned = (stride_bytes % sizeof(nk_e2m3_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e2m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_I16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum_value, left_sumsq_value, right_sum_value, right_sumsq_value;
        nk_reduce_moments_e2m3_neonsdot(data_ptr, left_count, stride_bytes, &left_sum_value, &left_sumsq_value);
        nk_reduce_moments_e2m3_neonsdot(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                        &right_sum_value, &right_sumsq_value);
        *sum_ptr = left_sum_value + right_sum_value, *sumsq_ptr = left_sumsq_value + right_sumsq_value;
    }
    else if (stride_elements == 1) nk_reduce_moments_e2m3_neonsdot_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_e2m3_neonsdot_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e2m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONSDOT
#endif // NK_TARGET_ARM_
#endif // NK_REDUCE_NEONSDOT_H
