/**
 *  @brief NEON FP16 implementations for the redesigned reduction API (moments + minmax).
 *  @file include/numkong/reduce/neonhalf.h
 *  @author Ash Vardanian
 *  @date February 13, 2026
 *
 *  @sa include/numkong/reduce.h
 *
 *  @section reduce_neonhalf_new_design Design Notes
 *
 *  Moments (sum + sum-of-squares) accumulate in f32 via vcvt_f32_f16 widening, giving
 *  full f32 precision. The contiguous path processes 8 f16 elements per iteration, widening
 *  to two f32x4 halves and using vfmaq_f32 for fused multiply-accumulate of squares.
 *
 *  Minmax tracks min/max values as native f16x8 with u16x8 iteration counters (same width
 *  as f16). The u16 counters wrap at 65536, so the dispatcher splits arrays larger than
 *  65536 * 8 = 524288 elements via recursive halving.
 */
#ifndef NK_REDUCE_NEONHALF_H
#define NK_REDUCE_NEONHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF

#include "numkong/types.h"
#include "numkong/cast/neon.h"
#include "numkong/cast/serial.h"
#include "numkong/reduce/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

NK_INTERNAL void nk_reduce_moments_f16_neonhalf_contiguous_( //
    nk_f16_t const *data_ptr, nk_size_t count,               //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        float16x8_t data_f16x8 = vld1q_f16((nk_f16_for_arm_simd_t const *)(data_ptr + idx));
        float32x4_t low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        sum_f32x4 = vaddq_f32(sum_f32x4, low_f32x4);
        sum_f32x4 = vaddq_f32(sum_f32x4, high_f32x4);
        sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, low_f32x4, low_f32x4);
        sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, high_f32x4, high_f32x4);
    }

    // Scalar tail
    nk_f32_t sum = vaddvq_f32(sum_f32x4);
    nk_f32_t sumsq = vaddvq_f32(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t value_f32;
        nk_f16_to_f32_serial(data_ptr + idx, &value_f32);
        sum += value_f32, sumsq += value_f32 * value_f32;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_f16_neonhalf_strided_(                 //
    nk_f16_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16((uint16_t const *)(data_ptr + idx * 2));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x2.val[0]);
            float32x4_t low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
            float32x4_t high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
            sum_f32x4 = vaddq_f32(sum_f32x4, low_f32x4);
            sum_f32x4 = vaddq_f32(sum_f32x4, high_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, low_f32x4, low_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, high_f32x4, high_f32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16((uint16_t const *)(data_ptr + idx * 3));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x3.val[0]);
            float32x4_t low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
            float32x4_t high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
            sum_f32x4 = vaddq_f32(sum_f32x4, low_f32x4);
            sum_f32x4 = vaddq_f32(sum_f32x4, high_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, low_f32x4, low_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, high_f32x4, high_f32x4);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16((uint16_t const *)(data_ptr + idx * 4));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x4.val[0]);
            float32x4_t low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
            float32x4_t high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
            sum_f32x4 = vaddq_f32(sum_f32x4, low_f32x4);
            sum_f32x4 = vaddq_f32(sum_f32x4, high_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, low_f32x4, low_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, high_f32x4, high_f32x4);
        }
    }

    // Scalar tail for remaining elements
    nk_f32_t sum = vaddvq_f32(sum_f32x4);
    nk_f32_t sumsq = vaddvq_f32(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t value_f32;
        nk_f16_to_f32_serial((nk_f16_t const *)(data_ptr + idx * stride_elements), &value_f32);
        sum += value_f32, sumsq += value_f32 * value_f32;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_f16_neonhalf(                         //
    nk_f16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_f16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum_value, left_sumsq_value, right_sum_value, right_sumsq_value;
        nk_reduce_moments_f16_neonhalf(data_ptr, left_count, stride_bytes, &left_sum_value, &left_sumsq_value);
        nk_reduce_moments_f16_neonhalf(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum_value, &right_sumsq_value);
        *sum_ptr = left_sum_value + right_sum_value, *sumsq_ptr = left_sumsq_value + right_sumsq_value;
    }
    else if (stride_elements == 1) nk_reduce_moments_f16_neonhalf_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_f16_neonhalf_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_f16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONHALF
#endif // NK_TARGET_ARM_
#endif // NK_REDUCE_NEONHALF_H
