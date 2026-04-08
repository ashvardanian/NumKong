/**
 *  @brief ARMv8.6-BF16 implementations for the redesigned reduction API.
 *  @file include/numkong/reduce/neonbfdot.h
 *  @author Ash Vardanian
 *  @date February 13, 2026
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_NEONBFDOT_H
#define NK_REDUCE_NEONBFDOT_H

#if NK_TARGET_ARM64_
#if NK_TARGET_NEONBFDOT

#include "numkong/types.h"         // `nk_bf16_t`
#include "numkong/cast/neon.h"     // `nk_e4m3x8_to_f16x8_neon_`
#include "numkong/cast/serial.h"   // `nk_f32_to_bf16_serial`
#include "numkong/reduce/serial.h" // `nk_reduce_moments_bf16_serial`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16+fp16")
#endif

NK_INTERNAL void nk_reduce_moments_bf16_neonbfdot_contiguous_( //
    nk_bf16_t const *data_ptr, nk_size_t count,                //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    // bf16 representation of 1.0 is 0x3F80 (same as upper 16 bits of f32 1.0)
    bfloat16x8_t ones_bf16x8 = vreinterpretq_bf16_u16(vdupq_n_u16(0x3F80));
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        bfloat16x8_t data_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)(data_ptr + idx));
        sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
        sumsq_f32x4 = vbfdotq_f32(sumsq_f32x4, data_bf16x8, data_bf16x8);
    }

    // Handle tail with type-agnostic partial load
    if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b16x8_serial_(data_ptr + idx, &tail_vec, count - idx);
        bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(tail_vec.u16x8);
        sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
        sumsq_f32x4 = vbfdotq_f32(sumsq_f32x4, data_bf16x8, data_bf16x8);
    }

    *sum_ptr = vaddvq_f32(sum_f32x4);
    *sumsq_ptr = vaddvq_f32(sumsq_f32x4);
}

NK_INTERNAL void nk_reduce_moments_bf16_neonbfdot_strided_(                //
    nk_bf16_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    bfloat16x8_t ones_bf16x8 = vreinterpretq_bf16_u16(vdupq_n_u16(0x3F80));
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 < count; idx += 8) {
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16((nk_u16_t const *)(data_ptr + idx * 2));
            bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(loaded_u16x8x2.val[0]);
            sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
            sumsq_f32x4 = vbfdotq_f32(sumsq_f32x4, data_bf16x8, data_bf16x8);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 < count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16((nk_u16_t const *)(data_ptr + idx * 3));
            bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(loaded_u16x8x3.val[0]);
            sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
            sumsq_f32x4 = vbfdotq_f32(sumsq_f32x4, data_bf16x8, data_bf16x8);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 < count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16((nk_u16_t const *)(data_ptr + idx * 4));
            bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(loaded_u16x8x4.val[0]);
            sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
            sumsq_f32x4 = vbfdotq_f32(sumsq_f32x4, data_bf16x8, data_bf16x8);
        }
    }

    // Gather tail into contiguous buffer, then dot with ones
    if (idx < count) {
        nk_b128_vec_t tail_vec = {0};
        nk_size_t remaining = count - idx;
        for (nk_size_t k = 0; k < remaining; ++k)
            tail_vec.u16s[k] = *(nk_u16_t const *)(data_ptr + (idx + k) * stride_elements);
        bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(tail_vec.u16x8);
        sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
        sumsq_f32x4 = vbfdotq_f32(sumsq_f32x4, data_bf16x8, data_bf16x8);
    }

    *sum_ptr = vaddvq_f32(sum_f32x4);
    *sumsq_ptr = vaddvq_f32(sumsq_f32x4);
}

NK_PUBLIC void nk_reduce_moments_bf16_neonbfdot(                        //
    nk_bf16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_bf16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum_value, left_sumsq_value, right_sum_value, right_sumsq_value;
        nk_reduce_moments_bf16_neonbfdot(data_ptr, left_count, stride_bytes, &left_sum_value, &left_sumsq_value);
        nk_reduce_moments_bf16_neonbfdot(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                         &right_sum_value, &right_sumsq_value);
        *sum_ptr = left_sum_value + right_sum_value, *sumsq_ptr = left_sumsq_value + right_sumsq_value;
    }
    else if (stride_elements == 1) nk_reduce_moments_bf16_neonbfdot_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_bf16_neonbfdot_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_bf16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONBFDOT
#endif // NK_TARGET_ARM64_
#endif // NK_REDUCE_NEONBFDOT_H
