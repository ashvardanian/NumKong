/**
 *  @brief ARMv8.4-FHM implementations for the redesigned reduction API.
 *  @file include/numkong/reduce/neonfhm.h
 *  @author Ash Vardanian
 *  @date February 13, 2026
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_NEONFHM_H
#define NK_REDUCE_NEONFHM_H

#if NK_TARGET_ARM64_
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

NK_HELPER_INLINE void nk_reduce_moments_e4m3_neonfhm_contiguous_( //
    nk_e4m3_t const *data_ptr, nk_size_t count,                   //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vreinterpretq_f16_u16(nk_u16x8_splat_(0x3C00));
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        uint8x8_t data_u8x8 = vld1_u8((nk_u8_t const *)(data_ptr + idx));
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

NK_HELPER_INLINE void nk_reduce_moments_e4m3_neonfhm_strided_(             //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vreinterpretq_f16_u16(nk_u16x8_splat_(0x3C00));
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 < count; idx += 8) {
            uint8x8x2_t loaded_u8x8x2 = vld2_u8((nk_u8_t const *)(data_ptr + idx * 2));
            float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(loaded_u8x8x2.val[0]);
            sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
            sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 < count; idx += 8) {
            uint8x8x3_t loaded_u8x8x3 = vld3_u8((nk_u8_t const *)(data_ptr + idx * 3));
            float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(loaded_u8x8x3.val[0]);
            sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
            sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 < count; idx += 8) {
            uint8x8x4_t loaded_u8x8x4 = vld4_u8((nk_u8_t const *)(data_ptr + idx * 4));
            float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(loaded_u8x8x4.val[0]);
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

NK_API_COMPTIME void nk_reduce_moments_e4m3_neonfhm(                    //
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

NK_HELPER_INLINE void nk_reduce_moments_e5m2_neonfhm_contiguous_( //
    nk_e5m2_t const *data_ptr, nk_size_t count,                   //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vreinterpretq_f16_u16(nk_u16x8_splat_(0x3C00));
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        uint8x8_t data_u8x8 = vld1_u8((nk_u8_t const *)(data_ptr + idx));
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

NK_HELPER_INLINE void nk_reduce_moments_e5m2_neonfhm_strided_(             //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vreinterpretq_f16_u16(nk_u16x8_splat_(0x3C00));
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 < count; idx += 8) {
            uint8x8x2_t loaded_u8x8x2 = vld2_u8((nk_u8_t const *)(data_ptr + idx * 2));
            float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(loaded_u8x8x2.val[0]);
            sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
            sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 < count; idx += 8) {
            uint8x8x3_t loaded_u8x8x3 = vld3_u8((nk_u8_t const *)(data_ptr + idx * 3));
            float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(loaded_u8x8x3.val[0]);
            sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
            sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
            sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 < count; idx += 8) {
            uint8x8x4_t loaded_u8x8x4 = vld4_u8((nk_u8_t const *)(data_ptr + idx * 4));
            float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(loaded_u8x8x4.val[0]);
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

NK_API_COMPTIME void nk_reduce_moments_e5m2_neonfhm(                    //
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

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONFHM
#endif // NK_TARGET_ARM64_
#endif // NK_REDUCE_NEONFHM_H
