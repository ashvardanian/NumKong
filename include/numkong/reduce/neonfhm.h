/**
 *  @file include/numkong/reduce/neonfhm.h
 *  @author Ash Vardanian
 *  @date February 13, 2026
 *  @brief ARMv8.4-FHM implementations for the redesigned reduction API.
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NUMKONG_REDUCE_NEONFHM_H
#define NUMKONG_REDUCE_NEONFHM_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONFHM

#include "numkong/types.h"         // `nk_e4m3_t`
#include "numkong/cast/serial.h"   // `nk_partial_load_b8x8_serial_`
#include "numkong/cast/neon.h"     // `nk_e4m3x8_to_f16x8_neon_`
#include "numkong/reduce/serial.h" // `nk_reduce_moments_strided_e4m3_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

NUMKONG_INLINE void nk_reduce_moments_contiguous_e4m3_neonfhm_( //
    nk_e4m3_t const *data_ptr, nk_size_t count,                 //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vreinterpretq_f16_u16(nk_splat_u16x8_neon_(0x3C00));
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

NUMKONG_INLINE void nk_reduce_moments_strided_e4m3_neonfhm_(               //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vreinterpretq_f16_u16(nk_splat_u16x8_neon_(0x3C00));
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

NUMKONG_INLINE void nk_reduce_moments_chunked_e4m3_neonfhm_(      //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride / sizeof(nk_e4m3_t);
    int aligned = (stride % sizeof(nk_e4m3_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_strided_e4m3_serial_(data_ptr, count, stride, sum_ptr, sumsq_ptr);
    else {
        nk_size_t const chunk_limit = (nk_size_t)5000 * 8;
        for (nk_size_t start = 0; start < count; start += chunk_limit) {
            nk_e4m3_t const *chunk_ptr = data_ptr + start * stride_elements;
            nk_size_t chunk_count = count - start < chunk_limit ? count - start : chunk_limit;
            nk_f32_t sum, sumsq;
            if (stride_elements == 1) nk_reduce_moments_contiguous_e4m3_neonfhm_(chunk_ptr, chunk_count, &sum, &sumsq);
            else nk_reduce_moments_strided_e4m3_neonfhm_(chunk_ptr, chunk_count, stride_elements, &sum, &sumsq);
            if (start == 0) *sum_ptr = sum, *sumsq_ptr = sumsq;
            else *sum_ptr += sum, *sumsq_ptr += sumsq;
        }
    }
}

NUMKONG_API nk_status_t nk_reduce_moments_e4m3_neonfhm(           //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_reduce_moments_chunked_e4m3_neonfhm_(data_ptr, count, stride, sum_ptr, sumsq_ptr);
    return nk_success_k;
}

NUMKONG_INLINE void nk_reduce_moments_contiguous_e5m2_neonfhm_( //
    nk_e5m2_t const *data_ptr, nk_size_t count,                 //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vreinterpretq_f16_u16(nk_splat_u16x8_neon_(0x3C00));
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

NUMKONG_INLINE void nk_reduce_moments_strided_e5m2_neonfhm_(               //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {

    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vreinterpretq_f16_u16(nk_splat_u16x8_neon_(0x3C00));
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

NUMKONG_INLINE void nk_reduce_moments_chunked_e5m2_neonfhm_(      //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride / sizeof(nk_e5m2_t);
    int aligned = (stride % sizeof(nk_e5m2_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_strided_e5m2_serial_(data_ptr, count, stride, sum_ptr, sumsq_ptr);
    else {
        nk_size_t const chunk_limit = (nk_size_t)5000 * 8;
        for (nk_size_t start = 0; start < count; start += chunk_limit) {
            nk_e5m2_t const *chunk_ptr = data_ptr + start * stride_elements;
            nk_size_t chunk_count = count - start < chunk_limit ? count - start : chunk_limit;
            nk_f32_t sum, sumsq;
            if (stride_elements == 1) nk_reduce_moments_contiguous_e5m2_neonfhm_(chunk_ptr, chunk_count, &sum, &sumsq);
            else nk_reduce_moments_strided_e5m2_neonfhm_(chunk_ptr, chunk_count, stride_elements, &sum, &sumsq);
            if (start == 0) *sum_ptr = sum, *sumsq_ptr = sumsq;
            else *sum_ptr += sum, *sumsq_ptr += sumsq;
        }
    }
}

NUMKONG_API nk_status_t nk_reduce_moments_e5m2_neonfhm(           //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_reduce_moments_chunked_e5m2_neonfhm_(data_ptr, count, stride, sum_ptr, sumsq_ptr);
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

#endif // NUMKONG_TARGET_NEONFHM
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_REDUCE_NEONFHM_H
