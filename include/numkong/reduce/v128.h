/**
 *  @file include/numkong/reduce/v128.h
 *  @author Ash Vardanian
 *  @date September 12, 2026
 *  @brief SIMD-accelerated reductions for WASM.
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_V128_H
#define NK_REDUCE_V128_H

#if NK_TARGET_V128

#include "numkong/types.h"
#include "numkong/reduce/serial.h"
#include "numkong/cast/serial.h" // `nk_bf16_to_f32_serial`
#include "numkong/cast/v128.h"   // `nk_bf16x4_to_f32x4_v128_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

/** Horizontal sum of 4 floats using shuffle tree. */
NK_HELPER_INLINE nk_f32_t nk_reduce_add_f32x4_v128_(v128_t vec_f32x4) {
    v128_t high_f32x4 = wasm_i32x4_shuffle(vec_f32x4, vec_f32x4, 2, 3, 0, 0);
    v128_t sum1_f32x4 = wasm_f32x4_add(vec_f32x4, high_f32x4);
    v128_t high2_f32x4 = wasm_i32x4_shuffle(sum1_f32x4, sum1_f32x4, 1, 0, 0, 0);
    v128_t sum2_f32x4 = wasm_f32x4_add(sum1_f32x4, high2_f32x4);
    return wasm_f32x4_extract_lane(sum2_f32x4, 0);
}

/** Horizontal maximum of 4 floats using shuffle tree. */
NK_HELPER_INLINE nk_f32_t nk_reduce_max_f32x4_v128_(v128_t vec_f32x4) {
    v128_t high_f32x4 = wasm_i32x4_shuffle(vec_f32x4, vec_f32x4, 2, 3, 0, 0);
    v128_t max1_f32x4 = wasm_f32x4_max(vec_f32x4, high_f32x4);
    v128_t high2_f32x4 = wasm_i32x4_shuffle(max1_f32x4, max1_f32x4, 1, 0, 0, 0);
    v128_t max2_f32x4 = wasm_f32x4_max(max1_f32x4, high2_f32x4);
    return wasm_f32x4_extract_lane(max2_f32x4, 0);
}

/** Horizontal sum of 2 doubles using single shuffle. */
NK_HELPER_INLINE nk_f64_t nk_reduce_add_f64x2_v128_(v128_t vec_f64x2) {
    v128_t high_f64x2 = wasm_i64x2_shuffle(vec_f64x2, vec_f64x2, 1, 0);
    v128_t sum_f64x2 = wasm_f64x2_add(vec_f64x2, high_f64x2);
    return wasm_f64x2_extract_lane(sum_f64x2, 0);
}

/** Horizontal sum of 4 signed 32-bit integers using shuffle tree. */
NK_HELPER_INLINE nk_i32_t nk_reduce_add_i32x4_v128_(v128_t vec_i32x4) {
    v128_t high_i32x4 = wasm_i32x4_shuffle(vec_i32x4, vec_i32x4, 2, 3, 0, 0);
    v128_t sum1_i32x4 = wasm_i32x4_add(vec_i32x4, high_i32x4);
    v128_t high2_i32x4 = wasm_i32x4_shuffle(sum1_i32x4, sum1_i32x4, 1, 0, 0, 0);
    v128_t sum2_i32x4 = wasm_i32x4_add(sum1_i32x4, high2_i32x4);
    return wasm_i32x4_extract_lane(sum2_i32x4, 0);
}

/** Horizontal sum of 4 unsigned 32-bit integers using shuffle tree. */
NK_HELPER_INLINE nk_u32_t nk_reduce_add_u32x4_v128_(v128_t vec_u32x4) {
    v128_t high_u32x4 = wasm_i32x4_shuffle(vec_u32x4, vec_u32x4, 2, 3, 0, 0);
    v128_t sum1_u32x4 = wasm_i32x4_add(vec_u32x4, high_u32x4);
    v128_t high2_u32x4 = wasm_i32x4_shuffle(sum1_u32x4, sum1_u32x4, 1, 0, 0, 0);
    v128_t sum2_u32x4 = wasm_i32x4_add(sum1_u32x4, high2_u32x4);
    return (nk_u32_t)wasm_i32x4_extract_lane(sum2_u32x4, 0);
}

/** Horizontal sum of 16 unsigned 8-bit integers using pairwise widening. */
NK_HELPER_INLINE nk_u32_t nk_reduce_add_u8x16_v128_(v128_t vec_u8x16) {
    v128_t sum_u16x8 = wasm_u16x8_extadd_pairwise_u8x16(vec_u8x16);
    v128_t sum_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(sum_u16x8);
    return nk_reduce_add_u32x4_v128_(sum_u32x4);
}

NK_HELPER_INLINE nk_i64_t nk_reduce_add_i64x2_v128_(v128_t vec_i64x2) {
    v128_t high_i64x2 = wasm_i64x2_shuffle(vec_i64x2, vec_i64x2, 1, 0);
    v128_t sum_i64x2 = wasm_i64x2_add(vec_i64x2, high_i64x2);
    return (nk_i64_t)wasm_i64x2_extract_lane(sum_i64x2, 0);
}

NK_HELPER_INLINE nk_u64_t nk_reduce_add_u64x2_v128_(v128_t vec_u64x2) {
    v128_t high_u64x2 = wasm_i64x2_shuffle(vec_u64x2, vec_u64x2, 1, 0);
    v128_t sum_u64x2 = wasm_i64x2_add(vec_u64x2, high_u64x2);
    return (nk_u64_t)wasm_i64x2_extract_lane(sum_u64x2, 0);
}

NK_HELPER_INLINE nk_i64_t nk_reduce_add_i32x4_to_i64_v128_(v128_t vec_i32x4) {
    v128_t low_i64x2 = wasm_i64x2_extend_low_i32x4(vec_i32x4);
    v128_t high_i64x2 = wasm_i64x2_extend_high_i32x4(vec_i32x4);
    v128_t sum_i64x2 = wasm_i64x2_add(low_i64x2, high_i64x2);
    return nk_reduce_add_i64x2_v128_(sum_i64x2);
}

NK_HELPER_INLINE nk_u64_t nk_reduce_add_u32x4_to_u64_v128_(v128_t vec_u32x4) {
    v128_t low_u64x2 = wasm_u64x2_extend_low_u32x4(vec_u32x4);
    v128_t high_u64x2 = wasm_u64x2_extend_high_u32x4(vec_u32x4);
    v128_t sum_u64x2 = wasm_i64x2_add(low_u64x2, high_u64x2);
    return nk_reduce_add_u64x2_v128_(sum_u64x2);
}

NK_HELPER_INLINE v128_t nk_u64_sadd_epi64_v128_(v128_t a_u64x2, v128_t b_u64x2) {
    v128_t result_u64x2 = wasm_i64x2_add(a_u64x2, b_u64x2);
    v128_t sign_bit_i64x2 = wasm_i64x2_splat((nk_i64_t)0x8000000000000000LL);
    v128_t a_biased_i64x2 = wasm_v128_xor(a_u64x2, sign_bit_i64x2);
    v128_t result_biased_i64x2 = wasm_v128_xor(result_u64x2, sign_bit_i64x2);
    v128_t overflow_u64x2 = wasm_i64x2_gt(a_biased_i64x2, result_biased_i64x2);
    return wasm_v128_or(result_u64x2, overflow_u64x2);
}

NK_HELPER_INLINE nk_u64_t nk_reduce_sadd_u64x2_v128_(v128_t v_u64x2) {
    v128_t swapped_u64x2 = wasm_i64x2_shuffle(v_u64x2, v_u64x2, 1, 0);
    v128_t sum_u64x2 = wasm_i64x2_add(v_u64x2, swapped_u64x2);
    v128_t sign_bit_i64x2 = wasm_i64x2_splat((nk_i64_t)0x8000000000000000LL);
    v128_t v_biased_i64x2 = wasm_v128_xor(v_u64x2, sign_bit_i64x2);
    v128_t sum_biased_i64x2 = wasm_v128_xor(sum_u64x2, sign_bit_i64x2);
    v128_t overflow_u64x2 = wasm_i64x2_gt(v_biased_i64x2, sum_biased_i64x2);
    sum_u64x2 = wasm_v128_or(sum_u64x2, overflow_u64x2);
    return (nk_u64_t)wasm_i64x2_extract_lane(sum_u64x2, 0);
}

NK_HELPER_INLINE void nk_reduce_moments_f64_v128_contiguous_( //
    nk_f64_t const *data, nk_size_t count,                    //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    v128_t sum_f64x2 = wasm_f64x2_splat(0);
    v128_t sum_comp_f64x2 = wasm_f64x2_splat(0);
    v128_t sumsq_f64x2 = wasm_f64x2_splat(0);
    v128_t sumsq_comp_f64x2 = wasm_f64x2_splat(0);
    nk_size_t index = 0;
    for (; index + 2 <= count; index += 2) {
        v128_t value_f64x2 = wasm_v128_load(data + index);
        v128_t tentative_f64x2 = wasm_f64x2_add(sum_f64x2, value_f64x2);
        v128_t round_f64x2 = wasm_f64x2_sub(tentative_f64x2, sum_f64x2);
        v128_t corr_f64x2 = wasm_f64x2_add(wasm_f64x2_sub(sum_f64x2, wasm_f64x2_sub(tentative_f64x2, round_f64x2)),
                                           wasm_f64x2_sub(value_f64x2, round_f64x2));
        sum_comp_f64x2 = wasm_f64x2_add(sum_comp_f64x2, corr_f64x2);
        sum_f64x2 = tentative_f64x2;
        v128_t sq_f64x2 = wasm_f64x2_mul(value_f64x2, value_f64x2);
        v128_t tentative_sq_f64x2 = wasm_f64x2_add(sumsq_f64x2, sq_f64x2);
        v128_t round_sq_f64x2 = wasm_f64x2_sub(tentative_sq_f64x2, sumsq_f64x2);
        v128_t corr_sq_f64x2 = wasm_f64x2_add(
            wasm_f64x2_sub(sumsq_f64x2, wasm_f64x2_sub(tentative_sq_f64x2, round_sq_f64x2)),
            wasm_f64x2_sub(sq_f64x2, round_sq_f64x2));
        sumsq_comp_f64x2 = wasm_f64x2_add(sumsq_comp_f64x2, corr_sq_f64x2);
        sumsq_f64x2 = tentative_sq_f64x2;
    }
    nk_f64_t sum = nk_reduce_add_f64x2_v128_(wasm_f64x2_add(sum_f64x2, sum_comp_f64x2));
    nk_f64_t sumsq = nk_reduce_add_f64x2_v128_(wasm_f64x2_add(sumsq_f64x2, sumsq_comp_f64x2));
    for (; index < count; ++index) {
        nk_f64_t value = data[index];
        sum += value;
        sumsq += value * value;
    }
    *sum_ptr = sum;
    *sumsq_ptr = sumsq;
}

NK_API_COMPTIME void nk_reduce_moments_f64_v128(                   //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f64_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 2) {
        nk_size_t left_count = count / 2;
        nk_f64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f64_v128(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f64_v128(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                   &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f64_v128_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_f64_serial(data, count, stride_bytes, sum, sumsq);
}

NK_HELPER_INLINE void nk_reduce_moments_bf16_v128_contiguous_( //
    nk_bf16_t const *data, nk_size_t count,                    //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0);
    v128_t sumsq_f32x4 = wasm_f32x4_splat(0);
    nk_size_t index = 0;
    for (; index + 4 <= count; index += 4) {
        nk_b64_vec_t raw;
        raw.u64 = *(nk_u64_t const *)(data + index);
        v128_t data_f32x4 = nk_bf16x4_to_f32x4_v128_(raw).v128;
        sum_f32x4 = wasm_f32x4_add(sum_f32x4, data_f32x4);
        sumsq_f32x4 = wasm_f32x4_add(sumsq_f32x4, wasm_f32x4_mul(data_f32x4, data_f32x4));
    }
    nk_f32_t sum = nk_reduce_add_f32x4_v128_(sum_f32x4);
    nk_f32_t sumsq = nk_reduce_add_f32x4_v128_(sumsq_f32x4);
    for (; index < count; ++index) {
        nk_f32_t value;
        nk_bf16_to_f32_serial(data + index, &value);
        sum += value, sumsq += value * value;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_API_COMPTIME void nk_reduce_moments_bf16_v128(                   //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_bf16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 4) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_bf16_v128(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_bf16_v128(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                    &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_bf16_v128_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_bf16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_HELPER_INLINE void nk_reduce_moments_i8_v128_contiguous_( //
    nk_i8_t const *data, nk_size_t count,                    //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    v128_t sum_i32x4 = wasm_i32x4_splat(0);
    v128_t sumsq_u64x2 = wasm_i64x2_splat(0);
    nk_size_t index = 0;
    for (; index + 16 <= count; index += 16) {
        v128_t data_i8x16 = wasm_v128_load(data + index);
        v128_t pairwise_i16x8 = wasm_i16x8_extadd_pairwise_i8x16(data_i8x16);
        v128_t pairwise_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(pairwise_i16x8);
        sum_i32x4 = wasm_i32x4_add(sum_i32x4, pairwise_i32x4);
        v128_t sq_low_i16x8 = wasm_i16x8_extmul_low_i8x16(data_i8x16, data_i8x16);
        v128_t sq_high_i16x8 = wasm_i16x8_extmul_high_i8x16(data_i8x16, data_i8x16);
        v128_t sq_u32x4 = wasm_i32x4_add(wasm_u32x4_extadd_pairwise_u16x8(sq_low_i16x8),
                                         wasm_u32x4_extadd_pairwise_u16x8(sq_high_i16x8));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_low_u32x4(sq_u32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_high_u32x4(sq_u32x4));
    }
    nk_i64_t sum = nk_reduce_add_i32x4_v128_(sum_i32x4);
    nk_u64_t sumsq = nk_reduce_add_u64x2_v128_(sumsq_u64x2);
    for (; index < count; ++index) {
        nk_i64_t value = (nk_i64_t)data[index];
        sum += value, sumsq += (nk_u64_t)(value * value);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_API_COMPTIME void nk_reduce_moments_i8_v128(                   //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i8_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum, right_sum;
        nk_u64_t left_sumsq, right_sumsq;
        nk_reduce_moments_i8_v128(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i8_v128(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                  &right_sumsq);
        *sum = nk_i64_saturating_add_serial(left_sum, right_sum);
        *sumsq = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i8_v128_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_i8_serial(data, count, stride_bytes, sum, sumsq);
}

NK_HELPER_INLINE void nk_reduce_moments_u8_v128_contiguous_( //
    nk_u8_t const *data, nk_size_t count,                    //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    v128_t sum_u32x4 = wasm_i32x4_splat(0);
    v128_t sumsq_u64x2 = wasm_i64x2_splat(0);
    nk_size_t index = 0;
    for (; index + 16 <= count; index += 16) {
        v128_t data_u8x16 = wasm_v128_load(data + index);
        v128_t pairwise_u16x8 = wasm_u16x8_extadd_pairwise_u8x16(data_u8x16);
        v128_t pairwise_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(pairwise_u16x8);
        sum_u32x4 = wasm_i32x4_add(sum_u32x4, pairwise_u32x4);
        v128_t sq_low_u16x8 = wasm_u16x8_extmul_low_u8x16(data_u8x16, data_u8x16);
        v128_t sq_high_u16x8 = wasm_u16x8_extmul_high_u8x16(data_u8x16, data_u8x16);
        v128_t sq_u32x4 = wasm_i32x4_add(wasm_u32x4_extadd_pairwise_u16x8(sq_low_u16x8),
                                         wasm_u32x4_extadd_pairwise_u16x8(sq_high_u16x8));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_low_u32x4(sq_u32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_high_u32x4(sq_u32x4));
    }
    nk_u64_t sum = nk_reduce_add_u32x4_v128_(sum_u32x4);
    nk_u64_t sumsq = nk_reduce_add_u64x2_v128_(sumsq_u64x2);
    for (; index < count; ++index) {
        nk_u64_t value = (nk_u64_t)data[index];
        sum += value, sumsq += value * value;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_API_COMPTIME void nk_reduce_moments_u8_v128(                   //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u8_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u8_v128(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u8_v128(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                  &right_sumsq);
        *sum = nk_u64_saturating_add_serial(left_sum, right_sum);
        *sumsq = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u8_v128_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_u8_serial(data, count, stride_bytes, sum, sumsq);
}

NK_HELPER_INLINE void nk_reduce_moments_i16_v128_contiguous_( //
    nk_i16_t const *data, nk_size_t count,                    //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    v128_t sum_i64x2 = wasm_i64x2_splat(0);
    v128_t sumsq_u64x2 = wasm_i64x2_splat(0);
    nk_size_t index = 0;
    for (; index + 8 <= count; index += 8) {
        v128_t data_i16x8 = wasm_v128_load(data + index);
        v128_t pairwise_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(data_i16x8);
        sum_i64x2 = wasm_i64x2_add(sum_i64x2, wasm_i64x2_extend_low_i32x4(pairwise_i32x4));
        sum_i64x2 = wasm_i64x2_add(sum_i64x2, wasm_i64x2_extend_high_i32x4(pairwise_i32x4));
        v128_t sq_low_i32x4 = wasm_i32x4_extmul_low_i16x8(data_i16x8, data_i16x8);
        v128_t sq_high_i32x4 = wasm_i32x4_extmul_high_i16x8(data_i16x8, data_i16x8);
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_low_u32x4(sq_low_i32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_high_u32x4(sq_low_i32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_low_u32x4(sq_high_i32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_high_u32x4(sq_high_i32x4));
    }
    nk_i64_t sum = nk_reduce_add_i64x2_v128_(sum_i64x2);
    nk_u64_t sumsq = nk_reduce_add_u64x2_v128_(sumsq_u64x2);
    for (; index < count; ++index) {
        nk_i64_t value = (nk_i64_t)data[index];
        sum += value, sumsq += (nk_u64_t)(value * value);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_API_COMPTIME void nk_reduce_moments_i16_v128(                   //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum, right_sum;
        nk_u64_t left_sumsq, right_sumsq;
        nk_reduce_moments_i16_v128(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i16_v128(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                   &right_sumsq);
        *sum = nk_i64_saturating_add_serial(left_sum, right_sum);
        *sumsq = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i16_v128_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_i16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_HELPER_INLINE void nk_reduce_moments_u16_v128_contiguous_( //
    nk_u16_t const *data, nk_size_t count,                    //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    v128_t sum_u64x2 = wasm_i64x2_splat(0);
    v128_t sumsq_u64x2 = wasm_i64x2_splat(0);
    nk_size_t index = 0;
    for (; index + 8 <= count; index += 8) {
        v128_t data_u16x8 = wasm_v128_load(data + index);
        v128_t pairwise_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(data_u16x8);
        sum_u64x2 = wasm_i64x2_add(sum_u64x2, wasm_u64x2_extend_low_u32x4(pairwise_u32x4));
        sum_u64x2 = wasm_i64x2_add(sum_u64x2, wasm_u64x2_extend_high_u32x4(pairwise_u32x4));
        v128_t sq_low_u32x4 = wasm_u32x4_extmul_low_u16x8(data_u16x8, data_u16x8);
        v128_t sq_high_u32x4 = wasm_u32x4_extmul_high_u16x8(data_u16x8, data_u16x8);
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_low_u32x4(sq_low_u32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_high_u32x4(sq_low_u32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_low_u32x4(sq_high_u32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_high_u32x4(sq_high_u32x4));
    }
    nk_u64_t sum = nk_reduce_add_u64x2_v128_(sum_u64x2);
    nk_u64_t sumsq = nk_reduce_add_u64x2_v128_(sumsq_u64x2);
    for (; index < count; ++index) {
        nk_u64_t value = (nk_u64_t)data[index];
        sum += value, sumsq += value * value;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_API_COMPTIME void nk_reduce_moments_u16_v128(                   //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u16_v128(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u16_v128(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                   &right_sumsq);
        *sum = nk_u64_saturating_add_serial(left_sum, right_sum);
        *sumsq = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u16_v128_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_u16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_HELPER_INLINE void nk_reduce_moments_i32_v128_contiguous_( //
    nk_i32_t const *data, nk_size_t count,                    //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    v128_t sum_low_u64x2 = wasm_i64x2_splat(0);
    v128_t sum_high_i64x2 = wasm_i64x2_splat(0);
    v128_t sumsq_u64x2 = wasm_i64x2_splat(0);
    v128_t sumsq_overflow_u64x2 = wasm_i64x2_splat(0);
    v128_t sign_bit_i64x2 = wasm_i64x2_splat((nk_i64_t)0x8000000000000000LL);
    nk_size_t index = 0;
    for (; index + 4 <= count; index += 4) {
        v128_t data_i32x4 = wasm_v128_load(data + index);
        v128_t data_low_i64x2 = wasm_i64x2_extend_low_i32x4(data_i32x4);
        v128_t before_u64x2 = sum_low_u64x2;
        sum_low_u64x2 = wasm_i64x2_add(sum_low_u64x2, data_low_i64x2);
        v128_t result_biased_i64x2 = wasm_v128_xor(sum_low_u64x2, sign_bit_i64x2);
        v128_t before_biased_i64x2 = wasm_v128_xor(before_u64x2, sign_bit_i64x2);
        v128_t carry_u64x2 = wasm_i64x2_gt(before_biased_i64x2, result_biased_i64x2);
        sum_high_i64x2 = wasm_i64x2_sub(sum_high_i64x2, carry_u64x2);
        sum_high_i64x2 = wasm_i64x2_add(sum_high_i64x2, wasm_i64x2_shr(data_low_i64x2, 63));
        v128_t data_high_i64x2 = wasm_i64x2_extend_high_i32x4(data_i32x4);
        before_u64x2 = sum_low_u64x2;
        sum_low_u64x2 = wasm_i64x2_add(sum_low_u64x2, data_high_i64x2);
        result_biased_i64x2 = wasm_v128_xor(sum_low_u64x2, sign_bit_i64x2);
        before_biased_i64x2 = wasm_v128_xor(before_u64x2, sign_bit_i64x2);
        carry_u64x2 = wasm_i64x2_gt(before_biased_i64x2, result_biased_i64x2);
        sum_high_i64x2 = wasm_i64x2_sub(sum_high_i64x2, carry_u64x2);
        sum_high_i64x2 = wasm_i64x2_add(sum_high_i64x2, wasm_i64x2_shr(data_high_i64x2, 63));
        v128_t sq_low_i64x2 = wasm_i64x2_extmul_low_i32x4(data_i32x4, data_i32x4);
        v128_t sq_high_i64x2 = wasm_i64x2_extmul_high_i32x4(data_i32x4, data_i32x4);
        v128_t sq_before_u64x2 = sumsq_u64x2;
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, sq_low_i64x2);
        sumsq_overflow_u64x2 = wasm_v128_or(
            sumsq_overflow_u64x2,
            wasm_i64x2_gt(wasm_v128_xor(sq_before_u64x2, sign_bit_i64x2), wasm_v128_xor(sumsq_u64x2, sign_bit_i64x2)));
        sq_before_u64x2 = sumsq_u64x2;
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, sq_high_i64x2);
        sumsq_overflow_u64x2 = wasm_v128_or(
            sumsq_overflow_u64x2,
            wasm_i64x2_gt(wasm_v128_xor(sq_before_u64x2, sign_bit_i64x2), wasm_v128_xor(sumsq_u64x2, sign_bit_i64x2)));
    }
    int sumsq_overflow = (int)(wasm_i64x2_extract_lane(sumsq_overflow_u64x2, 0) |
                               wasm_i64x2_extract_lane(sumsq_overflow_u64x2, 1));
    nk_u64_t sumsq = sumsq_overflow ? NK_U64_MAX : nk_reduce_sadd_u64x2_v128_(sumsq_u64x2);
    nk_b128_vec_t lower_vec, upper_vec;
    lower_vec.v128 = sum_low_u64x2;
    upper_vec.v128 = sum_high_i64x2;
    nk_u64_t sum_low = 0;
    nk_i64_t sum_high = 0;
    nk_u64_t sum_before = sum_low;
    sum_low += lower_vec.u64s[0], sum_high += (sum_low < sum_before) + upper_vec.i64s[0];
    sum_before = sum_low;
    sum_low += lower_vec.u64s[1], sum_high += (sum_low < sum_before) + upper_vec.i64s[1];
    for (; index < count; ++index) {
        nk_i64_t value = (nk_i64_t)data[index];
        sum_before = sum_low;
        sum_low += (nk_u64_t)value;
        if (sum_low < sum_before) sum_high++;
        sum_high += (value >> 63);
        nk_u64_t product = (nk_u64_t)(value * value);
        sumsq = nk_u64_saturating_add_serial(sumsq, product);
    }
    nk_i64_t sum_low_signed = (nk_i64_t)sum_low;
    if (sum_high == (sum_low_signed >> 63)) *sum_ptr = sum_low_signed;
    else if (sum_high >= 0) *sum_ptr = NK_I64_MAX;
    else *sum_ptr = NK_I64_MIN;
    *sumsq_ptr = sumsq;
}

NK_API_COMPTIME void nk_reduce_moments_i32_v128(                   //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i32_serial(data, count, stride_bytes, sum, sumsq);
    else if (stride_elements == 1) nk_reduce_moments_i32_v128_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_i32_serial(data, count, stride_bytes, sum, sumsq);
}

NK_HELPER_INLINE void nk_reduce_moments_u32_v128_contiguous_( //
    nk_u32_t const *data, nk_size_t count,                    //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    v128_t sum_u64x2 = wasm_i64x2_splat(0);
    v128_t sumsq_u64x2 = wasm_i64x2_splat(0);
    nk_size_t index = 0;
    for (; index + 4 <= count; index += 4) {
        v128_t data_u32x4 = wasm_v128_load(data + index);
        sum_u64x2 = wasm_i64x2_add(sum_u64x2, wasm_u64x2_extend_low_u32x4(data_u32x4));
        sum_u64x2 = wasm_i64x2_add(sum_u64x2, wasm_u64x2_extend_high_u32x4(data_u32x4));
        v128_t sq_low_u64x2 = wasm_u64x2_extmul_low_u32x4(data_u32x4, data_u32x4);
        v128_t sq_high_u64x2 = wasm_u64x2_extmul_high_u32x4(data_u32x4, data_u32x4);
        sumsq_u64x2 = nk_u64_sadd_epi64_v128_(sumsq_u64x2, sq_low_u64x2);
        sumsq_u64x2 = nk_u64_sadd_epi64_v128_(sumsq_u64x2, sq_high_u64x2);
    }
    nk_u64_t sum = nk_reduce_add_u64x2_v128_(sum_u64x2);
    nk_u64_t sumsq = nk_reduce_sadd_u64x2_v128_(sumsq_u64x2);
    for (; index < count; ++index) {
        nk_u64_t value = (nk_u64_t)data[index];
        sum += value;
        nk_u64_t product = value * value;
        sumsq = nk_u64_saturating_add_serial(sumsq, product);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_API_COMPTIME void nk_reduce_moments_u32_v128(                   //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u32_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 4) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u32_v128(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u32_v128(data + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                   &right_sumsq);
        *sum = nk_u64_saturating_add_serial(left_sum, right_sum);
        *sumsq = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u32_v128_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_u32_serial(data, count, stride_bytes, sum, sumsq);
}

NK_HELPER_INLINE nk_u8_t nk_comparable_to_fp8_v128_(nk_u8_t comparable) {
    if (comparable >= 0x80) return comparable ^ 0x80;
    else return ~comparable;
}

NK_HELPER_INLINE nk_u8_t nk_comparable_to_fp6_v128_(nk_u8_t comparable) {
    if (comparable >= 0x20) return comparable ^ 0x20;
    else return (0x1F - comparable) | 0x20;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128
#endif // NK_REDUCE_V128_H
