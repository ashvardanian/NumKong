/**
 *  @brief ARMv8.6-BF16 implementations for the redesigned reduction API (moments + minmax).
 *  @file include/numkong/reduce/neonbfdot.h
 *  @author Ash Vardanian
 *  @date February 13, 2026
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_NEONBFDOT_H
#define NK_REDUCE_NEONBFDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONBFDOT

#include "numkong/types.h"         // `nk_bf16_t`
#include "numkong/cast/neon.h"     // `nk_e4m3x8_to_f16x8_neon_`
#include "numkong/cast/serial.h"   // `nk_f32_to_bf16_serial`
#include "numkong/reduce/serial.h" // `nk_reduce_moments_bf16_serial`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
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
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16((uint16_t const *)(data_ptr + idx * 2));
            bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(loaded_u16x8x2.val[0]);
            sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
            sumsq_f32x4 = vbfdotq_f32(sumsq_f32x4, data_bf16x8, data_bf16x8);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16((uint16_t const *)(data_ptr + idx * 3));
            bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(loaded_u16x8x3.val[0]);
            sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
            sumsq_f32x4 = vbfdotq_f32(sumsq_f32x4, data_bf16x8, data_bf16x8);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16((uint16_t const *)(data_ptr + idx * 4));
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

/** @brief Convert 8 raw bf16 sign-magnitude u16 to order-preserving comparable i16.
 *  Positive bf16 values (sign=0) are left as-is: they already sort correctly as i16.
 *  Negative bf16 values (sign=1) have their magnitude bits flipped (XOR 0x7FFF)
 *  so that more-negative values map to more-negative i16 values. */
NK_INTERNAL int16x8_t nk_bf16x8_to_comparable_i16x8_neon_(uint16x8_t raw_u16x8) {
    int16x8_t raw_i16x8 = vreinterpretq_s16_u16(raw_u16x8);
    uint16x8_t is_negative_u16x8 = vtstq_u16(raw_u16x8, vdupq_n_u16(0x8000));
    int16x8_t flipped_i16x8 = veorq_s16(raw_i16x8, vdupq_n_s16(0x7FFF));
    return vbslq_s16(is_negative_u16x8, flipped_i16x8, raw_i16x8);
}

/** @brief Convert a comparable i16 value back to raw bf16 u16 bits.
 *  Reverses the transformation from nk_bf16x8_to_comparable_i16x8_neon_. */
NK_INTERNAL nk_u16_t nk_comparable_i16_to_bf16_raw_(nk_i16_t comparable) {
    if (comparable < 0) return (nk_u16_t)(comparable ^ 0x7FFF);
    return (nk_u16_t)comparable;
}

NK_INTERNAL void nk_reduce_minmax_bf16_neonbfdot_contiguous_( //
    nk_bf16_t const *data_ptr, nk_size_t count,               //
    nk_bf16_t *min_value_ptr, nk_size_t *min_index_ptr,       //
    nk_bf16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    int16x8_t min_i16x8 = vdupq_n_s16(NK_I16_MAX), max_i16x8 = vdupq_n_s16(NK_I16_MIN);
    uint16x8_t min_iter_u16x8 = vdupq_n_u16(0), max_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t iter_u16x8 = vdupq_n_u16(0), one_u16x8 = vdupq_n_u16(1);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        uint16x8_t raw_u16x8 = vld1q_u16((uint16_t const *)(data_ptr + idx));
        int16x8_t comparable_i16x8 = nk_bf16x8_to_comparable_i16x8_neon_(raw_u16x8);
        uint16x8_t less_u16x8 = vcltq_s16(comparable_i16x8, min_i16x8);
        uint16x8_t greater_u16x8 = vcgtq_s16(comparable_i16x8, max_i16x8);
        min_i16x8 = vbslq_s16(less_u16x8, comparable_i16x8, min_i16x8);
        max_i16x8 = vbslq_s16(greater_u16x8, comparable_i16x8, max_i16x8);
        min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
        max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
        iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
    }
    // Handle tail with partial load and identity masking
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b16x8_serial_(data_ptr + idx, &tail_vec, remaining);
        int16x8_t comparable_i16x8 = nk_bf16x8_to_comparable_i16x8_neon_(tail_vec.u16x8);
        uint16x8_t lane_indices_u16x8 = vcombine_u16(vreinterpret_u16_u64(vcreate_u64(0x0003000200010000ULL)),
                                                     vreinterpret_u16_u64(vcreate_u64(0x0007000600050004ULL)));
        uint16x8_t valid_u16x8 = vcltq_u16(lane_indices_u16x8, vdupq_n_u16((uint16_t)remaining));
        int16x8_t data_for_min_i16x8 = vbslq_s16(valid_u16x8, comparable_i16x8, vdupq_n_s16(NK_I16_MAX));
        int16x8_t data_for_max_i16x8 = vbslq_s16(valid_u16x8, comparable_i16x8, vdupq_n_s16(NK_I16_MIN));
        uint16x8_t less_u16x8 = vcltq_s16(data_for_min_i16x8, min_i16x8);
        uint16x8_t greater_u16x8 = vcgtq_s16(data_for_max_i16x8, max_i16x8);
        min_i16x8 = vbslq_s16(less_u16x8, data_for_min_i16x8, min_i16x8);
        max_i16x8 = vbslq_s16(greater_u16x8, data_for_max_i16x8, max_i16x8);
        min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
        max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
    }
    // Horizontal reduction
    nk_i16_t min_comparable = vminvq_s16(min_i16x8), max_comparable = vmaxvq_s16(max_i16x8);
    // All-NaN early return: both sentinels unchanged means no valid data was found
    if (min_comparable == NK_I16_MAX && max_comparable == NK_I16_MIN) {
        *(nk_u16_t *)min_value_ptr = nk_comparable_i16_to_bf16_raw_(min_comparable), *min_index_ptr = NK_SIZE_MAX;
        *(nk_u16_t *)max_value_ptr = nk_comparable_i16_to_bf16_raw_(max_comparable), *max_index_ptr = NK_SIZE_MAX;
        return;
    }
    uint16x8_t min_value_match_u16x8 = vceqq_s16(min_i16x8, vdupq_n_s16(min_comparable));
    uint16x8_t masked_min_iter_u16x8 = vbslq_u16(min_value_match_u16x8, min_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_min_cycle = vminvq_u16(masked_min_iter_u16x8);
    uint16x8_t max_value_match_u16x8 = vceqq_s16(max_i16x8, vdupq_n_s16(max_comparable));
    uint16x8_t masked_max_iter_u16x8 = vbslq_u16(max_value_match_u16x8, max_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_max_cycle = vminvq_u16(masked_max_iter_u16x8);
    uint16x8_t lane_indices_u16x8 = vcombine_u16(vreinterpret_u16_u64(vcreate_u64(0x0003000200010000ULL)),
                                                 vreinterpret_u16_u64(vcreate_u64(0x0007000600050004ULL)));
    uint16x8_t min_cycle_match_u16x8 = vceqq_u16(min_iter_u16x8, vdupq_n_u16(earliest_min_cycle));
    uint16x8_t min_both_match_u16x8 = vandq_u16(min_value_match_u16x8, min_cycle_match_u16x8);
    uint16x8_t min_masked_lanes_u16x8 = vbslq_u16(min_both_match_u16x8, lane_indices_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t min_lane_offset = vminvq_u16(min_masked_lanes_u16x8);
    nk_size_t min_idx = (nk_size_t)earliest_min_cycle * 8 + (nk_size_t)min_lane_offset;
    uint16x8_t max_cycle_match_u16x8 = vceqq_u16(max_iter_u16x8, vdupq_n_u16(earliest_max_cycle));
    uint16x8_t max_both_match_u16x8 = vandq_u16(max_value_match_u16x8, max_cycle_match_u16x8);
    uint16x8_t max_masked_lanes_u16x8 = vbslq_u16(max_both_match_u16x8, lane_indices_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t max_lane_offset = vminvq_u16(max_masked_lanes_u16x8);
    nk_size_t max_idx = (nk_size_t)earliest_max_cycle * 8 + (nk_size_t)max_lane_offset;
    // Convert comparable back to bf16 raw bits
    nk_u16_t min_raw = nk_comparable_i16_to_bf16_raw_(min_comparable);
    nk_u16_t max_raw = nk_comparable_i16_to_bf16_raw_(max_comparable);
    *(nk_u16_t *)min_value_ptr = min_raw, *min_index_ptr = min_idx;
    *(nk_u16_t *)max_value_ptr = max_raw, *max_index_ptr = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_bf16_neonbfdot_strided_(                 //
    nk_bf16_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_bf16_t *min_value_ptr, nk_size_t *min_index_ptr,                    //
    nk_bf16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    int16x8_t min_i16x8 = vdupq_n_s16(NK_I16_MAX), max_i16x8 = vdupq_n_s16(NK_I16_MIN);
    uint16x8_t min_iter_u16x8 = vdupq_n_u16(0), max_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t iter_u16x8 = vdupq_n_u16(0), one_u16x8 = vdupq_n_u16(1);
    uint16x8_t lane_indices_u16x8 = vcombine_u16(vreinterpret_u16_u64(vcreate_u64(0x0003000200010000ULL)),
                                                 vreinterpret_u16_u64(vcreate_u64(0x0007000600050004ULL)));
    nk_size_t idx = 0;
    int16x8_t data_for_min_i16x8, data_for_max_i16x8;

nk_reduce_minmax_bf16_neonbfdot_cycle:
    if (stride_elements == 2 && idx + 8 <= count) {
        uint16x8x2_t loaded = vld2q_u16((uint16_t const *)(data_ptr + idx * 2));
        int16x8_t comparable_i16x8 = nk_bf16x8_to_comparable_i16x8_neon_(loaded.val[0]);
        data_for_min_i16x8 = comparable_i16x8;
        data_for_max_i16x8 = comparable_i16x8;
        idx += 8;
    }
    else if (stride_elements == 3 && idx + 8 <= count) {
        uint16x8x3_t loaded = vld3q_u16((uint16_t const *)(data_ptr + idx * 3));
        int16x8_t comparable_i16x8 = nk_bf16x8_to_comparable_i16x8_neon_(loaded.val[0]);
        data_for_min_i16x8 = comparable_i16x8;
        data_for_max_i16x8 = comparable_i16x8;
        idx += 8;
    }
    else if (stride_elements == 4 && idx + 8 <= count) {
        uint16x8x4_t loaded = vld4q_u16((uint16_t const *)(data_ptr + idx * 4));
        int16x8_t comparable_i16x8 = nk_bf16x8_to_comparable_i16x8_neon_(loaded.val[0]);
        data_for_min_i16x8 = comparable_i16x8;
        data_for_max_i16x8 = comparable_i16x8;
        idx += 8;
    }
    else if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_strided_load_b16x8_serial_(data_ptr + idx * stride_elements, stride_elements, &tail_vec, count - idx);
        int16x8_t comparable_i16x8 = nk_bf16x8_to_comparable_i16x8_neon_(tail_vec.u16x8);
        uint16x8_t valid_u16x8 = vcltq_u16(lane_indices_u16x8, vdupq_n_u16((uint16_t)(count - idx)));
        data_for_min_i16x8 = vbslq_s16(valid_u16x8, comparable_i16x8, vdupq_n_s16(NK_I16_MAX));
        data_for_max_i16x8 = vbslq_s16(valid_u16x8, comparable_i16x8, vdupq_n_s16(NK_I16_MIN));
        idx = count;
    }
    else {
        nk_i16_t min_comparable = vminvq_s16(min_i16x8), max_comparable = vmaxvq_s16(max_i16x8);
        if (min_comparable == NK_I16_MAX && max_comparable == NK_I16_MIN) {
            *(nk_u16_t *)min_value_ptr = nk_comparable_i16_to_bf16_raw_(min_comparable), *min_index_ptr = NK_SIZE_MAX;
            *(nk_u16_t *)max_value_ptr = nk_comparable_i16_to_bf16_raw_(max_comparable), *max_index_ptr = NK_SIZE_MAX;
            return;
        }
        uint16x8_t min_value_match_u16x8 = vceqq_s16(min_i16x8, vdupq_n_s16(min_comparable));
        uint16x8_t masked_min_iter_u16x8 = vbslq_u16(min_value_match_u16x8, min_iter_u16x8, vdupq_n_u16(0xFFFF));
        nk_u16_t earliest_min_cycle = vminvq_u16(masked_min_iter_u16x8);
        uint16x8_t max_value_match_u16x8 = vceqq_s16(max_i16x8, vdupq_n_s16(max_comparable));
        uint16x8_t masked_max_iter_u16x8 = vbslq_u16(max_value_match_u16x8, max_iter_u16x8, vdupq_n_u16(0xFFFF));
        nk_u16_t earliest_max_cycle = vminvq_u16(masked_max_iter_u16x8);
        uint16x8_t min_cycle_match_u16x8 = vceqq_u16(min_iter_u16x8, vdupq_n_u16(earliest_min_cycle));
        uint16x8_t min_both_match_u16x8 = vandq_u16(min_value_match_u16x8, min_cycle_match_u16x8);
        uint16x8_t min_masked_lanes_u16x8 = vbslq_u16(min_both_match_u16x8, lane_indices_u16x8, vdupq_n_u16(0xFFFF));
        nk_u16_t min_lane_offset = vminvq_u16(min_masked_lanes_u16x8);
        nk_size_t min_idx = (nk_size_t)earliest_min_cycle * 8 + (nk_size_t)min_lane_offset;
        uint16x8_t max_cycle_match_u16x8 = vceqq_u16(max_iter_u16x8, vdupq_n_u16(earliest_max_cycle));
        uint16x8_t max_both_match_u16x8 = vandq_u16(max_value_match_u16x8, max_cycle_match_u16x8);
        uint16x8_t max_masked_lanes_u16x8 = vbslq_u16(max_both_match_u16x8, lane_indices_u16x8, vdupq_n_u16(0xFFFF));
        nk_u16_t max_lane_offset = vminvq_u16(max_masked_lanes_u16x8);
        nk_size_t max_idx = (nk_size_t)earliest_max_cycle * 8 + (nk_size_t)max_lane_offset;
        nk_u16_t min_raw = nk_comparable_i16_to_bf16_raw_(min_comparable);
        nk_u16_t max_raw = nk_comparable_i16_to_bf16_raw_(max_comparable);
        *(nk_u16_t *)min_value_ptr = min_raw, *min_index_ptr = min_idx;
        *(nk_u16_t *)max_value_ptr = max_raw, *max_index_ptr = max_idx;
        return;
    }

    // Shared update body
    uint16x8_t less_u16x8 = vcltq_s16(data_for_min_i16x8, min_i16x8);
    uint16x8_t greater_u16x8 = vcgtq_s16(data_for_max_i16x8, max_i16x8);
    min_i16x8 = vbslq_s16(less_u16x8, data_for_min_i16x8, min_i16x8);
    max_i16x8 = vbslq_s16(greater_u16x8, data_for_max_i16x8, max_i16x8);
    min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
    max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
    iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
    goto nk_reduce_minmax_bf16_neonbfdot_cycle;
}

NK_PUBLIC void nk_reduce_minmax_bf16_neonbfdot(                         //
    nk_bf16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_bf16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_bf16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (count == 0) {
        *(nk_u16_t *)min_value_ptr = NK_BF16_MAX, *min_index_ptr = NK_SIZE_MAX;
        *(nk_u16_t *)max_value_ptr = NK_BF16_MIN, *max_index_ptr = NK_SIZE_MAX;
    }
    else if (!aligned)
        nk_reduce_minmax_bf16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_bf16_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_bf16_neonbfdot(data_ptr, left_count, stride_bytes, &left_min_value, &left_min_index,
                                        &left_max_value, &left_max_index);
        nk_reduce_minmax_bf16_neonbfdot(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                        &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (nk_bf16_order_serial(right_min_value, left_min_value) < 0)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (nk_bf16_order_serial(right_max_value, left_max_value) > 0)
            *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_bf16_neonbfdot_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                    max_index_ptr);
    else if (stride_elements <= 4)
        nk_reduce_minmax_bf16_neonbfdot_strided_(data_ptr, count, stride_elements, min_value_ptr, min_index_ptr,
                                                 max_value_ptr, max_index_ptr);
    else
        nk_reduce_minmax_bf16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
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

#endif // NK_TARGET_NEONBFDOT
#endif // NK_TARGET_ARM_
#endif // NK_REDUCE_NEONBFDOT_H
