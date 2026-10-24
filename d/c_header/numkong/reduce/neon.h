/**
 *  @brief Base NEON (ARMv8-A) implementations for the redesigned reduction API (moments + minmax).
 *  @file include/numkong/reduce/neon.h
 *  @author Ash Vardanian
 *  @date February 13, 2026
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_NEON_H
#define NK_REDUCE_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/types.h"       // `nk_size_t`
#include "numkong/cast/neon.h"   // `nk_e4m3x16_to_f16x8x2_neon_`
#include "numkong/cast/serial.h" // `nk_e4m3_to_f16_serial`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

NK_INTERNAL nk_u64_t nk_reduce_sadd_u64x2_neon_(uint64x2_t v) {
    uint64x2_t swapped_u64x2 = vextq_u64(v, v, 1);
    return vgetq_lane_u64(vqaddq_u64(v, swapped_u64x2), 0);
}

/** @brief Saturating square of each i64 lane → u64. If |a| >= 2^32, a² overflows u64 → saturate. */
NK_INTERNAL uint64x2_t nk_i64_smul_sq_i64x2_neon_(int64x2_t val) {
    uint64x2_t absolute_u64x2 = vreinterpretq_u64_s64(vabsq_s64(val));
    uint32x2_t low_halves_u32x2 = vmovn_u64(absolute_u64x2);
    uint64x2_t high_bits_u64x2 = vshrq_n_u64(absolute_u64x2, 32);
    uint64x2_t low_squared_u64x2 = vmull_u32(low_halves_u32x2, low_halves_u32x2);
    uint64x2_t is_small_u64x2 = vceqq_u64(high_bits_u64x2, vdupq_n_u64(0));
    return vbslq_u64(is_small_u64x2, low_squared_u64x2, vdupq_n_u64(NK_U64_MAX));
}

/** @brief Saturating square of each u64 lane → u64. If a >= 2^32, a² overflows u64 → saturate. */
NK_INTERNAL uint64x2_t nk_u64_smul_sq_u64x2_neon_(uint64x2_t val) {
    uint32x2_t low_halves_u32x2 = vmovn_u64(val);
    uint64x2_t high_bits_u64x2 = vshrq_n_u64(val, 32);
    uint64x2_t low_squared_u64x2 = vmull_u32(low_halves_u32x2, low_halves_u32x2);
    uint64x2_t is_small_u64x2 = vceqq_u64(high_bits_u64x2, vdupq_n_u64(0));
    return vbslq_u64(is_small_u64x2, low_squared_u64x2, vdupq_n_u64(NK_U64_MAX));
}

NK_INTERNAL void nk_reduce_moments_f32_neon_contiguous_( //
    nk_f32_t const *data_ptr, nk_size_t count,           //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    float64x2_t sum_f64x2 = vdupq_n_f64(0), sumsq_f64x2 = vdupq_n_f64(0);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        float32x4_t data_f32x4 = vld1q_f32(data_ptr + idx);
        float64x2_t data_low_f64x2 = vcvt_f64_f32(vget_low_f32(data_f32x4));
        float64x2_t data_high_f64x2 = vcvt_f64_f32(vget_high_f32(data_f32x4));
        sum_f64x2 = vaddq_f64(sum_f64x2, data_low_f64x2);
        sum_f64x2 = vaddq_f64(sum_f64x2, data_high_f64x2);
        sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, data_low_f64x2, data_low_f64x2);
        sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, data_high_f64x2, data_high_f64x2);
    }
    nk_f64_t sum = vaddvq_f64(sum_f64x2), sumsq = vaddvq_f64(sumsq_f64x2);
    for (; idx < count; ++idx) {
        nk_f64_t value = (nk_f64_t)data_ptr[idx];
        sum += value, sumsq += value * value;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_f32_neon_strided_(                     //
    nk_f32_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    float64x2_t sum_f64x2 = vdupq_n_f64(0), sumsq_f64x2 = vdupq_n_f64(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            float32x4x2_t loaded_f32x4x2 = vld2q_f32(data_ptr + idx * 2);
            float64x2_t data_low_f64x2 = vcvt_f64_f32(vget_low_f32(loaded_f32x4x2.val[0]));
            float64x2_t data_high_f64x2 = vcvt_f64_f32(vget_high_f32(loaded_f32x4x2.val[0]));
            sum_f64x2 = vaddq_f64(sum_f64x2, data_low_f64x2);
            sum_f64x2 = vaddq_f64(sum_f64x2, data_high_f64x2);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, data_low_f64x2, data_low_f64x2);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, data_high_f64x2, data_high_f64x2);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            float32x4x3_t loaded_f32x4x3 = vld3q_f32(data_ptr + idx * 3);
            float64x2_t data_low_f64x2 = vcvt_f64_f32(vget_low_f32(loaded_f32x4x3.val[0]));
            float64x2_t data_high_f64x2 = vcvt_f64_f32(vget_high_f32(loaded_f32x4x3.val[0]));
            sum_f64x2 = vaddq_f64(sum_f64x2, data_low_f64x2);
            sum_f64x2 = vaddq_f64(sum_f64x2, data_high_f64x2);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, data_low_f64x2, data_low_f64x2);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, data_high_f64x2, data_high_f64x2);
        }
    }
    else {
        for (; idx + 4 <= count; idx += 4) {
            float32x4x4_t loaded_f32x4x4 = vld4q_f32(data_ptr + idx * 4);
            float64x2_t data_low_f64x2 = vcvt_f64_f32(vget_low_f32(loaded_f32x4x4.val[0]));
            float64x2_t data_high_f64x2 = vcvt_f64_f32(vget_high_f32(loaded_f32x4x4.val[0]));
            sum_f64x2 = vaddq_f64(sum_f64x2, data_low_f64x2);
            sum_f64x2 = vaddq_f64(sum_f64x2, data_high_f64x2);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, data_low_f64x2, data_low_f64x2);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, data_high_f64x2, data_high_f64x2);
        }
    }
    nk_f64_t sum = vaddvq_f64(sum_f64x2), sumsq = vaddvq_f64(sumsq_f64x2);
    nk_f32_t const *current_ptr = data_ptr + idx * stride_elements;
    for (; idx < count; ++idx, current_ptr += stride_elements) {
        nk_f64_t value = (nk_f64_t)(*current_ptr);
        sum += value, sumsq += value * value;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_f32_neon(                             //
    nk_f32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_f32_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 4) {
        nk_size_t left_count = count / 2;
        nk_f64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f32_neon(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f32_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                   &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum, *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f32_neon_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_f32_neon_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_f32_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_f32_neon_contiguous_( //
    nk_f32_t const *data_ptr, nk_size_t count,          //
    nk_f32_t *min_value_ptr, nk_size_t *min_index_ptr,  //
    nk_f32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    float32x4_t min_f32x4 = vdupq_n_f32(NK_F32_MAX), max_f32x4 = vdupq_n_f32(NK_F32_MIN);
    uint32x4_t min_iter_u32x4 = vdupq_n_u32(0), max_iter_u32x4 = vdupq_n_u32(0);
    uint32x4_t iter_u32x4 = vdupq_n_u32(0), one_u32x4 = vdupq_n_u32(1);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        float32x4_t data_f32x4 = vld1q_f32(data_ptr + idx);
        uint32x4_t less_u32x4 = vcltq_f32(data_f32x4, min_f32x4);
        uint32x4_t greater_u32x4 = vcgtq_f32(data_f32x4, max_f32x4);
        min_f32x4 = vbslq_f32(less_u32x4, data_f32x4, min_f32x4);
        max_f32x4 = vbslq_f32(greater_u32x4, data_f32x4, max_f32x4);
        min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
        max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
        iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b32x4_serial_(data_ptr + idx, &tail_vec, remaining);
        uint32x4_t lane_u32x4 = vcombine_u32(vreinterpret_u32_u64(vcreate_u64(0x0000000100000000ULL)),
                                             vreinterpret_u32_u64(vcreate_u64(0x0000000300000002ULL)));
        uint32x4_t valid_u32x4 = vcltq_u32(lane_u32x4, vdupq_n_u32((uint32_t)remaining));
        float32x4_t data_min_f32x4 = vbslq_f32(valid_u32x4, tail_vec.f32x4, min_f32x4);
        float32x4_t data_max_f32x4 = vbslq_f32(valid_u32x4, tail_vec.f32x4, max_f32x4);
        uint32x4_t less_u32x4 = vcltq_f32(data_min_f32x4, min_f32x4);
        uint32x4_t greater_u32x4 = vcgtq_f32(data_max_f32x4, max_f32x4);
        min_f32x4 = vbslq_f32(less_u32x4, data_min_f32x4, min_f32x4);
        max_f32x4 = vbslq_f32(greater_u32x4, data_max_f32x4, max_f32x4);
        min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
        max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
    }
    nk_f32_t min_value = vminvq_f32(min_f32x4), max_value = vmaxvq_f32(max_f32x4);

    // All-NaN / sentinel check: sentinels remain unchanged when all data is NaN.
    if (min_value == NK_F32_MAX && max_value == NK_F32_MIN) {
        *min_value_ptr = NK_F32_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F32_MIN,
        *max_index_ptr = NK_SIZE_MAX;
        return;
    }

    uint32x4_t min_value_match_u32x4 = vceqq_f32(min_f32x4, vdupq_n_f32(min_value));
    uint32x4_t masked_min_iter_u32x4 = vbslq_u32(min_value_match_u32x4, min_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_min_cycle = vminvq_u32(masked_min_iter_u32x4);
    uint32x4_t max_value_match_u32x4 = vceqq_f32(max_f32x4, vdupq_n_f32(max_value));
    uint32x4_t masked_max_iter_u32x4 = vbslq_u32(max_value_match_u32x4, max_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_max_cycle = vminvq_u32(masked_max_iter_u32x4);
    uint32x4_t lane_indices_u32x4 = vcombine_u32(vreinterpret_u32_u64(vcreate_u64(0x0000000100000000ULL)),
                                                 vreinterpret_u32_u64(vcreate_u64(0x0000000300000002ULL)));
    uint32x4_t min_cycle_match_u32x4 = vceqq_u32(min_iter_u32x4, vdupq_n_u32(earliest_min_cycle));
    uint32x4_t min_both_match_u32x4 = vandq_u32(min_value_match_u32x4, min_cycle_match_u32x4);
    uint32x4_t min_masked_lanes_u32x4 = vbslq_u32(min_both_match_u32x4, lane_indices_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t min_lane_offset = vminvq_u32(min_masked_lanes_u32x4);
    nk_size_t min_idx = (nk_size_t)earliest_min_cycle * 4 + (nk_size_t)min_lane_offset;
    uint32x4_t max_cycle_match_u32x4 = vceqq_u32(max_iter_u32x4, vdupq_n_u32(earliest_max_cycle));
    uint32x4_t max_both_match_u32x4 = vandq_u32(max_value_match_u32x4, max_cycle_match_u32x4);
    uint32x4_t max_masked_lanes_u32x4 = vbslq_u32(max_both_match_u32x4, lane_indices_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t max_lane_offset = vminvq_u32(max_masked_lanes_u32x4);
    nk_size_t max_idx = (nk_size_t)earliest_max_cycle * 4 + (nk_size_t)max_lane_offset;
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_f32_neon_strided_(                      //
    nk_f32_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value_ptr, nk_size_t *min_index_ptr,                    //
    nk_f32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    float32x4_t min_f32x4 = vdupq_n_f32(NK_F32_MAX), max_f32x4 = vdupq_n_f32(NK_F32_MIN);
    uint32x4_t min_iter_u32x4 = vdupq_n_u32(0), max_iter_u32x4 = vdupq_n_u32(0);
    uint32x4_t iter_u32x4 = vdupq_n_u32(0), one_u32x4 = vdupq_n_u32(1);
    uint32x4_t lane_indices_u32x4 = vcombine_u32(vreinterpret_u32_u64(vcreate_u64(0x0000000100000000ULL)),
                                                 vreinterpret_u32_u64(vcreate_u64(0x0000000300000002ULL)));
    nk_size_t idx = 0;
    float32x4_t data_for_min_f32x4, data_for_max_f32x4;

nk_reduce_minmax_f32_neon_cycle:
    if (stride_elements == 2 && idx + 4 <= count) {
        float32x4x2_t loaded = vld2q_f32(data_ptr + idx * 2);
        data_for_min_f32x4 = loaded.val[0];
        data_for_max_f32x4 = loaded.val[0];
        idx += 4;
    }
    else if (stride_elements == 3 && idx + 4 <= count) {
        float32x4x3_t loaded = vld3q_f32(data_ptr + idx * 3);
        data_for_min_f32x4 = loaded.val[0];
        data_for_max_f32x4 = loaded.val[0];
        idx += 4;
    }
    else if (stride_elements == 4 && idx + 4 <= count) {
        float32x4x4_t loaded = vld4q_f32(data_ptr + idx * 4);
        data_for_min_f32x4 = loaded.val[0];
        data_for_max_f32x4 = loaded.val[0];
        idx += 4;
    }
    else if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_strided_load_b32x4_serial_(data_ptr + idx * stride_elements, stride_elements, &tail_vec, count - idx);
        uint32x4_t valid_u32x4 = vcltq_u32(lane_indices_u32x4, vdupq_n_u32((uint32_t)(count - idx)));
        data_for_min_f32x4 = vbslq_f32(valid_u32x4, tail_vec.f32x4, min_f32x4);
        data_for_max_f32x4 = vbslq_f32(valid_u32x4, tail_vec.f32x4, max_f32x4);
        idx = count;
    }
    else {
        nk_f32_t min_value = vminvq_f32(min_f32x4), max_value = vmaxvq_f32(max_f32x4);
        if (min_value == NK_F32_MAX && max_value == NK_F32_MIN) {
            *min_value_ptr = NK_F32_MAX, *min_index_ptr = NK_SIZE_MAX;
            *max_value_ptr = NK_F32_MIN, *max_index_ptr = NK_SIZE_MAX;
            return;
        }
        uint32x4_t min_value_match_u32x4 = vceqq_f32(min_f32x4, vdupq_n_f32(min_value));
        uint32x4_t masked_min_iter_u32x4 = vbslq_u32(min_value_match_u32x4, min_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
        nk_u32_t earliest_min_cycle = vminvq_u32(masked_min_iter_u32x4);
        uint32x4_t max_value_match_u32x4 = vceqq_f32(max_f32x4, vdupq_n_f32(max_value));
        uint32x4_t masked_max_iter_u32x4 = vbslq_u32(max_value_match_u32x4, max_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
        nk_u32_t earliest_max_cycle = vminvq_u32(masked_max_iter_u32x4);
        uint32x4_t min_cycle_match_u32x4 = vceqq_u32(min_iter_u32x4, vdupq_n_u32(earliest_min_cycle));
        uint32x4_t min_both_match_u32x4 = vandq_u32(min_value_match_u32x4, min_cycle_match_u32x4);
        uint32x4_t min_masked_lanes_u32x4 = vbslq_u32(min_both_match_u32x4, lane_indices_u32x4,
                                                      vdupq_n_u32(NK_U32_MAX));
        nk_u32_t min_lane_offset = vminvq_u32(min_masked_lanes_u32x4);
        nk_size_t min_idx = (nk_size_t)earliest_min_cycle * 4 + (nk_size_t)min_lane_offset;
        uint32x4_t max_cycle_match_u32x4 = vceqq_u32(max_iter_u32x4, vdupq_n_u32(earliest_max_cycle));
        uint32x4_t max_both_match_u32x4 = vandq_u32(max_value_match_u32x4, max_cycle_match_u32x4);
        uint32x4_t max_masked_lanes_u32x4 = vbslq_u32(max_both_match_u32x4, lane_indices_u32x4,
                                                      vdupq_n_u32(NK_U32_MAX));
        nk_u32_t max_lane_offset = vminvq_u32(max_masked_lanes_u32x4);
        nk_size_t max_idx = (nk_size_t)earliest_max_cycle * 4 + (nk_size_t)max_lane_offset;
        *min_value_ptr = min_value, *min_index_ptr = min_idx;
        *max_value_ptr = max_value, *max_index_ptr = max_idx;
        return;
    }

    // Shared update body
    uint32x4_t less_u32x4 = vcltq_f32(data_for_min_f32x4, min_f32x4);
    uint32x4_t greater_u32x4 = vcgtq_f32(data_for_max_f32x4, max_f32x4);
    min_f32x4 = vbslq_f32(less_u32x4, data_for_min_f32x4, min_f32x4);
    max_f32x4 = vbslq_f32(greater_u32x4, data_for_max_f32x4, max_f32x4);
    min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
    max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
    iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
    goto nk_reduce_minmax_f32_neon_cycle;
}

NK_PUBLIC void nk_reduce_minmax_f32_neon(                              //
    nk_f32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_f32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_F32_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F32_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f32_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)NK_U32_MAX * 4) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_f32_neon(data_ptr, left_count, stride_bytes, &left_min_value, &left_min_index, &left_max_value,
                                  &left_max_index);
        nk_reduce_minmax_f32_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                  &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (right_min_value < left_min_value)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (right_max_value > left_max_value)
            *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_f32_neon_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else if (stride_elements <= 4)
        nk_reduce_minmax_f32_neon_strided_(data_ptr, count, stride_elements, min_value_ptr, min_index_ptr,
                                           max_value_ptr, max_index_ptr);
    else
        nk_reduce_minmax_f32_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_f64_neon_contiguous_( //
    nk_f64_t const *data_ptr, nk_size_t count,           //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    float64x2_t sum_f64x2 = vdupq_n_f64(0), sum_compensation_f64x2 = vdupq_n_f64(0);
    float64x2_t sumsq_f64x2 = vdupq_n_f64(0), sumsq_compensation_f64x2 = vdupq_n_f64(0);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        float64x2_t data_f64x2 = vld1q_f64(data_ptr + idx);
        float64x2_t temp_sum_f64x2 = vaddq_f64(sum_f64x2, data_f64x2);
        float64x2_t residual_f64x2 = vsubq_f64(temp_sum_f64x2, sum_f64x2);
        sum_compensation_f64x2 = vaddq_f64(sum_compensation_f64x2,
                                           vaddq_f64(vsubq_f64(sum_f64x2, vsubq_f64(temp_sum_f64x2, residual_f64x2)),
                                                     vsubq_f64(data_f64x2, residual_f64x2)));
        sum_f64x2 = temp_sum_f64x2;
        float64x2_t data_squared_f64x2 = vmulq_f64(data_f64x2, data_f64x2);
        float64x2_t temp_sumsq_f64x2 = vaddq_f64(sumsq_f64x2, data_squared_f64x2);
        float64x2_t residual_sumsq_f64x2 = vsubq_f64(temp_sumsq_f64x2, sumsq_f64x2);
        sumsq_compensation_f64x2 = vaddq_f64(
            sumsq_compensation_f64x2,
            vaddq_f64(vsubq_f64(sumsq_f64x2, vsubq_f64(temp_sumsq_f64x2, residual_sumsq_f64x2)),
                      vsubq_f64(data_squared_f64x2, residual_sumsq_f64x2)));
        sumsq_f64x2 = temp_sumsq_f64x2;
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b64x2_serial_(data_ptr + idx, &tail_vec, remaining);
        float64x2_t data_f64x2 = tail_vec.f64x2;
        float64x2_t temp_sum_f64x2 = vaddq_f64(sum_f64x2, data_f64x2);
        float64x2_t residual_f64x2 = vsubq_f64(temp_sum_f64x2, sum_f64x2);
        sum_compensation_f64x2 = vaddq_f64(sum_compensation_f64x2,
                                           vaddq_f64(vsubq_f64(sum_f64x2, vsubq_f64(temp_sum_f64x2, residual_f64x2)),
                                                     vsubq_f64(data_f64x2, residual_f64x2)));
        sum_f64x2 = temp_sum_f64x2;
        float64x2_t data_squared_f64x2 = vmulq_f64(data_f64x2, data_f64x2);
        float64x2_t temp_sumsq_f64x2 = vaddq_f64(sumsq_f64x2, data_squared_f64x2);
        float64x2_t residual_sumsq_f64x2 = vsubq_f64(temp_sumsq_f64x2, sumsq_f64x2);
        sumsq_compensation_f64x2 = vaddq_f64(
            sumsq_compensation_f64x2,
            vaddq_f64(vsubq_f64(sumsq_f64x2, vsubq_f64(temp_sumsq_f64x2, residual_sumsq_f64x2)),
                      vsubq_f64(data_squared_f64x2, residual_sumsq_f64x2)));
        sumsq_f64x2 = temp_sumsq_f64x2;
    }
    *sum_ptr = vaddvq_f64(vaddq_f64(sum_f64x2, sum_compensation_f64x2));
    *sumsq_ptr = vaddvq_f64(vaddq_f64(sumsq_f64x2, sumsq_compensation_f64x2));
}

NK_PUBLIC void nk_reduce_moments_f64_neon(                             //
    nk_f64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_f64_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 2) {
        nk_size_t left_count = count / 2;
        nk_f64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f64_neon(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f64_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                   &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum, *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f64_neon_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_f64_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_f64_neon_contiguous_( //
    nk_f64_t const *data_ptr, nk_size_t count,          //
    nk_f64_t *min_value_ptr, nk_size_t *min_index_ptr,  //
    nk_f64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    float64x2_t min_f64x2 = vdupq_n_f64(NK_F64_MAX), max_f64x2 = vdupq_n_f64(NK_F64_MIN);
    uint64x2_t min_iter = vdupq_n_u64(0), max_iter = vdupq_n_u64(0);
    uint64x2_t iter = vdupq_n_u64(0), one = vdupq_n_u64(1);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        float64x2_t data_f64x2 = vld1q_f64(data_ptr + idx);
        uint64x2_t less_u64x2 = vcltq_f64(data_f64x2, min_f64x2);
        uint64x2_t greater_u64x2 = vcgtq_f64(data_f64x2, max_f64x2);
        min_f64x2 = vbslq_f64(less_u64x2, data_f64x2, min_f64x2);
        max_f64x2 = vbslq_f64(greater_u64x2, data_f64x2, max_f64x2);
        min_iter = vbslq_u64(less_u64x2, iter, min_iter);
        max_iter = vbslq_u64(greater_u64x2, iter, max_iter);
        iter = vaddq_u64(iter, one);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_indices_vec, max_indices_vec;
    min_values_vec.f64x2 = min_f64x2;
    min_indices_vec.u64x2 = min_iter;
    max_values_vec.f64x2 = max_f64x2;
    max_indices_vec.u64x2 = max_iter;
    nk_f64_t min_value, max_value;
    nk_size_t min_index, max_index;
    if (min_values_vec.f64s[0] <= min_values_vec.f64s[1])
        min_value = min_values_vec.f64s[0], min_index = (nk_size_t)min_indices_vec.u64s[0] * 2;
    else min_value = min_values_vec.f64s[1], min_index = (nk_size_t)min_indices_vec.u64s[1] * 2 + 1;
    if (max_values_vec.f64s[0] >= max_values_vec.f64s[1])
        max_value = max_values_vec.f64s[0], max_index = (nk_size_t)max_indices_vec.u64s[0] * 2;
    else max_value = max_values_vec.f64s[1], max_index = (nk_size_t)max_indices_vec.u64s[1] * 2 + 1;
    for (; idx < count; ++idx) {
        nk_f64_t val = data_ptr[idx];
        if (val < min_value) min_value = val, min_index = idx;
        if (val > max_value) max_value = val, max_index = idx;
    }
    // All-NaN / sentinel check: sentinels remain unchanged when all data is NaN.
    if (min_value == NK_F64_MAX && max_value == NK_F64_MIN) {
        *min_value_ptr = NK_F64_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F64_MIN,
        *max_index_ptr = NK_SIZE_MAX;
        return;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_index;
    *max_value_ptr = max_value, *max_index_ptr = max_index;
}

NK_PUBLIC void nk_reduce_minmax_f64_neon(                              //
    nk_f64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_f64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_F64_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F64_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f64_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_f64_neon_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else
        nk_reduce_minmax_f64_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_i8_neon_contiguous_( //
    nk_i8_t const *data_ptr, nk_size_t count,           //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        int8x16_t data_i8x16 = vld1q_s8(data_ptr + idx);
        int16x8_t pairwise_i16x8 = vpaddlq_s8(data_i8x16);
        sum_i32x4 = vaddq_s32(sum_i32x4, vpaddlq_s16(pairwise_i16x8));
        int16x8_t squares_lo_i16x8 = vmull_s8(vget_low_s8(data_i8x16), vget_low_s8(data_i8x16));
        int16x8_t squares_hi_i16x8 = vmull_high_s8(data_i8x16, data_i8x16);
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_lo_i16x8))));
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_hi_i16x8))));
    }
    nk_i64_t sum = vaddlvq_s32(sum_i32x4);
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_i64_t value_i64 = (nk_i64_t)data_ptr[idx];
        sum += value_i64, sumsq += (nk_u64_t)(value_i64 * value_i64);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_i8_neon_strided_(                     //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            int8x16x2_t loaded_i8x16x2 = vld2q_s8(data_ptr + idx * 2);
            int8x16_t data_i8x16 = loaded_i8x16x2.val[0];
            int16x8_t pairwise_i16x8 = vpaddlq_s8(data_i8x16);
            sum_i32x4 = vaddq_s32(sum_i32x4, vpaddlq_s16(pairwise_i16x8));
            int16x8_t squares_lo_i16x8 = vmull_s8(vget_low_s8(data_i8x16), vget_low_s8(data_i8x16));
            int16x8_t squares_hi_i16x8 = vmull_high_s8(data_i8x16, data_i8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_lo_i16x8))));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_hi_i16x8))));
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            int8x16x3_t loaded_i8x16x3 = vld3q_s8(data_ptr + idx * 3);
            int8x16_t data_i8x16 = loaded_i8x16x3.val[0];
            int16x8_t pairwise_i16x8 = vpaddlq_s8(data_i8x16);
            sum_i32x4 = vaddq_s32(sum_i32x4, vpaddlq_s16(pairwise_i16x8));
            int16x8_t squares_lo_i16x8 = vmull_s8(vget_low_s8(data_i8x16), vget_low_s8(data_i8x16));
            int16x8_t squares_hi_i16x8 = vmull_high_s8(data_i8x16, data_i8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_lo_i16x8))));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_hi_i16x8))));
        }
    }
    else {
        for (; idx + 16 <= count; idx += 16) {
            int8x16x4_t loaded_i8x16x4 = vld4q_s8(data_ptr + idx * 4);
            int8x16_t data_i8x16 = loaded_i8x16x4.val[0];
            int16x8_t pairwise_i16x8 = vpaddlq_s8(data_i8x16);
            sum_i32x4 = vaddq_s32(sum_i32x4, vpaddlq_s16(pairwise_i16x8));
            int16x8_t squares_lo_i16x8 = vmull_s8(vget_low_s8(data_i8x16), vget_low_s8(data_i8x16));
            int16x8_t squares_hi_i16x8 = vmull_high_s8(data_i8x16, data_i8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_lo_i16x8))));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_hi_i16x8))));
        }
    }
    nk_i64_t sum = vaddlvq_s32(sum_i32x4);
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_i64_t value_i64 = (nk_i64_t)data_ptr[idx * stride_elements];
        sum += value_i64, sumsq += (nk_u64_t)(value_i64 * value_i64);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i8_neon(                             //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_i8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum, right_sum;
        nk_u64_t left_sumsq, right_sumsq;
        nk_reduce_moments_i8_neon(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i8_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                  &right_sumsq);
        *sum_ptr = nk_i64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i8_neon_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_i8_neon_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_i8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_i8_neon_contiguous_( //
    nk_i8_t const *data_ptr, nk_size_t count,          //
    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr,  //
    nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    int8x16_t min_i8x16 = vdupq_n_s8(NK_I8_MAX), max_i8x16 = vdupq_n_s8(NK_I8_MIN);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        int8x16_t data_i8x16 = vld1q_s8(data_ptr + idx);
        uint8x16_t less_u8x16 = vcltq_s8(data_i8x16, min_i8x16);
        uint8x16_t greater_u8x16 = vcgtq_s8(data_i8x16, max_i8x16);
        min_i8x16 = vbslq_s8(less_u8x16, data_i8x16, min_i8x16);
        max_i8x16 = vbslq_s8(greater_u8x16, data_i8x16, max_i8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
        iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b8x16_serial_(data_ptr + idx, &tail_vec, remaining);
        uint8x16_t lane_indices_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                    vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
        uint8x16_t valid_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((uint8_t)remaining));
        int8x16_t data_for_min_i8x16 = vbslq_s8(valid_u8x16, tail_vec.i8x16, vdupq_n_s8(NK_I8_MAX));
        int8x16_t data_for_max_i8x16 = vbslq_s8(valid_u8x16, tail_vec.i8x16, vdupq_n_s8(NK_I8_MIN));
        uint8x16_t less_u8x16 = vcltq_s8(data_for_min_i8x16, min_i8x16);
        uint8x16_t greater_u8x16 = vcgtq_s8(data_for_max_i8x16, max_i8x16);
        min_i8x16 = vbslq_s8(less_u8x16, data_for_min_i8x16, min_i8x16);
        max_i8x16 = vbslq_s8(greater_u8x16, data_for_max_i8x16, max_i8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
    }
    nk_i8_t min_value = vminvq_s8(min_i8x16), max_value = vmaxvq_s8(max_i8x16);
    uint8x16_t min_value_match_u8x16 = vceqq_s8(min_i8x16, vdupq_n_s8(min_value));
    uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
    uint8x16_t max_value_match_u8x16 = vceqq_s8(max_i8x16, vdupq_n_s8(max_value));
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
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_i8_neon_strided_(                      //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr,                    //
    nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    int8x16_t min_i8x16 = vdupq_n_s8(NK_I8_MAX), max_i8x16 = vdupq_n_s8(NK_I8_MIN);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    uint8x16_t lane_indices_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
    nk_size_t idx = 0;
    int8x16_t data_for_min_i8x16, data_for_max_i8x16;

nk_reduce_minmax_i8_neon_cycle:
    if (stride_elements == 2 && idx + 16 <= count) {
        int8x16x2_t loaded = vld2q_s8(data_ptr + idx * 2);
        data_for_min_i8x16 = loaded.val[0];
        data_for_max_i8x16 = loaded.val[0];
        idx += 16;
    }
    else if (stride_elements == 3 && idx + 16 <= count) {
        int8x16x3_t loaded = vld3q_s8(data_ptr + idx * 3);
        data_for_min_i8x16 = loaded.val[0];
        data_for_max_i8x16 = loaded.val[0];
        idx += 16;
    }
    else if (stride_elements == 4 && idx + 16 <= count) {
        int8x16x4_t loaded = vld4q_s8(data_ptr + idx * 4);
        data_for_min_i8x16 = loaded.val[0];
        data_for_max_i8x16 = loaded.val[0];
        idx += 16;
    }
    else if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_strided_load_b8x16_serial_(data_ptr + idx * stride_elements, stride_elements, &tail_vec, count - idx);
        uint8x16_t valid_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((uint8_t)(count - idx)));
        data_for_min_i8x16 = vbslq_s8(valid_u8x16, tail_vec.i8x16, min_i8x16);
        data_for_max_i8x16 = vbslq_s8(valid_u8x16, tail_vec.i8x16, max_i8x16);
        idx = count;
    }
    else {
        nk_i8_t min_value = vminvq_s8(min_i8x16), max_value = vmaxvq_s8(max_i8x16);
        uint8x16_t min_value_match_u8x16 = vceqq_s8(min_i8x16, vdupq_n_s8(min_value));
        uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
        nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
        uint8x16_t max_value_match_u8x16 = vceqq_s8(max_i8x16, vdupq_n_s8(max_value));
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
        *min_value_ptr = min_value, *min_index_ptr = min_idx;
        *max_value_ptr = max_value, *max_index_ptr = max_idx;
        return;
    }

    // Shared update body
    uint8x16_t less_u8x16 = vcltq_s8(data_for_min_i8x16, min_i8x16);
    uint8x16_t greater_u8x16 = vcgtq_s8(data_for_max_i8x16, max_i8x16);
    min_i8x16 = vbslq_s8(less_u8x16, data_for_min_i8x16, min_i8x16);
    max_i8x16 = vbslq_s8(greater_u8x16, data_for_max_i8x16, max_i8x16);
    min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
    max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
    iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
    goto nk_reduce_minmax_i8_neon_cycle;
}

NK_PUBLIC void nk_reduce_minmax_i8_neon(                              //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_I8_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I8_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i8_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                   max_index_ptr);
    else if (count > (nk_size_t)(NK_U8_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_i8_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_i8_neon(data_ptr, left_count, stride_bytes, &left_min_value, &left_min_index, &left_max_value,
                                 &left_max_index);
        nk_reduce_minmax_i8_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                 &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (right_min_value < left_min_value)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (right_max_value > left_max_value)
            *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i8_neon_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                             max_index_ptr);
    else if (stride_elements <= 4)
        nk_reduce_minmax_i8_neon_strided_(data_ptr, count, stride_elements, min_value_ptr, min_index_ptr, max_value_ptr,
                                          max_index_ptr);
    else
        nk_reduce_minmax_i8_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                   max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_u8_neon_contiguous_( //
    nk_u8_t const *data_ptr, nk_size_t count,           //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    uint32x4_t sum_u32x4 = vdupq_n_u32(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t data_u8x16 = vld1q_u8(data_ptr + idx);
        uint16x8_t sum16 = vpaddlq_u8(data_u8x16);
        sum_u32x4 = vaddq_u32(sum_u32x4, vpaddlq_u16(sum16));
        uint16x8_t sq_lo = vmull_u8(vget_low_u8(data_u8x16), vget_low_u8(data_u8x16));
        uint16x8_t sq_hi = vmull_high_u8(data_u8x16, data_u8x16);
        uint32x4_t sq32_lo = vpaddlq_u16(sq_lo);
        uint32x4_t sq32_hi = vpaddlq_u16(sq_hi);
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq32_lo));
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq32_hi));
    }
    nk_u64_t sum = vaddlvq_u32(sum_u32x4);
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_u64_t value = (nk_u64_t)data_ptr[idx];
        sum += value, sumsq += value * value;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_u8_neon_strided_(                     //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    uint32x4_t sum_u32x4 = vdupq_n_u32(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x2_t loaded_u8x16x2 = vld2q_u8(data_ptr + idx * 2);
            uint8x16_t data_u8x16 = loaded_u8x16x2.val[0];
            uint16x8_t pairwise_u16x8 = vpaddlq_u8(data_u8x16);
            sum_u32x4 = vaddq_u32(sum_u32x4, vpaddlq_u16(pairwise_u16x8));
            uint16x8_t squares_lo_u16x8 = vmull_u8(vget_low_u8(data_u8x16), vget_low_u8(data_u8x16));
            uint16x8_t squares_hi_u16x8 = vmull_high_u8(data_u8x16, data_u8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(squares_lo_u16x8)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(squares_hi_u16x8)));
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x3_t loaded_u8x16x3 = vld3q_u8(data_ptr + idx * 3);
            uint8x16_t data_u8x16 = loaded_u8x16x3.val[0];
            uint16x8_t pairwise_u16x8 = vpaddlq_u8(data_u8x16);
            sum_u32x4 = vaddq_u32(sum_u32x4, vpaddlq_u16(pairwise_u16x8));
            uint16x8_t squares_lo_u16x8 = vmull_u8(vget_low_u8(data_u8x16), vget_low_u8(data_u8x16));
            uint16x8_t squares_hi_u16x8 = vmull_high_u8(data_u8x16, data_u8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(squares_lo_u16x8)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(squares_hi_u16x8)));
        }
    }
    else {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x4_t loaded_u8x16x4 = vld4q_u8(data_ptr + idx * 4);
            uint8x16_t data_u8x16 = loaded_u8x16x4.val[0];
            uint16x8_t pairwise_u16x8 = vpaddlq_u8(data_u8x16);
            sum_u32x4 = vaddq_u32(sum_u32x4, vpaddlq_u16(pairwise_u16x8));
            uint16x8_t squares_lo_u16x8 = vmull_u8(vget_low_u8(data_u8x16), vget_low_u8(data_u8x16));
            uint16x8_t squares_hi_u16x8 = vmull_high_u8(data_u8x16, data_u8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(squares_lo_u16x8)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(squares_hi_u16x8)));
        }
    }
    nk_u64_t sum = vaddlvq_u32(sum_u32x4);
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_u64_t value = (nk_u64_t)data_ptr[idx * stride_elements];
        sum += value, sumsq += value * value;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u8_neon(                             //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_u8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u8_neon(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u8_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes, &right_sum,
                                  &right_sumsq);
        *sum_ptr = nk_u64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u8_neon_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_u8_neon_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_u8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_u8_neon_contiguous_( //
    nk_u8_t const *data_ptr, nk_size_t count,          //
    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,  //
    nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    uint8x16_t min_u8x16 = vdupq_n_u8(NK_U8_MAX), max_u8x16 = vdupq_n_u8(0);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t data_u8x16 = vld1q_u8(data_ptr + idx);
        uint8x16_t less_u8x16 = vcltq_u8(data_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(data_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, data_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, data_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
        iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b8x16_serial_(data_ptr + idx, &tail_vec, remaining);
        uint8x16_t lane_indices_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                    vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
        uint8x16_t valid_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((uint8_t)remaining));
        uint8x16_t data_for_min_u8x16 = vbslq_u8(valid_u8x16, tail_vec.u8x16, vdupq_n_u8(NK_U8_MAX));
        uint8x16_t data_for_max_u8x16 = vbslq_u8(valid_u8x16, tail_vec.u8x16, vdupq_n_u8(0));
        uint8x16_t less_u8x16 = vcltq_u8(data_for_min_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(data_for_max_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, data_for_min_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, data_for_max_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
    }
    nk_u8_t min_value = vminvq_u8(min_u8x16), max_value = vmaxvq_u8(max_u8x16);
    uint8x16_t min_value_match_u8x16 = vceqq_u8(min_u8x16, vdupq_n_u8(min_value));
    uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
    uint8x16_t max_value_match_u8x16 = vceqq_u8(max_u8x16, vdupq_n_u8(max_value));
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
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_u8_neon_strided_(                      //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,                    //
    nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    uint8x16_t min_u8x16 = vdupq_n_u8(NK_U8_MAX), max_u8x16 = vdupq_n_u8(0);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    uint8x16_t lane_indices_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
    nk_size_t idx = 0;
    uint8x16_t data_for_min_u8x16, data_for_max_u8x16;

nk_reduce_minmax_u8_neon_cycle:
    if (stride_elements == 2 && idx + 16 <= count) {
        uint8x16x2_t loaded = vld2q_u8((nk_u8_t const *)data_ptr + idx * 2);
        data_for_min_u8x16 = loaded.val[0];
        data_for_max_u8x16 = loaded.val[0];
        idx += 16;
    }
    else if (stride_elements == 3 && idx + 16 <= count) {
        uint8x16x3_t loaded = vld3q_u8((nk_u8_t const *)data_ptr + idx * 3);
        data_for_min_u8x16 = loaded.val[0];
        data_for_max_u8x16 = loaded.val[0];
        idx += 16;
    }
    else if (stride_elements == 4 && idx + 16 <= count) {
        uint8x16x4_t loaded = vld4q_u8((nk_u8_t const *)data_ptr + idx * 4);
        data_for_min_u8x16 = loaded.val[0];
        data_for_max_u8x16 = loaded.val[0];
        idx += 16;
    }
    else if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_strided_load_b8x16_serial_(data_ptr + idx * stride_elements, stride_elements, &tail_vec, count - idx);
        uint8x16_t valid_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((uint8_t)(count - idx)));
        data_for_min_u8x16 = vbslq_u8(valid_u8x16, tail_vec.u8x16, min_u8x16);
        data_for_max_u8x16 = vbslq_u8(valid_u8x16, tail_vec.u8x16, max_u8x16);
        idx = count;
    }
    else {
        nk_u8_t min_value = vminvq_u8(min_u8x16), max_value = vmaxvq_u8(max_u8x16);
        uint8x16_t min_value_match_u8x16 = vceqq_u8(min_u8x16, vdupq_n_u8(min_value));
        uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
        nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
        uint8x16_t max_value_match_u8x16 = vceqq_u8(max_u8x16, vdupq_n_u8(max_value));
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
        *min_value_ptr = min_value, *min_index_ptr = min_idx;
        *max_value_ptr = max_value, *max_index_ptr = max_idx;
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
    goto nk_reduce_minmax_u8_neon_cycle;
}

NK_PUBLIC void nk_reduce_minmax_u8_neon(                              //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_U8_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = 0, *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u8_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                   max_index_ptr);
    else if (count > (nk_size_t)(NK_U8_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_u8_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_u8_neon(data_ptr, left_count, stride_bytes, &left_min_value, &left_min_index, &left_max_value,
                                 &left_max_index);
        nk_reduce_minmax_u8_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                 &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (right_min_value < left_min_value)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (right_max_value > left_max_value)
            *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u8_neon_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                             max_index_ptr);
    else if (stride_elements <= 4)
        nk_reduce_minmax_u8_neon_strided_(data_ptr, count, stride_elements, min_value_ptr, min_index_ptr, max_value_ptr,
                                          max_index_ptr);
    else
        nk_reduce_minmax_u8_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                   max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_i16_neon_contiguous_( //
    nk_i16_t const *data_ptr, nk_size_t count,           //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    int64x2_t sum_i64x2 = vdupq_n_s64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        int16x8_t data_i16x8 = vld1q_s16(data_ptr + idx);
        int32x4_t sum32 = vpaddlq_s16(data_i16x8);
        sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(sum32));
        // sumsq: widening multiply i16*i16 -> i32, then widen to u64
        int32x4_t sq_lo = vmull_s16(vget_low_s16(data_i16x8), vget_low_s16(data_i16x8));
        int32x4_t sq_hi = vmull_high_s16(data_i16x8, data_i16x8);
        // i16*i16 squares are always non-negative, safe to reinterpret as u32
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(sq_lo)));
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(sq_hi)));
    }
    nk_i64_t sum = vgetq_lane_s64(sum_i64x2, 0) + vgetq_lane_s64(sum_i64x2, 1);
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_i64_t value_i64 = (nk_i64_t)data_ptr[idx];
        sum += value_i64;
        sumsq += (nk_u64_t)(value_i64 * value_i64);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_i16_neon_strided_(                     //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    int64x2_t sum_i64x2 = vdupq_n_s64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            int16x8x2_t loaded_i16x8x2 = vld2q_s16(data_ptr + idx * 2);
            int16x8_t data_i16x8 = loaded_i16x8x2.val[0];
            int32x4_t pairwise_i32x4 = vpaddlq_s16(data_i16x8);
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(pairwise_i32x4));
            int32x4_t squares_lo_i32x4 = vmull_s16(vget_low_s16(data_i16x8), vget_low_s16(data_i16x8));
            int32x4_t squares_hi_i32x4 = vmull_high_s16(data_i16x8, data_i16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_i32x4)));
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            int16x8x3_t loaded_i16x8x3 = vld3q_s16(data_ptr + idx * 3);
            int16x8_t data_i16x8 = loaded_i16x8x3.val[0];
            int32x4_t pairwise_i32x4 = vpaddlq_s16(data_i16x8);
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(pairwise_i32x4));
            int32x4_t squares_lo_i32x4 = vmull_s16(vget_low_s16(data_i16x8), vget_low_s16(data_i16x8));
            int32x4_t squares_hi_i32x4 = vmull_high_s16(data_i16x8, data_i16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_i32x4)));
        }
    }
    else {
        for (; idx + 8 <= count; idx += 8) {
            int16x8x4_t loaded_i16x8x4 = vld4q_s16(data_ptr + idx * 4);
            int16x8_t data_i16x8 = loaded_i16x8x4.val[0];
            int32x4_t pairwise_i32x4 = vpaddlq_s16(data_i16x8);
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(pairwise_i32x4));
            int32x4_t squares_lo_i32x4 = vmull_s16(vget_low_s16(data_i16x8), vget_low_s16(data_i16x8));
            int32x4_t squares_hi_i32x4 = vmull_high_s16(data_i16x8, data_i16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_i32x4)));
        }
    }
    nk_i64_t sum = vgetq_lane_s64(sum_i64x2, 0) + vgetq_lane_s64(sum_i64x2, 1);
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_i64_t value_i64 = (nk_i64_t)data_ptr[idx * stride_elements];
        sum += value_i64, sumsq += (nk_u64_t)(value_i64 * value_i64);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i16_neon(                             //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_i16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_i64_t left_sum, right_sum;
        nk_u64_t left_sumsq, right_sumsq;
        nk_reduce_moments_i16_neon(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i16_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                   &right_sum, &right_sumsq);
        *sum_ptr = nk_i64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i16_neon_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_i16_neon_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_i16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_i16_neon_contiguous_( //
    nk_i16_t const *data_ptr, nk_size_t count,          //
    nk_i16_t *min_value_ptr, nk_size_t *min_index_ptr,  //
    nk_i16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    int16x8_t min_i16x8 = vdupq_n_s16(NK_I16_MAX), max_i16x8 = vdupq_n_s16(NK_I16_MIN);
    uint16x8_t min_iter_u16x8 = vdupq_n_u16(0), max_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t iter_u16x8 = vdupq_n_u16(0), one_u16x8 = vdupq_n_u16(1);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        int16x8_t data_i16x8 = vld1q_s16(data_ptr + idx);
        uint16x8_t less_u16x8 = vcltq_s16(data_i16x8, min_i16x8);
        uint16x8_t greater_u16x8 = vcgtq_s16(data_i16x8, max_i16x8);
        min_i16x8 = vbslq_s16(less_u16x8, data_i16x8, min_i16x8);
        max_i16x8 = vbslq_s16(greater_u16x8, data_i16x8, max_i16x8);
        min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
        max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
        iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b16x8_serial_(data_ptr + idx, &tail_vec, remaining);
        uint16x8_t lane_indices_u16x8 = vcombine_u16(vreinterpret_u16_u64(vcreate_u64(0x0003000200010000ULL)),
                                                     vreinterpret_u16_u64(vcreate_u64(0x0007000600050004ULL)));
        uint16x8_t valid_u16x8 = vcltq_u16(lane_indices_u16x8, vdupq_n_u16((uint16_t)remaining));
        int16x8_t data_for_min_i16x8 = vbslq_s16(valid_u16x8, tail_vec.i16x8, vdupq_n_s16(NK_I16_MAX));
        int16x8_t data_for_max_i16x8 = vbslq_s16(valid_u16x8, tail_vec.i16x8, vdupq_n_s16(NK_I16_MIN));
        uint16x8_t less_u16x8 = vcltq_s16(data_for_min_i16x8, min_i16x8);
        uint16x8_t greater_u16x8 = vcgtq_s16(data_for_max_i16x8, max_i16x8);
        min_i16x8 = vbslq_s16(less_u16x8, data_for_min_i16x8, min_i16x8);
        max_i16x8 = vbslq_s16(greater_u16x8, data_for_max_i16x8, max_i16x8);
        min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
        max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
    }
    nk_i16_t min_value = vminvq_s16(min_i16x8), max_value = vmaxvq_s16(max_i16x8);
    uint16x8_t min_value_match_u16x8 = vceqq_s16(min_i16x8, vdupq_n_s16(min_value));
    uint16x8_t masked_min_iter_u16x8 = vbslq_u16(min_value_match_u16x8, min_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_min_cycle = vminvq_u16(masked_min_iter_u16x8);
    uint16x8_t max_value_match_u16x8 = vceqq_s16(max_i16x8, vdupq_n_s16(max_value));
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
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_i16_neon_strided_(                      //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_i16_t *min_value_ptr, nk_size_t *min_index_ptr,                    //
    nk_i16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    int16x8_t min_i16x8 = vdupq_n_s16(NK_I16_MAX), max_i16x8 = vdupq_n_s16(NK_I16_MIN);
    uint16x8_t min_iter_u16x8 = vdupq_n_u16(0), max_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t iter_u16x8 = vdupq_n_u16(0), one_u16x8 = vdupq_n_u16(1);
    uint16x8_t lane_indices_u16x8 = vcombine_u16(vreinterpret_u16_u64(vcreate_u64(0x0003000200010000ULL)),
                                                 vreinterpret_u16_u64(vcreate_u64(0x0007000600050004ULL)));
    nk_size_t idx = 0;
    int16x8_t data_for_min_i16x8, data_for_max_i16x8;

nk_reduce_minmax_i16_neon_cycle:
    if (stride_elements == 2 && idx + 8 <= count) {
        int16x8x2_t loaded = vld2q_s16(data_ptr + idx * 2);
        data_for_min_i16x8 = loaded.val[0];
        data_for_max_i16x8 = loaded.val[0];
        idx += 8;
    }
    else if (stride_elements == 3 && idx + 8 <= count) {
        int16x8x3_t loaded = vld3q_s16(data_ptr + idx * 3);
        data_for_min_i16x8 = loaded.val[0];
        data_for_max_i16x8 = loaded.val[0];
        idx += 8;
    }
    else if (stride_elements == 4 && idx + 8 <= count) {
        int16x8x4_t loaded = vld4q_s16(data_ptr + idx * 4);
        data_for_min_i16x8 = loaded.val[0];
        data_for_max_i16x8 = loaded.val[0];
        idx += 8;
    }
    else if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_strided_load_b16x8_serial_(data_ptr + idx * stride_elements, stride_elements, &tail_vec, count - idx);
        uint16x8_t valid_u16x8 = vcltq_u16(lane_indices_u16x8, vdupq_n_u16((uint16_t)(count - idx)));
        data_for_min_i16x8 = vbslq_s16(valid_u16x8, tail_vec.i16x8, min_i16x8);
        data_for_max_i16x8 = vbslq_s16(valid_u16x8, tail_vec.i16x8, max_i16x8);
        idx = count;
    }
    else {
        nk_i16_t min_value = vminvq_s16(min_i16x8), max_value = vmaxvq_s16(max_i16x8);
        uint16x8_t min_value_match_u16x8 = vceqq_s16(min_i16x8, vdupq_n_s16(min_value));
        uint16x8_t masked_min_iter_u16x8 = vbslq_u16(min_value_match_u16x8, min_iter_u16x8, vdupq_n_u16(0xFFFF));
        nk_u16_t earliest_min_cycle = vminvq_u16(masked_min_iter_u16x8);
        uint16x8_t max_value_match_u16x8 = vceqq_s16(max_i16x8, vdupq_n_s16(max_value));
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
        *min_value_ptr = min_value, *min_index_ptr = min_idx;
        *max_value_ptr = max_value, *max_index_ptr = max_idx;
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
    goto nk_reduce_minmax_i16_neon_cycle;
}

NK_PUBLIC void nk_reduce_minmax_i16_neon(                              //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_I16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_i16_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_i16_neon(data_ptr, left_count, stride_bytes, &left_min_value, &left_min_index, &left_max_value,
                                  &left_max_index);
        nk_reduce_minmax_i16_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                  &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (right_min_value < left_min_value)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (right_max_value > left_max_value)
            *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i16_neon_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else if (stride_elements <= 4)
        nk_reduce_minmax_i16_neon_strided_(data_ptr, count, stride_elements, min_value_ptr, min_index_ptr,
                                           max_value_ptr, max_index_ptr);
    else
        nk_reduce_minmax_i16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_u16_neon_contiguous_( //
    nk_u16_t const *data_ptr, nk_size_t count,           //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        uint16x8_t data_u16x8 = vld1q_u16(data_ptr + idx);
        uint32x4_t sum32 = vpaddlq_u16(data_u16x8);
        sum_u64x2 = vaddq_u64(sum_u64x2, vpaddlq_u32(sum32));
        uint32x4_t sq_lo = vmull_u16(vget_low_u16(data_u16x8), vget_low_u16(data_u16x8));
        uint32x4_t sq_hi = vmull_high_u16(data_u16x8, data_u16x8);
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq_lo));
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq_hi));
    }
    nk_u64_t sum = vgetq_lane_u64(sum_u64x2, 0) + vgetq_lane_u64(sum_u64x2, 1);
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_u64_t value = (nk_u64_t)data_ptr[idx];
        sum += value, sumsq += value * value;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_u16_neon_strided_(                     //
    nk_u16_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16(data_ptr + idx * 2);
            uint16x8_t data_u16x8 = loaded_u16x8x2.val[0];
            uint32x4_t widened_sum_u32x4 = vpaddlq_u16(data_u16x8);
            sum_u64x2 = vaddq_u64(sum_u64x2, vpaddlq_u32(widened_sum_u32x4));
            uint32x4_t sq_lo_u32x4 = vmull_u16(vget_low_u16(data_u16x8), vget_low_u16(data_u16x8));
            uint32x4_t sq_hi_u32x4 = vmull_high_u16(data_u16x8, data_u16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq_lo_u32x4));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq_hi_u32x4));
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16(data_ptr + idx * 3);
            uint16x8_t data_u16x8 = loaded_u16x8x3.val[0];
            uint32x4_t widened_sum_u32x4 = vpaddlq_u16(data_u16x8);
            sum_u64x2 = vaddq_u64(sum_u64x2, vpaddlq_u32(widened_sum_u32x4));
            uint32x4_t sq_lo_u32x4 = vmull_u16(vget_low_u16(data_u16x8), vget_low_u16(data_u16x8));
            uint32x4_t sq_hi_u32x4 = vmull_high_u16(data_u16x8, data_u16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq_lo_u32x4));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq_hi_u32x4));
        }
    }
    else {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16(data_ptr + idx * 4);
            uint16x8_t data_u16x8 = loaded_u16x8x4.val[0];
            uint32x4_t widened_sum_u32x4 = vpaddlq_u16(data_u16x8);
            sum_u64x2 = vaddq_u64(sum_u64x2, vpaddlq_u32(widened_sum_u32x4));
            uint32x4_t sq_lo_u32x4 = vmull_u16(vget_low_u16(data_u16x8), vget_low_u16(data_u16x8));
            uint32x4_t sq_hi_u32x4 = vmull_high_u16(data_u16x8, data_u16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq_lo_u32x4));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq_hi_u32x4));
        }
    }

    nk_u64_t sum = vgetq_lane_u64(sum_u64x2, 0) + vgetq_lane_u64(sum_u64x2, 1);
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_u64_t value = (nk_u64_t)data_ptr[idx * stride_elements];
        sum += value, sumsq += value * value;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u16_neon(                             //
    nk_u16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_u16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u16_neon(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u16_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                   &right_sum, &right_sumsq);
        *sum_ptr = nk_u64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u16_neon_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_u16_neon_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_u16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_u16_neon_contiguous_( //
    nk_u16_t const *data_ptr, nk_size_t count,          //
    nk_u16_t *min_value_ptr, nk_size_t *min_index_ptr,  //
    nk_u16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    uint16x8_t min_u16x8 = vdupq_n_u16(NK_U16_MAX), max_u16x8 = vdupq_n_u16(0);
    uint16x8_t min_iter_u16x8 = vdupq_n_u16(0), max_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t iter_u16x8 = vdupq_n_u16(0), one_u16x8 = vdupq_n_u16(1);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        uint16x8_t data_u16x8 = vld1q_u16(data_ptr + idx);
        uint16x8_t less_u16x8 = vcltq_u16(data_u16x8, min_u16x8);
        uint16x8_t greater_u16x8 = vcgtq_u16(data_u16x8, max_u16x8);
        min_u16x8 = vbslq_u16(less_u16x8, data_u16x8, min_u16x8);
        max_u16x8 = vbslq_u16(greater_u16x8, data_u16x8, max_u16x8);
        min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
        max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
        iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b16x8_serial_(data_ptr + idx, &tail_vec, remaining);
        uint16x8_t lane_indices_u16x8 = vcombine_u16(vreinterpret_u16_u64(vcreate_u64(0x0003000200010000ULL)),
                                                     vreinterpret_u16_u64(vcreate_u64(0x0007000600050004ULL)));
        uint16x8_t valid_u16x8 = vcltq_u16(lane_indices_u16x8, vdupq_n_u16((uint16_t)remaining));
        uint16x8_t data_for_min_u16x8 = vbslq_u16(valid_u16x8, tail_vec.u16x8, vdupq_n_u16(NK_U16_MAX));
        uint16x8_t data_for_max_u16x8 = vbslq_u16(valid_u16x8, tail_vec.u16x8, vdupq_n_u16(0));
        uint16x8_t less_u16x8 = vcltq_u16(data_for_min_u16x8, min_u16x8);
        uint16x8_t greater_u16x8 = vcgtq_u16(data_for_max_u16x8, max_u16x8);
        min_u16x8 = vbslq_u16(less_u16x8, data_for_min_u16x8, min_u16x8);
        max_u16x8 = vbslq_u16(greater_u16x8, data_for_max_u16x8, max_u16x8);
        min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
        max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
    }
    nk_u16_t min_value = vminvq_u16(min_u16x8), max_value = vmaxvq_u16(max_u16x8);
    uint16x8_t min_value_match_u16x8 = vceqq_u16(min_u16x8, vdupq_n_u16(min_value));
    uint16x8_t masked_min_iter_u16x8 = vbslq_u16(min_value_match_u16x8, min_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_min_cycle = vminvq_u16(masked_min_iter_u16x8);
    uint16x8_t max_value_match_u16x8 = vceqq_u16(max_u16x8, vdupq_n_u16(max_value));
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
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_u16_neon_strided_(                      //
    nk_u16_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_u16_t *min_value_ptr, nk_size_t *min_index_ptr,                    //
    nk_u16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    uint16x8_t min_u16x8 = vdupq_n_u16(NK_U16_MAX), max_u16x8 = vdupq_n_u16(0);
    uint16x8_t min_iter_u16x8 = vdupq_n_u16(0), max_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t iter_u16x8 = vdupq_n_u16(0), one_u16x8 = vdupq_n_u16(1);
    uint16x8_t lane_indices_u16x8 = vcombine_u16(vreinterpret_u16_u64(vcreate_u64(0x0003000200010000ULL)),
                                                 vreinterpret_u16_u64(vcreate_u64(0x0007000600050004ULL)));
    nk_size_t idx = 0;
    uint16x8_t data_for_min_u16x8, data_for_max_u16x8;

nk_reduce_minmax_u16_neon_cycle:
    if (stride_elements == 2 && idx + 8 <= count) {
        uint16x8x2_t loaded = vld2q_u16((nk_u16_t const *)data_ptr + idx * 2);
        data_for_min_u16x8 = loaded.val[0];
        data_for_max_u16x8 = loaded.val[0];
        idx += 8;
    }
    else if (stride_elements == 3 && idx + 8 <= count) {
        uint16x8x3_t loaded = vld3q_u16((nk_u16_t const *)data_ptr + idx * 3);
        data_for_min_u16x8 = loaded.val[0];
        data_for_max_u16x8 = loaded.val[0];
        idx += 8;
    }
    else if (stride_elements == 4 && idx + 8 <= count) {
        uint16x8x4_t loaded = vld4q_u16((nk_u16_t const *)data_ptr + idx * 4);
        data_for_min_u16x8 = loaded.val[0];
        data_for_max_u16x8 = loaded.val[0];
        idx += 8;
    }
    else if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_strided_load_b16x8_serial_(data_ptr + idx * stride_elements, stride_elements, &tail_vec, count - idx);
        uint16x8_t valid_u16x8 = vcltq_u16(lane_indices_u16x8, vdupq_n_u16((uint16_t)(count - idx)));
        data_for_min_u16x8 = vbslq_u16(valid_u16x8, tail_vec.u16x8, min_u16x8);
        data_for_max_u16x8 = vbslq_u16(valid_u16x8, tail_vec.u16x8, max_u16x8);
        idx = count;
    }
    else {
        nk_u16_t min_value = vminvq_u16(min_u16x8), max_value = vmaxvq_u16(max_u16x8);
        uint16x8_t min_value_match_u16x8 = vceqq_u16(min_u16x8, vdupq_n_u16(min_value));
        uint16x8_t masked_min_iter_u16x8 = vbslq_u16(min_value_match_u16x8, min_iter_u16x8, vdupq_n_u16(0xFFFF));
        nk_u16_t earliest_min_cycle = vminvq_u16(masked_min_iter_u16x8);
        uint16x8_t max_value_match_u16x8 = vceqq_u16(max_u16x8, vdupq_n_u16(max_value));
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
        *min_value_ptr = min_value, *min_index_ptr = min_idx;
        *max_value_ptr = max_value, *max_index_ptr = max_idx;
        return;
    }

    // Shared update body
    uint16x8_t less_u16x8 = vcltq_u16(data_for_min_u16x8, min_u16x8);
    uint16x8_t greater_u16x8 = vcgtq_u16(data_for_max_u16x8, max_u16x8);
    min_u16x8 = vbslq_u16(less_u16x8, data_for_min_u16x8, min_u16x8);
    max_u16x8 = vbslq_u16(greater_u16x8, data_for_max_u16x8, max_u16x8);
    min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
    max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
    iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
    goto nk_reduce_minmax_u16_neon_cycle;
}

NK_PUBLIC void nk_reduce_minmax_u16_neon(                              //
    nk_u16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_U16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = 0, *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_u16_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_u16_neon(data_ptr, left_count, stride_bytes, &left_min_value, &left_min_index, &left_max_value,
                                  &left_max_index);
        nk_reduce_minmax_u16_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                  &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (right_min_value < left_min_value)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (right_max_value > left_max_value)
            *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u16_neon_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else if (stride_elements <= 4)
        nk_reduce_minmax_u16_neon_strided_(data_ptr, count, stride_elements, min_value_ptr, min_index_ptr,
                                           max_value_ptr, max_index_ptr);
    else
        nk_reduce_minmax_u16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_i32_neon_contiguous_( //
    nk_i32_t const *data_ptr, nk_size_t count,           //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    // 128-bit accumulation: lower (u64) + upper (i64) per lane
    uint64x2_t sum_lower_u64x2 = vdupq_n_u64(0);
    int64x2_t sum_upper_i64x2 = vdupq_n_s64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    int sumsq_overflow = 0;
    // XOR sign-bit trick for unsigned u64 compare on NEON
    int64x2_t sign_bit_i64x2 = vdupq_n_s64((nk_i64_t)0x8000000000000000ULL);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        int32x4_t data_i32x4 = vld1q_s32(data_ptr + idx);
        // Sum: widen i32->i64 and accumulate with carry detection
        int64x2_t data_low_f64x2 = vmovl_s32(vget_low_s32(data_i32x4));
        uint64x2_t before = sum_lower_u64x2;
        sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(data_low_f64x2));
        int64x2_t result_biased = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
        int64x2_t before_biased = veorq_s64(vreinterpretq_s64_u64(before), sign_bit_i64x2);
        uint64x2_t carry = vcgtq_s64(before_biased, result_biased);
        sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry));
        sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, vshrq_n_s64(data_low_f64x2, 63));

        int64x2_t data_high_f64x2 = vmovl_high_s32(data_i32x4);
        before = sum_lower_u64x2;
        sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(data_high_f64x2));
        result_biased = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
        before_biased = veorq_s64(vreinterpretq_s64_u64(before), sign_bit_i64x2);
        carry = vcgtq_s64(before_biased, result_biased);
        sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry));
        sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, vshrq_n_s64(data_high_f64x2, 63));

        // Sumsq: widening multiply i32*i32 -> i64 (always non-negative for squares)
        int64x2_t sq_lo = vmull_s32(vget_low_s32(data_i32x4), vget_low_s32(data_i32x4));
        int64x2_t sq_hi = vmull_high_s32(data_i32x4, data_i32x4);
        uint64x2_t sq_before = sumsq_u64x2;
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vreinterpretq_u64_s64(sq_lo));
        result_biased = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
        before_biased = veorq_s64(vreinterpretq_s64_u64(sq_before), sign_bit_i64x2);
        sumsq_overflow |= (vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased, result_biased)), 0) |
                           vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased, result_biased)), 1));
        sq_before = sumsq_u64x2;
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vreinterpretq_u64_s64(sq_hi));
        result_biased = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
        before_biased = veorq_s64(vreinterpretq_s64_u64(sq_before), sign_bit_i64x2);
        sumsq_overflow |= (vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased, result_biased)), 0) |
                           vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased, result_biased)), 1));
    }
    // Sumsq horizontal saturating reduction
    nk_u64_t sumsq;
    if (sumsq_overflow) sumsq = NK_U64_MAX;
    else sumsq = nk_reduce_sadd_u64x2_neon_(sumsq_u64x2);
    // Sum: horizontal 128-bit reduction (2 lanes -> scalar)
    nk_b128_vec_t lower_vec, upper_vec;
    lower_vec.u64x2 = sum_lower_u64x2;
    upper_vec.i64x2 = sum_upper_i64x2;
    nk_u64_t sum_lower = 0;
    nk_i64_t sum_upper = 0;
    nk_u64_t sum_before = sum_lower;
    sum_lower += lower_vec.u64s[0], sum_upper += (sum_lower < sum_before) + upper_vec.i64s[0];
    sum_before = sum_lower;
    sum_lower += lower_vec.u64s[1], sum_upper += (sum_lower < sum_before) + upper_vec.i64s[1];
    // Scalar tail
    for (; idx < count; ++idx) {
        nk_i64_t value_i64 = (nk_i64_t)data_ptr[idx];
        sum_before = sum_lower;
        sum_lower += (nk_u64_t)value_i64;
        if (sum_lower < sum_before) sum_upper++;
        sum_upper += (value_i64 >> 63);
        nk_i64_t product = nk_i64_saturating_mul_serial(value_i64, value_i64);
        nk_u64_t unsigned_product = (nk_u64_t)product;
        sumsq = nk_u64_saturating_add_serial(sumsq, unsigned_product);
    }
    // Clamp 128-bit sum to i64 range
    nk_i64_t sum_lower_signed = (nk_i64_t)sum_lower;
    if (sum_upper == (sum_lower_signed >> 63)) *sum_ptr = sum_lower_signed;
    else if (sum_upper >= 0) *sum_ptr = NK_I64_MAX;
    else *sum_ptr = NK_I64_MIN;
    *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_i32_neon_strided_(                     //
    nk_i32_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    uint64x2_t sum_lower_u64x2 = vdupq_n_u64(0);
    int64x2_t sum_upper_i64x2 = vdupq_n_s64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    int sumsq_overflow = 0;
    int64x2_t sign_bit_i64x2 = vdupq_n_s64((nk_i64_t)0x8000000000000000ULL);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x2_t loaded_i32x4x2 = vld2q_s32(data_ptr + idx * 2);
            int32x4_t data_i32x4 = loaded_i32x4x2.val[0];
            int64x2_t lo_i64x2 = vmovl_s32(vget_low_s32(data_i32x4));
            uint64x2_t before_u64x2 = sum_lower_u64x2;
            sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(lo_i64x2));
            int64x2_t result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
            int64x2_t before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(before_u64x2), sign_bit_i64x2);
            uint64x2_t carry_u64x2 = vcgtq_s64(before_biased_i64x2, result_biased_i64x2);
            sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry_u64x2));
            sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, vshrq_n_s64(lo_i64x2, 63));
            int64x2_t hi_i64x2 = vmovl_high_s32(data_i32x4);
            before_u64x2 = sum_lower_u64x2;
            sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(hi_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(before_u64x2), sign_bit_i64x2);
            carry_u64x2 = vcgtq_s64(before_biased_i64x2, result_biased_i64x2);
            sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry_u64x2));
            sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, vshrq_n_s64(hi_i64x2, 63));
            int64x2_t squares_lo_i64x2 = vmull_s32(vget_low_s32(data_i32x4), vget_low_s32(data_i32x4));
            int64x2_t squares_hi_i64x2 = vmull_high_s32(data_i32x4, data_i32x4);
            uint64x2_t sq_before_u64x2 = sumsq_u64x2;
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vreinterpretq_u64_s64(squares_lo_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sq_before_u64x2), sign_bit_i64x2);
            sumsq_overflow |=
                (vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2)), 0) |
                 vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2)), 1));
            sq_before_u64x2 = sumsq_u64x2;
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vreinterpretq_u64_s64(squares_hi_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sq_before_u64x2), sign_bit_i64x2);
            sumsq_overflow |=
                (vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2)), 0) |
                 vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2)), 1));
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x3_t loaded_i32x4x3 = vld3q_s32(data_ptr + idx * 3);
            int32x4_t data_i32x4 = loaded_i32x4x3.val[0];
            int64x2_t lo_i64x2 = vmovl_s32(vget_low_s32(data_i32x4));
            uint64x2_t before_u64x2 = sum_lower_u64x2;
            sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(lo_i64x2));
            int64x2_t result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
            int64x2_t before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(before_u64x2), sign_bit_i64x2);
            uint64x2_t carry_u64x2 = vcgtq_s64(before_biased_i64x2, result_biased_i64x2);
            sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry_u64x2));
            sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, vshrq_n_s64(lo_i64x2, 63));
            int64x2_t hi_i64x2 = vmovl_high_s32(data_i32x4);
            before_u64x2 = sum_lower_u64x2;
            sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(hi_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(before_u64x2), sign_bit_i64x2);
            carry_u64x2 = vcgtq_s64(before_biased_i64x2, result_biased_i64x2);
            sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry_u64x2));
            sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, vshrq_n_s64(hi_i64x2, 63));
            int64x2_t squares_lo_i64x2 = vmull_s32(vget_low_s32(data_i32x4), vget_low_s32(data_i32x4));
            int64x2_t squares_hi_i64x2 = vmull_high_s32(data_i32x4, data_i32x4);
            uint64x2_t sq_before_u64x2 = sumsq_u64x2;
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vreinterpretq_u64_s64(squares_lo_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sq_before_u64x2), sign_bit_i64x2);
            sumsq_overflow |=
                (vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2)), 0) |
                 vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2)), 1));
            sq_before_u64x2 = sumsq_u64x2;
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vreinterpretq_u64_s64(squares_hi_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sq_before_u64x2), sign_bit_i64x2);
            sumsq_overflow |=
                (vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2)), 0) |
                 vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2)), 1));
        }
    }
    else {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x4_t loaded_i32x4x4 = vld4q_s32(data_ptr + idx * 4);
            int32x4_t data_i32x4 = loaded_i32x4x4.val[0];
            int64x2_t lo_i64x2 = vmovl_s32(vget_low_s32(data_i32x4));
            uint64x2_t before_u64x2 = sum_lower_u64x2;
            sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(lo_i64x2));
            int64x2_t result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
            int64x2_t before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(before_u64x2), sign_bit_i64x2);
            uint64x2_t carry_u64x2 = vcgtq_s64(before_biased_i64x2, result_biased_i64x2);
            sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry_u64x2));
            sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, vshrq_n_s64(lo_i64x2, 63));
            int64x2_t hi_i64x2 = vmovl_high_s32(data_i32x4);
            before_u64x2 = sum_lower_u64x2;
            sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(hi_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(before_u64x2), sign_bit_i64x2);
            carry_u64x2 = vcgtq_s64(before_biased_i64x2, result_biased_i64x2);
            sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry_u64x2));
            sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, vshrq_n_s64(hi_i64x2, 63));
            int64x2_t squares_lo_i64x2 = vmull_s32(vget_low_s32(data_i32x4), vget_low_s32(data_i32x4));
            int64x2_t squares_hi_i64x2 = vmull_high_s32(data_i32x4, data_i32x4);
            uint64x2_t sq_before_u64x2 = sumsq_u64x2;
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vreinterpretq_u64_s64(squares_lo_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sq_before_u64x2), sign_bit_i64x2);
            sumsq_overflow |=
                (vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2)), 0) |
                 vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2)), 1));
            sq_before_u64x2 = sumsq_u64x2;
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vreinterpretq_u64_s64(squares_hi_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sq_before_u64x2), sign_bit_i64x2);
            sumsq_overflow |=
                (vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2)), 0) |
                 vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2)), 1));
        }
    }
    nk_u64_t sumsq;
    if (sumsq_overflow) sumsq = NK_U64_MAX;
    else sumsq = nk_reduce_sadd_u64x2_neon_(sumsq_u64x2);
    nk_b128_vec_t lower_vec, upper_vec;
    lower_vec.u64x2 = sum_lower_u64x2;
    upper_vec.i64x2 = sum_upper_i64x2;
    nk_u64_t sum_lower = 0;
    nk_i64_t sum_upper = 0;
    nk_u64_t sum_before = sum_lower;
    sum_lower += lower_vec.u64s[0], sum_upper += (sum_lower < sum_before) + upper_vec.i64s[0];
    sum_before = sum_lower;
    sum_lower += lower_vec.u64s[1], sum_upper += (sum_lower < sum_before) + upper_vec.i64s[1];
    for (; idx < count; ++idx) {
        nk_i64_t val = (nk_i64_t) * (data_ptr + idx * stride_elements);
        sum_before = sum_lower;
        sum_lower += (nk_u64_t)val;
        if (sum_lower < sum_before) sum_upper++;
        sum_upper += (val >> 63);
        nk_i64_t product = nk_i64_saturating_mul_serial(val, val);
        nk_u64_t unsigned_product = (nk_u64_t)product;
        sumsq = nk_u64_saturating_add_serial(sumsq, unsigned_product);
    }
    nk_i64_t sum_lower_signed = (nk_i64_t)sum_lower;
    if (sum_upper == (sum_lower_signed >> 63)) *sum_ptr = sum_lower_signed;
    else if (sum_upper >= 0) *sum_ptr = NK_I64_MAX;
    else *sum_ptr = NK_I64_MIN;
    *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i32_neon(                             //
    nk_i32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_i32_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_i32_neon_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_i32_neon_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_i32_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_i32_neon_contiguous_( //
    nk_i32_t const *data_ptr, nk_size_t count,          //
    nk_i32_t *min_value_ptr, nk_size_t *min_index_ptr,  //
    nk_i32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    int32x4_t min_i32x4 = vdupq_n_s32(NK_I32_MAX), max_i32x4 = vdupq_n_s32(NK_I32_MIN);
    uint32x4_t min_iter_u32x4 = vdupq_n_u32(0), max_iter_u32x4 = vdupq_n_u32(0);
    uint32x4_t iter_u32x4 = vdupq_n_u32(0), one_u32x4 = vdupq_n_u32(1);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        int32x4_t data_i32x4 = vld1q_s32(data_ptr + idx);
        uint32x4_t less_u32x4 = vcltq_s32(data_i32x4, min_i32x4);
        uint32x4_t greater_u32x4 = vcgtq_s32(data_i32x4, max_i32x4);
        min_i32x4 = vbslq_s32(less_u32x4, data_i32x4, min_i32x4);
        max_i32x4 = vbslq_s32(greater_u32x4, data_i32x4, max_i32x4);
        min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
        max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
        iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b32x4_serial_(data_ptr + idx, &tail_vec, remaining);
        uint32x4_t lane_indices_u32x4 = vcombine_u32(vreinterpret_u32_u64(vcreate_u64(0x0000000100000000ULL)),
                                                     vreinterpret_u32_u64(vcreate_u64(0x0000000300000002ULL)));
        uint32x4_t valid_u32x4 = vcltq_u32(lane_indices_u32x4, vdupq_n_u32((uint32_t)remaining));
        int32x4_t data_min_i32x4 = vbslq_s32(valid_u32x4, tail_vec.i32x4, vdupq_n_s32(NK_I32_MAX));
        int32x4_t data_max_i32x4 = vbslq_s32(valid_u32x4, tail_vec.i32x4, vdupq_n_s32(NK_I32_MIN));
        uint32x4_t less_u32x4 = vcltq_s32(data_min_i32x4, min_i32x4);
        uint32x4_t greater_u32x4 = vcgtq_s32(data_max_i32x4, max_i32x4);
        min_i32x4 = vbslq_s32(less_u32x4, data_min_i32x4, min_i32x4);
        max_i32x4 = vbslq_s32(greater_u32x4, data_max_i32x4, max_i32x4);
        min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
        max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
    }
    nk_i32_t min_value = vminvq_s32(min_i32x4), max_value = vmaxvq_s32(max_i32x4);
    uint32x4_t min_value_match_u32x4 = vceqq_s32(min_i32x4, vdupq_n_s32(min_value));
    uint32x4_t masked_min_iter_u32x4 = vbslq_u32(min_value_match_u32x4, min_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_min_cycle = vminvq_u32(masked_min_iter_u32x4);
    uint32x4_t max_value_match_u32x4 = vceqq_s32(max_i32x4, vdupq_n_s32(max_value));
    uint32x4_t masked_max_iter_u32x4 = vbslq_u32(max_value_match_u32x4, max_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_max_cycle = vminvq_u32(masked_max_iter_u32x4);
    uint32x4_t lane_indices_u32x4 = vcombine_u32(vreinterpret_u32_u64(vcreate_u64(0x0000000100000000ULL)),
                                                 vreinterpret_u32_u64(vcreate_u64(0x0000000300000002ULL)));
    uint32x4_t min_cycle_match_u32x4 = vceqq_u32(min_iter_u32x4, vdupq_n_u32(earliest_min_cycle));
    uint32x4_t min_both_match_u32x4 = vandq_u32(min_value_match_u32x4, min_cycle_match_u32x4);
    uint32x4_t min_masked_lanes_u32x4 = vbslq_u32(min_both_match_u32x4, lane_indices_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t min_lane_offset = vminvq_u32(min_masked_lanes_u32x4);
    nk_size_t min_idx = (nk_size_t)earliest_min_cycle * 4 + (nk_size_t)min_lane_offset;
    uint32x4_t max_cycle_match_u32x4 = vceqq_u32(max_iter_u32x4, vdupq_n_u32(earliest_max_cycle));
    uint32x4_t max_both_match_u32x4 = vandq_u32(max_value_match_u32x4, max_cycle_match_u32x4);
    uint32x4_t max_masked_lanes_u32x4 = vbslq_u32(max_both_match_u32x4, lane_indices_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t max_lane_offset = vminvq_u32(max_masked_lanes_u32x4);
    nk_size_t max_idx = (nk_size_t)earliest_max_cycle * 4 + (nk_size_t)max_lane_offset;
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_i32_neon_strided_(                      //
    nk_i32_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_i32_t *min_value_ptr, nk_size_t *min_index_ptr,                    //
    nk_i32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    int32x4_t min_i32x4 = vdupq_n_s32(NK_I32_MAX), max_i32x4 = vdupq_n_s32(NK_I32_MIN);
    uint32x4_t min_iter_u32x4 = vdupq_n_u32(0), max_iter_u32x4 = vdupq_n_u32(0);
    uint32x4_t iter_u32x4 = vdupq_n_u32(0), one_u32x4 = vdupq_n_u32(1);
    uint32x4_t lane_indices_u32x4 = vcombine_u32(vreinterpret_u32_u64(vcreate_u64(0x0000000100000000ULL)),
                                                 vreinterpret_u32_u64(vcreate_u64(0x0000000300000002ULL)));
    nk_size_t idx = 0;
    int32x4_t data_for_min_i32x4, data_for_max_i32x4;

nk_reduce_minmax_i32_neon_cycle:
    if (stride_elements == 2 && idx + 4 <= count) {
        int32x4x2_t loaded = vld2q_s32(data_ptr + idx * 2);
        data_for_min_i32x4 = loaded.val[0];
        data_for_max_i32x4 = loaded.val[0];
        idx += 4;
    }
    else if (stride_elements == 3 && idx + 4 <= count) {
        int32x4x3_t loaded = vld3q_s32(data_ptr + idx * 3);
        data_for_min_i32x4 = loaded.val[0];
        data_for_max_i32x4 = loaded.val[0];
        idx += 4;
    }
    else if (stride_elements == 4 && idx + 4 <= count) {
        int32x4x4_t loaded = vld4q_s32(data_ptr + idx * 4);
        data_for_min_i32x4 = loaded.val[0];
        data_for_max_i32x4 = loaded.val[0];
        idx += 4;
    }
    else if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_strided_load_b32x4_serial_(data_ptr + idx * stride_elements, stride_elements, &tail_vec, count - idx);
        uint32x4_t valid_u32x4 = vcltq_u32(lane_indices_u32x4, vdupq_n_u32((uint32_t)(count - idx)));
        data_for_min_i32x4 = vbslq_s32(valid_u32x4, tail_vec.i32x4, min_i32x4);
        data_for_max_i32x4 = vbslq_s32(valid_u32x4, tail_vec.i32x4, max_i32x4);
        idx = count;
    }
    else {
        nk_i32_t min_value = vminvq_s32(min_i32x4), max_value = vmaxvq_s32(max_i32x4);
        uint32x4_t min_value_match_u32x4 = vceqq_s32(min_i32x4, vdupq_n_s32(min_value));
        uint32x4_t masked_min_iter_u32x4 = vbslq_u32(min_value_match_u32x4, min_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
        nk_u32_t earliest_min_cycle = vminvq_u32(masked_min_iter_u32x4);
        uint32x4_t max_value_match_u32x4 = vceqq_s32(max_i32x4, vdupq_n_s32(max_value));
        uint32x4_t masked_max_iter_u32x4 = vbslq_u32(max_value_match_u32x4, max_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
        nk_u32_t earliest_max_cycle = vminvq_u32(masked_max_iter_u32x4);
        uint32x4_t min_cycle_match_u32x4 = vceqq_u32(min_iter_u32x4, vdupq_n_u32(earliest_min_cycle));
        uint32x4_t min_both_match_u32x4 = vandq_u32(min_value_match_u32x4, min_cycle_match_u32x4);
        uint32x4_t min_masked_lanes_u32x4 = vbslq_u32(min_both_match_u32x4, lane_indices_u32x4,
                                                      vdupq_n_u32(NK_U32_MAX));
        nk_u32_t min_lane_offset = vminvq_u32(min_masked_lanes_u32x4);
        nk_size_t min_idx = (nk_size_t)earliest_min_cycle * 4 + (nk_size_t)min_lane_offset;
        uint32x4_t max_cycle_match_u32x4 = vceqq_u32(max_iter_u32x4, vdupq_n_u32(earliest_max_cycle));
        uint32x4_t max_both_match_u32x4 = vandq_u32(max_value_match_u32x4, max_cycle_match_u32x4);
        uint32x4_t max_masked_lanes_u32x4 = vbslq_u32(max_both_match_u32x4, lane_indices_u32x4,
                                                      vdupq_n_u32(NK_U32_MAX));
        nk_u32_t max_lane_offset = vminvq_u32(max_masked_lanes_u32x4);
        nk_size_t max_idx = (nk_size_t)earliest_max_cycle * 4 + (nk_size_t)max_lane_offset;
        *min_value_ptr = min_value, *min_index_ptr = min_idx;
        *max_value_ptr = max_value, *max_index_ptr = max_idx;
        return;
    }

    // Shared update body
    uint32x4_t less_u32x4 = vcltq_s32(data_for_min_i32x4, min_i32x4);
    uint32x4_t greater_u32x4 = vcgtq_s32(data_for_max_i32x4, max_i32x4);
    min_i32x4 = vbslq_s32(less_u32x4, data_for_min_i32x4, min_i32x4);
    max_i32x4 = vbslq_s32(greater_u32x4, data_for_max_i32x4, max_i32x4);
    min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
    max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
    iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
    goto nk_reduce_minmax_i32_neon_cycle;
}

NK_PUBLIC void nk_reduce_minmax_i32_neon(                              //
    nk_i32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_I32_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I32_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i32_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)NK_U32_MAX * 4) {
        nk_size_t left_count = count / 2;
        nk_i32_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_i32_neon(data_ptr, left_count, stride_bytes, &left_min_value, &left_min_index, &left_max_value,
                                  &left_max_index);
        nk_reduce_minmax_i32_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                  &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (right_min_value < left_min_value)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (right_max_value > left_max_value)
            *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i32_neon_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else if (stride_elements <= 4)
        nk_reduce_minmax_i32_neon_strided_(data_ptr, count, stride_elements, min_value_ptr, min_index_ptr,
                                           max_value_ptr, max_index_ptr);
    else
        nk_reduce_minmax_i32_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_u32_neon_contiguous_( //
    nk_u32_t const *data_ptr, nk_size_t count,           //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        uint32x4_t data_u32x4 = vld1q_u32(data_ptr + idx);
        // Widen u32 -> u64 and accumulate sum
        sum_u64x2 = vaddq_u64(sum_u64x2, vmovl_u32(vget_low_u32(data_u32x4)));
        sum_u64x2 = vaddq_u64(sum_u64x2, vmovl_high_u32(data_u32x4));
        // Sumsq: widening multiply u32*u32 -> u64, saturating add
        uint64x2_t sq_lo = vmull_u32(vget_low_u32(data_u32x4), vget_low_u32(data_u32x4));
        uint64x2_t sq_hi = vmull_high_u32(data_u32x4, data_u32x4);
        sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, sq_lo);
        sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, sq_hi);
    }
    nk_u64_t sum = vgetq_lane_u64(sum_u64x2, 0) + vgetq_lane_u64(sum_u64x2, 1);
    nk_u64_t sumsq = nk_reduce_sadd_u64x2_neon_(sumsq_u64x2);
    for (; idx < count; ++idx) {
        nk_u64_t value = (nk_u64_t)data_ptr[idx];
        sum += value;
        nk_u64_t product = value * value;
        sumsq = nk_u64_saturating_add_serial(sumsq, product);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_u32_neon_strided_(                     //
    nk_u32_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x2_t loaded_u32x4x2 = vld2q_u32(data_ptr + idx * 2);
            uint32x4_t data_u32x4 = loaded_u32x4x2.val[0];
            sum_u64x2 = vaddq_u64(sum_u64x2, vmovl_u32(vget_low_u32(data_u32x4)));
            sum_u64x2 = vaddq_u64(sum_u64x2, vmovl_high_u32(data_u32x4));
            uint64x2_t squares_lo_u64x2 = vmull_u32(vget_low_u32(data_u32x4), vget_low_u32(data_u32x4));
            uint64x2_t squares_hi_u64x2 = vmull_high_u32(data_u32x4, data_u32x4);
            sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, squares_lo_u64x2);
            sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, squares_hi_u64x2);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x3_t loaded_u32x4x3 = vld3q_u32(data_ptr + idx * 3);
            uint32x4_t data_u32x4 = loaded_u32x4x3.val[0];
            sum_u64x2 = vaddq_u64(sum_u64x2, vmovl_u32(vget_low_u32(data_u32x4)));
            sum_u64x2 = vaddq_u64(sum_u64x2, vmovl_high_u32(data_u32x4));
            uint64x2_t squares_lo_u64x2 = vmull_u32(vget_low_u32(data_u32x4), vget_low_u32(data_u32x4));
            uint64x2_t squares_hi_u64x2 = vmull_high_u32(data_u32x4, data_u32x4);
            sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, squares_lo_u64x2);
            sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, squares_hi_u64x2);
        }
    }
    else {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x4_t loaded_u32x4x4 = vld4q_u32(data_ptr + idx * 4);
            uint32x4_t data_u32x4 = loaded_u32x4x4.val[0];
            sum_u64x2 = vaddq_u64(sum_u64x2, vmovl_u32(vget_low_u32(data_u32x4)));
            sum_u64x2 = vaddq_u64(sum_u64x2, vmovl_high_u32(data_u32x4));
            uint64x2_t squares_lo_u64x2 = vmull_u32(vget_low_u32(data_u32x4), vget_low_u32(data_u32x4));
            uint64x2_t squares_hi_u64x2 = vmull_high_u32(data_u32x4, data_u32x4);
            sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, squares_lo_u64x2);
            sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, squares_hi_u64x2);
        }
    }
    nk_u64_t sum = vgetq_lane_u64(sum_u64x2, 0) + vgetq_lane_u64(sum_u64x2, 1);
    nk_u64_t sumsq = nk_reduce_sadd_u64x2_neon_(sumsq_u64x2);
    for (; idx < count; ++idx) {
        nk_u64_t val = (nk_u64_t) * (data_ptr + idx * stride_elements);
        sum += val;
        nk_u64_t product = val * val;
        sumsq = nk_u64_saturating_add_serial(sumsq, product);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u32_neon(                             //
    nk_u32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_u32_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 4) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u32_neon(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u32_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                   &right_sum, &right_sumsq);
        *sum_ptr = nk_u64_saturating_add_serial(left_sum, right_sum);
        *sumsq_ptr = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u32_neon_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_u32_neon_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_u32_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_u32_neon_contiguous_( //
    nk_u32_t const *data_ptr, nk_size_t count,          //
    nk_u32_t *min_value_ptr, nk_size_t *min_index_ptr,  //
    nk_u32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    uint32x4_t min_u32x4 = vdupq_n_u32(NK_U32_MAX), max_u32x4 = vdupq_n_u32(0);
    uint32x4_t min_iter_u32x4 = vdupq_n_u32(0), max_iter_u32x4 = vdupq_n_u32(0);
    uint32x4_t iter_u32x4 = vdupq_n_u32(0), one_u32x4 = vdupq_n_u32(1);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        uint32x4_t data_u32x4 = vld1q_u32(data_ptr + idx);
        uint32x4_t less_u32x4 = vcltq_u32(data_u32x4, min_u32x4);
        uint32x4_t greater_u32x4 = vcgtq_u32(data_u32x4, max_u32x4);
        min_u32x4 = vbslq_u32(less_u32x4, data_u32x4, min_u32x4);
        max_u32x4 = vbslq_u32(greater_u32x4, data_u32x4, max_u32x4);
        min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
        max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
        iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b32x4_serial_(data_ptr + idx, &tail_vec, remaining);
        uint32x4_t lane_indices_u32x4 = vcombine_u32(vreinterpret_u32_u64(vcreate_u64(0x0000000100000000ULL)),
                                                     vreinterpret_u32_u64(vcreate_u64(0x0000000300000002ULL)));
        uint32x4_t valid_u32x4 = vcltq_u32(lane_indices_u32x4, vdupq_n_u32((uint32_t)remaining));
        uint32x4_t data_min_u32x4 = vbslq_u32(valid_u32x4, tail_vec.u32x4, vdupq_n_u32(NK_U32_MAX));
        uint32x4_t data_max_u32x4 = vbslq_u32(valid_u32x4, tail_vec.u32x4, vdupq_n_u32(0));
        uint32x4_t less_u32x4 = vcltq_u32(data_min_u32x4, min_u32x4);
        uint32x4_t greater_u32x4 = vcgtq_u32(data_max_u32x4, max_u32x4);
        min_u32x4 = vbslq_u32(less_u32x4, data_min_u32x4, min_u32x4);
        max_u32x4 = vbslq_u32(greater_u32x4, data_max_u32x4, max_u32x4);
        min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
        max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
    }
    nk_u32_t min_value = vminvq_u32(min_u32x4), max_value = vmaxvq_u32(max_u32x4);
    uint32x4_t min_value_match_u32x4 = vceqq_u32(min_u32x4, vdupq_n_u32(min_value));
    uint32x4_t masked_min_iter_u32x4 = vbslq_u32(min_value_match_u32x4, min_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_min_cycle = vminvq_u32(masked_min_iter_u32x4);
    uint32x4_t max_value_match_u32x4 = vceqq_u32(max_u32x4, vdupq_n_u32(max_value));
    uint32x4_t masked_max_iter_u32x4 = vbslq_u32(max_value_match_u32x4, max_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_max_cycle = vminvq_u32(masked_max_iter_u32x4);
    uint32x4_t lane_indices_u32x4 = vcombine_u32(vreinterpret_u32_u64(vcreate_u64(0x0000000100000000ULL)),
                                                 vreinterpret_u32_u64(vcreate_u64(0x0000000300000002ULL)));
    uint32x4_t min_cycle_match_u32x4 = vceqq_u32(min_iter_u32x4, vdupq_n_u32(earliest_min_cycle));
    uint32x4_t min_both_match_u32x4 = vandq_u32(min_value_match_u32x4, min_cycle_match_u32x4);
    uint32x4_t min_masked_lanes_u32x4 = vbslq_u32(min_both_match_u32x4, lane_indices_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t min_lane_offset = vminvq_u32(min_masked_lanes_u32x4);
    nk_size_t min_idx = (nk_size_t)earliest_min_cycle * 4 + (nk_size_t)min_lane_offset;
    uint32x4_t max_cycle_match_u32x4 = vceqq_u32(max_iter_u32x4, vdupq_n_u32(earliest_max_cycle));
    uint32x4_t max_both_match_u32x4 = vandq_u32(max_value_match_u32x4, max_cycle_match_u32x4);
    uint32x4_t max_masked_lanes_u32x4 = vbslq_u32(max_both_match_u32x4, lane_indices_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t max_lane_offset = vminvq_u32(max_masked_lanes_u32x4);
    nk_size_t max_idx = (nk_size_t)earliest_max_cycle * 4 + (nk_size_t)max_lane_offset;
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_u32_neon_strided_(                      //
    nk_u32_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_u32_t *min_value_ptr, nk_size_t *min_index_ptr,                    //
    nk_u32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    uint32x4_t min_u32x4 = vdupq_n_u32(NK_U32_MAX), max_u32x4 = vdupq_n_u32(0);
    uint32x4_t min_iter_u32x4 = vdupq_n_u32(0), max_iter_u32x4 = vdupq_n_u32(0);
    uint32x4_t iter_u32x4 = vdupq_n_u32(0), one_u32x4 = vdupq_n_u32(1);
    uint32x4_t lane_indices_u32x4 = vcombine_u32(vreinterpret_u32_u64(vcreate_u64(0x0000000100000000ULL)),
                                                 vreinterpret_u32_u64(vcreate_u64(0x0000000300000002ULL)));
    nk_size_t idx = 0;
    uint32x4_t data_for_min_u32x4, data_for_max_u32x4;

nk_reduce_minmax_u32_neon_cycle:
    if (stride_elements == 2 && idx + 4 <= count) {
        uint32x4x2_t loaded = vld2q_u32(data_ptr + idx * 2);
        data_for_min_u32x4 = loaded.val[0];
        data_for_max_u32x4 = loaded.val[0];
        idx += 4;
    }
    else if (stride_elements == 3 && idx + 4 <= count) {
        uint32x4x3_t loaded = vld3q_u32(data_ptr + idx * 3);
        data_for_min_u32x4 = loaded.val[0];
        data_for_max_u32x4 = loaded.val[0];
        idx += 4;
    }
    else if (stride_elements == 4 && idx + 4 <= count) {
        uint32x4x4_t loaded = vld4q_u32(data_ptr + idx * 4);
        data_for_min_u32x4 = loaded.val[0];
        data_for_max_u32x4 = loaded.val[0];
        idx += 4;
    }
    else if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_strided_load_b32x4_serial_(data_ptr + idx * stride_elements, stride_elements, &tail_vec, count - idx);
        uint32x4_t valid_u32x4 = vcltq_u32(lane_indices_u32x4, vdupq_n_u32((uint32_t)(count - idx)));
        data_for_min_u32x4 = vbslq_u32(valid_u32x4, tail_vec.u32x4, min_u32x4);
        data_for_max_u32x4 = vbslq_u32(valid_u32x4, tail_vec.u32x4, max_u32x4);
        idx = count;
    }
    else {
        nk_u32_t min_value = vminvq_u32(min_u32x4), max_value = vmaxvq_u32(max_u32x4);
        uint32x4_t min_value_match_u32x4 = vceqq_u32(min_u32x4, vdupq_n_u32(min_value));
        uint32x4_t masked_min_iter_u32x4 = vbslq_u32(min_value_match_u32x4, min_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
        nk_u32_t earliest_min_cycle = vminvq_u32(masked_min_iter_u32x4);
        uint32x4_t max_value_match_u32x4 = vceqq_u32(max_u32x4, vdupq_n_u32(max_value));
        uint32x4_t masked_max_iter_u32x4 = vbslq_u32(max_value_match_u32x4, max_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
        nk_u32_t earliest_max_cycle = vminvq_u32(masked_max_iter_u32x4);
        uint32x4_t min_cycle_match_u32x4 = vceqq_u32(min_iter_u32x4, vdupq_n_u32(earliest_min_cycle));
        uint32x4_t min_both_match_u32x4 = vandq_u32(min_value_match_u32x4, min_cycle_match_u32x4);
        uint32x4_t min_masked_lanes_u32x4 = vbslq_u32(min_both_match_u32x4, lane_indices_u32x4,
                                                      vdupq_n_u32(NK_U32_MAX));
        nk_u32_t min_lane_offset = vminvq_u32(min_masked_lanes_u32x4);
        nk_size_t min_idx = (nk_size_t)earliest_min_cycle * 4 + (nk_size_t)min_lane_offset;
        uint32x4_t max_cycle_match_u32x4 = vceqq_u32(max_iter_u32x4, vdupq_n_u32(earliest_max_cycle));
        uint32x4_t max_both_match_u32x4 = vandq_u32(max_value_match_u32x4, max_cycle_match_u32x4);
        uint32x4_t max_masked_lanes_u32x4 = vbslq_u32(max_both_match_u32x4, lane_indices_u32x4,
                                                      vdupq_n_u32(NK_U32_MAX));
        nk_u32_t max_lane_offset = vminvq_u32(max_masked_lanes_u32x4);
        nk_size_t max_idx = (nk_size_t)earliest_max_cycle * 4 + (nk_size_t)max_lane_offset;
        *min_value_ptr = min_value, *min_index_ptr = min_idx;
        *max_value_ptr = max_value, *max_index_ptr = max_idx;
        return;
    }

    // Shared update body
    uint32x4_t less_u32x4 = vcltq_u32(data_for_min_u32x4, min_u32x4);
    uint32x4_t greater_u32x4 = vcgtq_u32(data_for_max_u32x4, max_u32x4);
    min_u32x4 = vbslq_u32(less_u32x4, data_for_min_u32x4, min_u32x4);
    max_u32x4 = vbslq_u32(greater_u32x4, data_for_max_u32x4, max_u32x4);
    min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
    max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
    iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
    goto nk_reduce_minmax_u32_neon_cycle;
}

NK_PUBLIC void nk_reduce_minmax_u32_neon(                              //
    nk_u32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_U32_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = 0, *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u32_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)NK_U32_MAX * 4) {
        nk_size_t left_count = count / 2;
        nk_u32_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_u32_neon(data_ptr, left_count, stride_bytes, &left_min_value, &left_min_index, &left_max_value,
                                  &left_max_index);
        nk_reduce_minmax_u32_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                  &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (right_min_value < left_min_value)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (right_max_value > left_max_value)
            *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u32_neon_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else if (stride_elements <= 4)
        nk_reduce_minmax_u32_neon_strided_(data_ptr, count, stride_elements, min_value_ptr, min_index_ptr,
                                           max_value_ptr, max_index_ptr);
    else
        nk_reduce_minmax_u32_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_i64_neon_contiguous_( //
    nk_i64_t const *data_ptr, nk_size_t count,           //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    uint64x2_t sum_lower_u64x2 = vdupq_n_u64(0);
    int64x2_t sum_upper_i64x2 = vdupq_n_s64(0);
    // NEON can still load/extract i64 vectors for sumsq via scalar nk_i64_smul_
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    int sumsq_overflow = 0;
    int64x2_t sign_bit_i64x2 = vdupq_n_s64((nk_i64_t)0x8000000000000000ULL);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        int64x2_t data_i64x2 = vld1q_s64(data_ptr + idx);
        // Sumsq via helper (scalar per-lane multiply)
        uint64x2_t sq = nk_i64_smul_sq_i64x2_neon_(data_i64x2);
        uint64x2_t sq_before = sumsq_u64x2;
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, sq);
        int64x2_t result_biased = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
        int64x2_t before_biased = veorq_s64(vreinterpretq_s64_u64(sq_before), sign_bit_i64x2);
        sumsq_overflow |= (vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased, result_biased)), 0) |
                           vgetq_lane_s64(vreinterpretq_s64_u64(vcgtq_s64(before_biased, result_biased)), 1));
        // Vectorized 128-bit carry-propagating sum
        uint64x2_t sum_before_u64x2 = sum_lower_u64x2;
        sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(data_i64x2));
        int64x2_t sb_biased = veorq_s64(vreinterpretq_s64_u64(sum_before_u64x2), sign_bit_i64x2);
        int64x2_t sr_biased = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
        uint64x2_t carry_u64x2 = vcgtq_s64(sb_biased, sr_biased);
        sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry_u64x2));
        int64x2_t sign_ext_i64x2 = vshrq_n_s64(data_i64x2, 63);
        sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, sign_ext_i64x2);
    }
    // Horizontal reduction of 2 lanes to scalar (sum_lower, sum_upper)
    nk_u64_t sum_lower = vgetq_lane_u64(sum_lower_u64x2, 0);
    nk_i64_t sum_upper = vgetq_lane_s64(sum_upper_i64x2, 0);
    {
        nk_u64_t before = sum_lower;
        sum_lower += vgetq_lane_u64(sum_lower_u64x2, 1);
        if (sum_lower < before) sum_upper++;
        sum_upper += vgetq_lane_s64(sum_upper_i64x2, 1);
    }
    nk_u64_t sumsq;
    if (sumsq_overflow) sumsq = NK_U64_MAX;
    else sumsq = nk_reduce_sadd_u64x2_neon_(sumsq_u64x2);
    for (; idx < count; ++idx) {
        nk_i64_t val = data_ptr[idx];
        nk_i64_t product = nk_i64_saturating_mul_serial(val, val);
        nk_u64_t unsigned_product = (nk_u64_t)product;
        sumsq = nk_u64_saturating_add_serial(sumsq, unsigned_product);
        nk_u64_t before = sum_lower;
        sum_lower += (nk_u64_t)val;
        if (sum_lower < before) sum_upper++;
        sum_upper += (val >> 63);
    }
    nk_i64_t sum_lower_signed = (nk_i64_t)sum_lower;
    if (sum_upper == (sum_lower_signed >> 63)) *sum_ptr = sum_lower_signed;
    else if (sum_upper >= 0) *sum_ptr = NK_I64_MAX;
    else *sum_ptr = NK_I64_MIN;
    *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i64_neon(                             //
    nk_i64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_i64_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_i64_neon_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_i64_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_i64_neon_contiguous_( //
    nk_i64_t const *data_ptr, nk_size_t count,          //
    nk_i64_t *min_value_ptr, nk_size_t *min_index_ptr,  //
    nk_i64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    int64x2_t min_i64x2 = vdupq_n_s64(NK_I64_MAX), max_i64x2 = vdupq_n_s64(NK_I64_MIN);
    uint64x2_t min_iter = vdupq_n_u64(0), max_iter = vdupq_n_u64(0);
    uint64x2_t iter = vdupq_n_u64(0), one = vdupq_n_u64(1);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        int64x2_t data_i64x2 = vld1q_s64(data_ptr + idx);
        uint64x2_t less_u64x2 = vcltq_s64(data_i64x2, min_i64x2);
        uint64x2_t greater_u64x2 = vcgtq_s64(data_i64x2, max_i64x2);
        min_i64x2 = vbslq_s64(less_u64x2, data_i64x2, min_i64x2);
        max_i64x2 = vbslq_s64(greater_u64x2, data_i64x2, max_i64x2);
        min_iter = vbslq_u64(less_u64x2, iter, min_iter);
        max_iter = vbslq_u64(greater_u64x2, iter, max_iter);
        iter = vaddq_u64(iter, one);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_indices_vec, max_indices_vec;
    min_values_vec.i64x2 = min_i64x2;
    min_indices_vec.u64x2 = min_iter;
    max_values_vec.i64x2 = max_i64x2;
    max_indices_vec.u64x2 = max_iter;
    nk_i64_t min_value, max_value;
    nk_size_t min_index, max_index;
    if (min_values_vec.i64s[0] <= min_values_vec.i64s[1])
        min_value = min_values_vec.i64s[0], min_index = (nk_size_t)min_indices_vec.u64s[0] * 2;
    else min_value = min_values_vec.i64s[1], min_index = (nk_size_t)min_indices_vec.u64s[1] * 2 + 1;
    if (max_values_vec.i64s[0] >= max_values_vec.i64s[1])
        max_value = max_values_vec.i64s[0], max_index = (nk_size_t)max_indices_vec.u64s[0] * 2;
    else max_value = max_values_vec.i64s[1], max_index = (nk_size_t)max_indices_vec.u64s[1] * 2 + 1;
    for (; idx < count; ++idx) {
        nk_i64_t val = data_ptr[idx];
        if (val < min_value) min_value = val, min_index = idx;
        if (val > max_value) max_value = val, max_index = idx;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_index;
    *max_value_ptr = max_value, *max_index_ptr = max_index;
}

NK_PUBLIC void nk_reduce_minmax_i64_neon(                              //
    nk_i64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_I64_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I64_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i64_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_i64_neon_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else
        nk_reduce_minmax_i64_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_u64_neon_contiguous_( //
    nk_u64_t const *data_ptr, nk_size_t count,           //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        uint64x2_t data_u64x2 = vld1q_u64(data_ptr + idx);
        sum_u64x2 = vqaddq_u64(sum_u64x2, data_u64x2);
        uint64x2_t sq = nk_u64_smul_sq_u64x2_neon_(data_u64x2);
        sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, sq);
    }
    nk_u64_t sum = nk_reduce_sadd_u64x2_neon_(sum_u64x2);
    nk_u64_t sumsq = nk_reduce_sadd_u64x2_neon_(sumsq_u64x2);
    for (; idx < count; ++idx) {
        nk_u64_t val = data_ptr[idx];
        sum = nk_u64_saturating_add_serial(sum, val);
        nk_u64_t product = nk_u64_saturating_mul_serial(val, val);
        sumsq = nk_u64_saturating_add_serial(sumsq, product);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u64_neon(                             //
    nk_u64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_u64_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_u64_neon_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_u64_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_u64_neon_contiguous_( //
    nk_u64_t const *data_ptr, nk_size_t count,          //
    nk_u64_t *min_value_ptr, nk_size_t *min_index_ptr,  //
    nk_u64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    uint64x2_t min_u64x2 = vdupq_n_u64(NK_U64_MAX), max_u64x2 = vdupq_n_u64(0);
    uint64x2_t min_iter = vdupq_n_u64(0), max_iter = vdupq_n_u64(0);
    uint64x2_t iter = vdupq_n_u64(0), one = vdupq_n_u64(1);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        uint64x2_t data_u64x2 = vld1q_u64(data_ptr + idx);
        uint64x2_t less_u64x2 = vcltq_u64(data_u64x2, min_u64x2);
        uint64x2_t greater_u64x2 = vcgtq_u64(data_u64x2, max_u64x2);
        min_u64x2 = vbslq_u64(less_u64x2, data_u64x2, min_u64x2);
        max_u64x2 = vbslq_u64(greater_u64x2, data_u64x2, max_u64x2);
        min_iter = vbslq_u64(less_u64x2, iter, min_iter);
        max_iter = vbslq_u64(greater_u64x2, iter, max_iter);
        iter = vaddq_u64(iter, one);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_indices_vec, max_indices_vec;
    min_values_vec.u64x2 = min_u64x2;
    min_indices_vec.u64x2 = min_iter;
    max_values_vec.u64x2 = max_u64x2;
    max_indices_vec.u64x2 = max_iter;
    nk_u64_t min_value, max_value;
    nk_size_t min_index, max_index;
    if (min_values_vec.u64s[0] <= min_values_vec.u64s[1])
        min_value = min_values_vec.u64s[0], min_index = (nk_size_t)min_indices_vec.u64s[0] * 2;
    else min_value = min_values_vec.u64s[1], min_index = (nk_size_t)min_indices_vec.u64s[1] * 2 + 1;
    if (max_values_vec.u64s[0] >= max_values_vec.u64s[1])
        max_value = max_values_vec.u64s[0], max_index = (nk_size_t)max_indices_vec.u64s[0] * 2;
    else max_value = max_values_vec.u64s[1], max_index = (nk_size_t)max_indices_vec.u64s[1] * 2 + 1;
    for (; idx < count; ++idx) {
        nk_u64_t val = data_ptr[idx];
        if (val < min_value) min_value = val, min_index = idx;
        if (val > max_value) max_value = val, max_index = idx;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_index;
    *max_value_ptr = max_value, *max_index_ptr = max_index;
}

NK_PUBLIC void nk_reduce_minmax_u64_neon(                              //
    nk_u64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_U64_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = 0, *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u64_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_u64_neon_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else
        nk_reduce_minmax_u64_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

/** @brief Convert 16 raw FP6 (e2m3/e3m2) sign-magnitude bytes to unsigned-comparable bytes.
 *  FP6: sign bit 5, 5-bit magnitude. Positive maps to [0x20..0x3F], negative to [0x00..0x1F]. */
NK_INTERNAL uint8x16_t nk_fp6x16_to_comparable_neon_(uint8x16_t raw_u8x16) {
    uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
    uint8x16_t sign_mask_u8x16 = vdupq_n_u8(0x20);
    uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, sign_mask_u8x16);
    uint8x16_t positive_u8x16 = vorrq_u8(magnitude_u8x16, sign_mask_u8x16);
    uint8x16_t negative_u8x16 = vsubq_u8(vdupq_n_u8(0x1F), magnitude_u8x16);
    return vbslq_u8(is_negative_u8x16, negative_u8x16, positive_u8x16);
}

/** @brief Convert a single comparable byte back to raw FP6 sign-magnitude byte. */
NK_INTERNAL nk_u8_t nk_comparable_to_fp6_(nk_u8_t comparable) {
    if (comparable >= 0x20) return comparable ^ 0x20; // was positive
    else return (0x1F - comparable) | 0x20;           // was negative
}

/** @brief Convert 16 raw FP8 (e4m3/e5m2) sign-magnitude bytes to unsigned-comparable bytes. */
NK_INTERNAL uint8x16_t nk_fp8x16_to_comparable_neon_(uint8x16_t raw_u8x16) {
    uint8x16_t sign_mask_u8x16 = vdupq_n_u8(0x80);
    uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, sign_mask_u8x16);
    uint8x16_t flip_positive_u8x16 = veorq_u8(raw_u8x16, sign_mask_u8x16);
    uint8x16_t flip_negative_u8x16 = vmvnq_u8(raw_u8x16);
    return vbslq_u8(is_negative_u8x16, flip_negative_u8x16, flip_positive_u8x16);
}

/** @brief Convert a single comparable byte back to raw FP8 sign-magnitude byte. */
NK_INTERNAL nk_u8_t nk_comparable_to_fp8_(nk_u8_t comparable) {
    if (comparable >= 0x80) return comparable ^ 0x80; // was positive
    else return ~comparable;                          // was negative
}

NK_INTERNAL void nk_reduce_moments_e2m3_neon_contiguous_( //
    nk_e2m3_t const *data_ptr, nk_size_t count,           //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    // VTBL LUT: maps 6-bit magnitude (0..31) to value×16 (unsigned), fits in u8
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
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data_ptr + idx));
        uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
        uint8x16_t unsigned_u8x16 = vqtbl2q_u8(lut_e2m3_x16, magnitude_u8x16);
        uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
        int8x16_t positive_i8x16 = vreinterpretq_s8_u8(unsigned_u8x16);
        int8x16_t negative_i8x16 = vnegq_s8(positive_i8x16);
        int8x16_t scaled_i8x16 = vbslq_s8(is_negative_u8x16, negative_i8x16, positive_i8x16);
        int16x8_t pairwise_i16x8 = vpaddlq_s8(scaled_i8x16);
        sum_i32x4 = vaddq_s32(sum_i32x4, vpaddlq_s16(pairwise_i16x8));
        int16x8_t squares_lo_i16x8 = vmull_s8(vget_low_s8(scaled_i8x16), vget_low_s8(scaled_i8x16));
        int16x8_t squares_hi_i16x8 = vmull_high_s8(scaled_i8x16, scaled_i8x16);
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_lo_i16x8))));
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_hi_i16x8))));
    }
    nk_i64_t sum = vaddlvq_s32(sum_i32x4);
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_f32_t value_f32;
        nk_e2m3_to_f32_serial(&data_ptr[idx], &value_f32);
        sum += (nk_i64_t)(value_f32 * 16.0f), sumsq += (nk_u64_t)(nk_i64_t)(value_f32 * value_f32 * 256.0f);
    }
    *sum_ptr = (nk_f32_t)sum / 16.0f, *sumsq_ptr = (nk_f32_t)sumsq / 256.0f;
}

NK_INTERNAL void nk_reduce_moments_e2m3_neon_strided_(                     //
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
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
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
            int16x8_t pairwise_i16x8 = vpaddlq_s8(scaled_i8x16);
            sum_i32x4 = vaddq_s32(sum_i32x4, vpaddlq_s16(pairwise_i16x8));
            int16x8_t squares_lo_i16x8 = vmull_s8(vget_low_s8(scaled_i8x16), vget_low_s8(scaled_i8x16));
            int16x8_t squares_hi_i16x8 = vmull_high_s8(scaled_i8x16, scaled_i8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_lo_i16x8))));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_hi_i16x8))));
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
            int16x8_t pairwise_i16x8 = vpaddlq_s8(scaled_i8x16);
            sum_i32x4 = vaddq_s32(sum_i32x4, vpaddlq_s16(pairwise_i16x8));
            int16x8_t squares_lo_i16x8 = vmull_s8(vget_low_s8(scaled_i8x16), vget_low_s8(scaled_i8x16));
            int16x8_t squares_hi_i16x8 = vmull_high_s8(scaled_i8x16, scaled_i8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_lo_i16x8))));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_hi_i16x8))));
        }
    }
    else {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x4_t loaded_u8x16x4 = vld4q_u8((nk_u8_t const *)(data_ptr + idx * 4));
            uint8x16_t raw_u8x16 = loaded_u8x16x4.val[0];
            uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
            uint8x16_t unsigned_u8x16 = vqtbl2q_u8(lut_e2m3_x16, magnitude_u8x16);
            uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
            int8x16_t positive_i8x16 = vreinterpretq_s8_u8(unsigned_u8x16);
            int8x16_t negative_i8x16 = vnegq_s8(positive_i8x16);
            int8x16_t scaled_i8x16 = vbslq_s8(is_negative_u8x16, negative_i8x16, positive_i8x16);
            int16x8_t pairwise_i16x8 = vpaddlq_s8(scaled_i8x16);
            sum_i32x4 = vaddq_s32(sum_i32x4, vpaddlq_s16(pairwise_i16x8));
            int16x8_t squares_lo_i16x8 = vmull_s8(vget_low_s8(scaled_i8x16), vget_low_s8(scaled_i8x16));
            int16x8_t squares_hi_i16x8 = vmull_high_s8(scaled_i8x16, scaled_i8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_lo_i16x8))));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_hi_i16x8))));
        }
    }
    nk_i64_t sum = vaddlvq_s32(sum_i32x4);
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_e2m3_to_f32_serial((nk_e2m3_t const *)(data_ptr + idx * stride_elements), &val);
        sum += (nk_i64_t)(val * 16.0f), sumsq += (nk_u64_t)(nk_i64_t)(val * val * 256.0f);
    }
    *sum_ptr = (nk_f32_t)sum / 16.0f, *sumsq_ptr = (nk_f32_t)sumsq / 256.0f;
}

NK_PUBLIC void nk_reduce_moments_e2m3_neon(                             //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    int aligned = (stride_bytes % sizeof(nk_e2m3_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e2m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e2m3_neon(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e2m3_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                    &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum, *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e2m3_neon_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_e2m3_neon_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e2m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e2m3_neon_contiguous_( //
    nk_e2m3_t const *data_ptr, nk_size_t count,          //
    nk_e2m3_t *min_value_ptr, nk_size_t *min_index_ptr,  //
    nk_e2m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    // Handle initial chunk — partial or full
    uint8x16_t first_comparable_u8x16;
    nk_size_t first_count = count < 16 ? count : 16;
    if (count < 16) {
        nk_b128_vec_t first_vec;
        nk_partial_load_b8x16_serial_(data_ptr, &first_vec, count);
        first_comparable_u8x16 = nk_fp6x16_to_comparable_neon_(first_vec.u8x16);
        // Mask invalid lanes: min gets 0xFF (won't be selected), max gets 0x00
        uint8x16_t lane_indices_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                    vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
        uint8x16_t valid_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((uint8_t)count));
        first_comparable_u8x16 = vbslq_u8(valid_u8x16, first_comparable_u8x16, vdupq_n_u8(0));
    }
    else {
        uint8x16_t first_raw_u8x16 = vld1q_u8((nk_u8_t const *)data_ptr);
        first_comparable_u8x16 = nk_fp6x16_to_comparable_neon_(first_raw_u8x16);
    }
    // For min: invalid lanes (0x00) should not win, so initialize min from masked data where invalid = 0xFF
    // For max: invalid lanes (0x00) should not win, which is already correct since 0x00 won't beat real data
    uint8x16_t lane_indices_init_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                     vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
    uint8x16_t valid_init_u8x16 = vcltq_u8(lane_indices_init_u8x16, vdupq_n_u8((uint8_t)first_count));
    uint8x16_t min_u8x16 = vbslq_u8(valid_init_u8x16, first_comparable_u8x16, vdupq_n_u8(0xFF));
    uint8x16_t max_u8x16 = first_comparable_u8x16; // invalid lanes are 0x00, safe for max
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(1), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = first_count;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data_ptr + idx));
        uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(raw_u8x16);
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
        uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(tail_vec.u8x16);
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
    *min_value_ptr = nk_comparable_to_fp6_(min_comparable), *min_index_ptr = min_idx;
    *max_value_ptr = nk_comparable_to_fp6_(max_comparable), *max_index_ptr = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_e2m3_neon_strided_(                      //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_e2m3_t *min_value_ptr, nk_size_t *min_index_ptr,                    //
    nk_e2m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    uint8x16_t min_u8x16 = vdupq_n_u8(0xFF), max_u8x16 = vdupq_n_u8(0);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    uint8x16_t lane_indices_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
    nk_size_t idx = 0;
    uint8x16_t data_for_min_u8x16, data_for_max_u8x16;

nk_reduce_minmax_e2m3_neon_cycle:
    if (stride_elements == 2 && idx + 16 <= count) {
        uint8x16x2_t loaded = vld2q_u8((nk_u8_t const *)(data_ptr + idx * 2));
        uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(loaded.val[0]);
        data_for_min_u8x16 = comparable_u8x16;
        data_for_max_u8x16 = comparable_u8x16;
        idx += 16;
    }
    else if (stride_elements == 3 && idx + 16 <= count) {
        uint8x16x3_t loaded = vld3q_u8((nk_u8_t const *)(data_ptr + idx * 3));
        uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(loaded.val[0]);
        data_for_min_u8x16 = comparable_u8x16;
        data_for_max_u8x16 = comparable_u8x16;
        idx += 16;
    }
    else if (stride_elements == 4 && idx + 16 <= count) {
        uint8x16x4_t loaded = vld4q_u8((nk_u8_t const *)(data_ptr + idx * 4));
        uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(loaded.val[0]);
        data_for_min_u8x16 = comparable_u8x16;
        data_for_max_u8x16 = comparable_u8x16;
        idx += 16;
    }
    else if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_strided_load_b8x16_serial_(data_ptr + idx * stride_elements, stride_elements, &tail_vec, count - idx);
        uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(tail_vec.u8x16);
        uint8x16_t valid_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((uint8_t)(count - idx)));
        data_for_min_u8x16 = vbslq_u8(valid_u8x16, comparable_u8x16, vdupq_n_u8(0xFF));
        data_for_max_u8x16 = vbslq_u8(valid_u8x16, comparable_u8x16, vdupq_n_u8(0x00));
        idx = count;
    }
    else {
        nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
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
        *min_value_ptr = nk_comparable_to_fp6_(min_comparable), *min_index_ptr = min_idx;
        *max_value_ptr = nk_comparable_to_fp6_(max_comparable), *max_index_ptr = max_idx;
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
    goto nk_reduce_minmax_e2m3_neon_cycle;
}

NK_PUBLIC void nk_reduce_minmax_e2m3_neon(                              //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e2m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    int aligned = (stride_bytes % sizeof(nk_e2m3_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_E2M3_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_E2M3_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_e2m3_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (count > (nk_size_t)256 * 16) {
        nk_size_t left_count = count / 2;
        nk_e2m3_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_e2m3_neon(data_ptr, left_count, stride_bytes, &left_min_value, &left_min_index,
                                   &left_max_value, &left_max_index);
        nk_reduce_minmax_e2m3_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                   &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (nk_e2m3_order_serial(right_min_value, left_min_value) < 0)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (nk_e2m3_order_serial(right_max_value, left_max_value) > 0)
            *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e2m3_neon_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                               max_index_ptr);
    else if (stride_elements <= 4)
        nk_reduce_minmax_e2m3_neon_strided_(data_ptr, count, stride_elements, min_value_ptr, min_index_ptr,
                                            max_value_ptr, max_index_ptr);
    else
        nk_reduce_minmax_e2m3_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_e3m2_neon_contiguous_( //
    nk_e3m2_t const *data_ptr, nk_size_t count,           //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    // VTBL LUT: maps 6-bit magnitude (0..31) to (value×16) low byte; max value×16 = 448 needs i16
    uint8x16x2_t lut_e3m2_lo;
    // table[0]: low bytes for magnitudes 0..15
    // 0x0706050403020100 → bytes [0..7]  = 0,1,2,3,4,5,6,7
    // 0x1C1814100E0C0A08 → bytes [8..15] = 8,10,12,14,16,20,24,28
    lut_e3m2_lo.val[0] = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                     vreinterpret_u8_u64(vcreate_u64(0x1C1814100E0C0A08ULL)));
    // table[1]: low bytes for magnitudes 16..31
    // 0x7060504038302820 → bytes [0..7]  = 32,40,48,56,64,80,96,112
    // 0xC0804000E0C0A080 → bytes [8..15] = 128,160,192,224,0,64,128,192
    lut_e3m2_lo.val[1] = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x7060504038302820ULL)),
                                     vreinterpret_u8_u64(vcreate_u64(0xC0804000E0C0A080ULL)));
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data_ptr + idx));
        uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
        uint8x16_t low_byte_u8x16 = vqtbl2q_u8(lut_e3m2_lo, magnitude_u8x16);
        uint8x16_t high_byte_u8x16 = vandq_u8(vcgtq_u8(magnitude_u8x16, vdupq_n_u8(27)), vdupq_n_u8(1));
        uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
        // Interleave low+high bytes into i16 values (two halves of 8 each)
        uint16x8_t unsigned_lo_u16x8 = vreinterpretq_u16_u8(vzip1q_u8(low_byte_u8x16, high_byte_u8x16));
        uint16x8_t unsigned_hi_u16x8 = vreinterpretq_u16_u8(vzip2q_u8(low_byte_u8x16, high_byte_u8x16));
        // Sign-extend the per-byte negative mask to per-i16 lanes
        int8x8_t is_negative_lo_i8x8 = vreinterpret_s8_u8(vget_low_u8(is_negative_u8x16));
        int8x8_t is_negative_hi_i8x8 = vreinterpret_s8_u8(vget_high_u8(is_negative_u8x16));
        uint16x8_t is_negative_lo_u16x8 = vreinterpretq_u16_s16(vmovl_s8(is_negative_lo_i8x8));
        uint16x8_t is_negative_hi_u16x8 = vreinterpretq_u16_s16(vmovl_s8(is_negative_hi_i8x8));
        // Apply sign via conditional negate
        int16x8_t positive_lo_i16x8 = vreinterpretq_s16_u16(unsigned_lo_u16x8);
        int16x8_t scaled_lo_i16x8 = vbslq_s16(is_negative_lo_u16x8, vnegq_s16(positive_lo_i16x8), positive_lo_i16x8);
        int16x8_t positive_hi_i16x8 = vreinterpretq_s16_u16(unsigned_hi_u16x8);
        int16x8_t scaled_hi_i16x8 = vbslq_s16(is_negative_hi_u16x8, vnegq_s16(positive_hi_i16x8), positive_hi_i16x8);
        // Sum: i16→i32 widening, accumulate in i32x4
        sum_i32x4 = vaddq_s32(sum_i32x4, vpaddlq_s16(scaled_lo_i16x8));
        sum_i32x4 = vaddq_s32(sum_i32x4, vpaddlq_s16(scaled_hi_i16x8));
        // Sumsq: vmull_s16→i32 (always positive as squares), widen to u64
        int32x4_t squares_lo_a_i32x4 = vmull_s16(vget_low_s16(scaled_lo_i16x8), vget_low_s16(scaled_lo_i16x8));
        int32x4_t squares_lo_b_i32x4 = vmull_high_s16(scaled_lo_i16x8, scaled_lo_i16x8);
        int32x4_t squares_hi_a_i32x4 = vmull_s16(vget_low_s16(scaled_hi_i16x8), vget_low_s16(scaled_hi_i16x8));
        int32x4_t squares_hi_b_i32x4 = vmull_high_s16(scaled_hi_i16x8, scaled_hi_i16x8);
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_a_i32x4)));
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_b_i32x4)));
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_a_i32x4)));
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_b_i32x4)));
    }
    nk_i64_t sum = vaddlvq_s32(sum_i32x4);
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_f32_t value_f32;
        nk_e3m2_to_f32_serial(&data_ptr[idx], &value_f32);
        sum += (nk_i64_t)(value_f32 * 16.0f), sumsq += (nk_u64_t)(nk_i64_t)(value_f32 * value_f32 * 256.0f);
    }
    *sum_ptr = (nk_f32_t)sum / 16.0f, *sumsq_ptr = (nk_f32_t)sumsq / 256.0f;
}

NK_INTERNAL void nk_reduce_moments_e3m2_neon_strided_(                     //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    uint8x16x2_t lut_e3m2_lo;
    // table[0]: low bytes for magnitudes 0..15
    // 0x0706050403020100 → bytes [0..7]  = 0,1,2,3,4,5,6,7
    // 0x1C1814100E0C0A08 → bytes [8..15] = 8,10,12,14,16,20,24,28
    lut_e3m2_lo.val[0] = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                     vreinterpret_u8_u64(vcreate_u64(0x1C1814100E0C0A08ULL)));
    // table[1]: low bytes for magnitudes 16..31
    // 0x7060504038302820 → bytes [0..7]  = 32,40,48,56,64,80,96,112
    // 0xC0804000E0C0A080 → bytes [8..15] = 128,160,192,224,0,64,128,192
    lut_e3m2_lo.val[1] = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x7060504038302820ULL)),
                                     vreinterpret_u8_u64(vcreate_u64(0xC0804000E0C0A080ULL)));
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x2_t loaded_u8x16x2 = vld2q_u8((nk_u8_t const *)(data_ptr + idx * 2));
            uint8x16_t raw_u8x16 = loaded_u8x16x2.val[0];
            uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
            uint8x16_t low_byte_u8x16 = vqtbl2q_u8(lut_e3m2_lo, magnitude_u8x16);
            uint8x16_t high_byte_u8x16 = vandq_u8(vcgtq_u8(magnitude_u8x16, vdupq_n_u8(27)), vdupq_n_u8(1));
            uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
            uint16x8_t unsigned_lo_u16x8 = vreinterpretq_u16_u8(vzip1q_u8(low_byte_u8x16, high_byte_u8x16));
            uint16x8_t unsigned_hi_u16x8 = vreinterpretq_u16_u8(vzip2q_u8(low_byte_u8x16, high_byte_u8x16));
            int8x8_t is_negative_lo_i8x8 = vreinterpret_s8_u8(vget_low_u8(is_negative_u8x16));
            int8x8_t is_negative_hi_i8x8 = vreinterpret_s8_u8(vget_high_u8(is_negative_u8x16));
            uint16x8_t is_negative_lo_u16x8 = vreinterpretq_u16_s16(vmovl_s8(is_negative_lo_i8x8));
            uint16x8_t is_negative_hi_u16x8 = vreinterpretq_u16_s16(vmovl_s8(is_negative_hi_i8x8));
            int16x8_t positive_lo_i16x8 = vreinterpretq_s16_u16(unsigned_lo_u16x8);
            int16x8_t scaled_lo_i16x8 = vbslq_s16(is_negative_lo_u16x8, vnegq_s16(positive_lo_i16x8),
                                                  positive_lo_i16x8);
            int16x8_t positive_hi_i16x8 = vreinterpretq_s16_u16(unsigned_hi_u16x8);
            int16x8_t scaled_hi_i16x8 = vbslq_s16(is_negative_hi_u16x8, vnegq_s16(positive_hi_i16x8),
                                                  positive_hi_i16x8);
            sum_i32x4 = vaddq_s32(sum_i32x4, vpaddlq_s16(scaled_lo_i16x8));
            sum_i32x4 = vaddq_s32(sum_i32x4, vpaddlq_s16(scaled_hi_i16x8));
            int32x4_t squares_lo_a_i32x4 = vmull_s16(vget_low_s16(scaled_lo_i16x8), vget_low_s16(scaled_lo_i16x8));
            int32x4_t squares_lo_b_i32x4 = vmull_high_s16(scaled_lo_i16x8, scaled_lo_i16x8);
            int32x4_t squares_hi_a_i32x4 = vmull_s16(vget_low_s16(scaled_hi_i16x8), vget_low_s16(scaled_hi_i16x8));
            int32x4_t squares_hi_b_i32x4 = vmull_high_s16(scaled_hi_i16x8, scaled_hi_i16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_a_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_b_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_a_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_b_i32x4)));
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x3_t loaded_u8x16x3 = vld3q_u8((nk_u8_t const *)(data_ptr + idx * 3));
            uint8x16_t raw_u8x16 = loaded_u8x16x3.val[0];
            uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
            uint8x16_t low_byte_u8x16 = vqtbl2q_u8(lut_e3m2_lo, magnitude_u8x16);
            uint8x16_t high_byte_u8x16 = vandq_u8(vcgtq_u8(magnitude_u8x16, vdupq_n_u8(27)), vdupq_n_u8(1));
            uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
            uint16x8_t unsigned_lo_u16x8 = vreinterpretq_u16_u8(vzip1q_u8(low_byte_u8x16, high_byte_u8x16));
            uint16x8_t unsigned_hi_u16x8 = vreinterpretq_u16_u8(vzip2q_u8(low_byte_u8x16, high_byte_u8x16));
            int8x8_t is_negative_lo_i8x8 = vreinterpret_s8_u8(vget_low_u8(is_negative_u8x16));
            int8x8_t is_negative_hi_i8x8 = vreinterpret_s8_u8(vget_high_u8(is_negative_u8x16));
            uint16x8_t is_negative_lo_u16x8 = vreinterpretq_u16_s16(vmovl_s8(is_negative_lo_i8x8));
            uint16x8_t is_negative_hi_u16x8 = vreinterpretq_u16_s16(vmovl_s8(is_negative_hi_i8x8));
            int16x8_t positive_lo_i16x8 = vreinterpretq_s16_u16(unsigned_lo_u16x8);
            int16x8_t scaled_lo_i16x8 = vbslq_s16(is_negative_lo_u16x8, vnegq_s16(positive_lo_i16x8),
                                                  positive_lo_i16x8);
            int16x8_t positive_hi_i16x8 = vreinterpretq_s16_u16(unsigned_hi_u16x8);
            int16x8_t scaled_hi_i16x8 = vbslq_s16(is_negative_hi_u16x8, vnegq_s16(positive_hi_i16x8),
                                                  positive_hi_i16x8);
            sum_i32x4 = vaddq_s32(sum_i32x4, vpaddlq_s16(scaled_lo_i16x8));
            sum_i32x4 = vaddq_s32(sum_i32x4, vpaddlq_s16(scaled_hi_i16x8));
            int32x4_t squares_lo_a_i32x4 = vmull_s16(vget_low_s16(scaled_lo_i16x8), vget_low_s16(scaled_lo_i16x8));
            int32x4_t squares_lo_b_i32x4 = vmull_high_s16(scaled_lo_i16x8, scaled_lo_i16x8);
            int32x4_t squares_hi_a_i32x4 = vmull_s16(vget_low_s16(scaled_hi_i16x8), vget_low_s16(scaled_hi_i16x8));
            int32x4_t squares_hi_b_i32x4 = vmull_high_s16(scaled_hi_i16x8, scaled_hi_i16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_a_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_b_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_a_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_b_i32x4)));
        }
    }
    else {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x4_t loaded_u8x16x4 = vld4q_u8((nk_u8_t const *)(data_ptr + idx * 4));
            uint8x16_t raw_u8x16 = loaded_u8x16x4.val[0];
            uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
            uint8x16_t low_byte_u8x16 = vqtbl2q_u8(lut_e3m2_lo, magnitude_u8x16);
            uint8x16_t high_byte_u8x16 = vandq_u8(vcgtq_u8(magnitude_u8x16, vdupq_n_u8(27)), vdupq_n_u8(1));
            uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
            uint16x8_t unsigned_lo_u16x8 = vreinterpretq_u16_u8(vzip1q_u8(low_byte_u8x16, high_byte_u8x16));
            uint16x8_t unsigned_hi_u16x8 = vreinterpretq_u16_u8(vzip2q_u8(low_byte_u8x16, high_byte_u8x16));
            int8x8_t is_negative_lo_i8x8 = vreinterpret_s8_u8(vget_low_u8(is_negative_u8x16));
            int8x8_t is_negative_hi_i8x8 = vreinterpret_s8_u8(vget_high_u8(is_negative_u8x16));
            uint16x8_t is_negative_lo_u16x8 = vreinterpretq_u16_s16(vmovl_s8(is_negative_lo_i8x8));
            uint16x8_t is_negative_hi_u16x8 = vreinterpretq_u16_s16(vmovl_s8(is_negative_hi_i8x8));
            int16x8_t positive_lo_i16x8 = vreinterpretq_s16_u16(unsigned_lo_u16x8);
            int16x8_t scaled_lo_i16x8 = vbslq_s16(is_negative_lo_u16x8, vnegq_s16(positive_lo_i16x8),
                                                  positive_lo_i16x8);
            int16x8_t positive_hi_i16x8 = vreinterpretq_s16_u16(unsigned_hi_u16x8);
            int16x8_t scaled_hi_i16x8 = vbslq_s16(is_negative_hi_u16x8, vnegq_s16(positive_hi_i16x8),
                                                  positive_hi_i16x8);
            sum_i32x4 = vaddq_s32(sum_i32x4, vpaddlq_s16(scaled_lo_i16x8));
            sum_i32x4 = vaddq_s32(sum_i32x4, vpaddlq_s16(scaled_hi_i16x8));
            int32x4_t squares_lo_a_i32x4 = vmull_s16(vget_low_s16(scaled_lo_i16x8), vget_low_s16(scaled_lo_i16x8));
            int32x4_t squares_lo_b_i32x4 = vmull_high_s16(scaled_lo_i16x8, scaled_lo_i16x8);
            int32x4_t squares_hi_a_i32x4 = vmull_s16(vget_low_s16(scaled_hi_i16x8), vget_low_s16(scaled_hi_i16x8));
            int32x4_t squares_hi_b_i32x4 = vmull_high_s16(scaled_hi_i16x8, scaled_hi_i16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_a_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_b_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_a_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_b_i32x4)));
        }
    }
    nk_i64_t sum = vaddlvq_s32(sum_i32x4);
    nk_u64_t sumsq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_e3m2_to_f32_serial((nk_e3m2_t const *)(data_ptr + idx * stride_elements), &val);
        sum += (nk_i64_t)(val * 16.0f), sumsq += (nk_u64_t)(nk_i64_t)(val * val * 256.0f);
    }
    *sum_ptr = (nk_f32_t)sum / 16.0f, *sumsq_ptr = (nk_f32_t)sumsq / 256.0f;
}

NK_PUBLIC void nk_reduce_moments_e3m2_neon(                             //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    int aligned = (stride_bytes % sizeof(nk_e3m2_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e3m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e3m2_neon(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e3m2_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                    &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum, *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e3m2_neon_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_e3m2_neon_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e3m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e3m2_neon_contiguous_( //
    nk_e3m2_t const *data_ptr, nk_size_t count,          //
    nk_e3m2_t *min_value_ptr, nk_size_t *min_index_ptr,  //
    nk_e3m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    // Handle initial chunk — partial or full
    uint8x16_t first_comparable_u8x16;
    nk_size_t first_count = count < 16 ? count : 16;
    if (count < 16) {
        nk_b128_vec_t first_vec;
        nk_partial_load_b8x16_serial_(data_ptr, &first_vec, count);
        first_comparable_u8x16 = nk_fp6x16_to_comparable_neon_(first_vec.u8x16);
        uint8x16_t lane_indices_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                    vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
        uint8x16_t valid_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((uint8_t)count));
        first_comparable_u8x16 = vbslq_u8(valid_u8x16, first_comparable_u8x16, vdupq_n_u8(0));
    }
    else {
        uint8x16_t first_raw_u8x16 = vld1q_u8((nk_u8_t const *)data_ptr);
        first_comparable_u8x16 = nk_fp6x16_to_comparable_neon_(first_raw_u8x16);
    }
    uint8x16_t lane_indices_init_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                     vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
    uint8x16_t valid_init_u8x16 = vcltq_u8(lane_indices_init_u8x16, vdupq_n_u8((uint8_t)first_count));
    uint8x16_t min_u8x16 = vbslq_u8(valid_init_u8x16, first_comparable_u8x16, vdupq_n_u8(0xFF));
    uint8x16_t max_u8x16 = first_comparable_u8x16;
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(1), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = first_count;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data_ptr + idx));
        uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(raw_u8x16);
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
        uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(tail_vec.u8x16);
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
    *min_value_ptr = nk_comparable_to_fp6_(min_comparable), *min_index_ptr = min_idx;
    *max_value_ptr = nk_comparable_to_fp6_(max_comparable), *max_index_ptr = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_e3m2_neon_strided_(                      //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_e3m2_t *min_value_ptr, nk_size_t *min_index_ptr,                    //
    nk_e3m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    uint8x16_t min_u8x16 = vdupq_n_u8(0xFF), max_u8x16 = vdupq_n_u8(0);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    uint8x16_t lane_indices_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
    nk_size_t idx = 0;
    uint8x16_t data_for_min_u8x16, data_for_max_u8x16;

nk_reduce_minmax_e3m2_neon_cycle:
    if (stride_elements == 2 && idx + 16 <= count) {
        uint8x16x2_t loaded = vld2q_u8((nk_u8_t const *)(data_ptr + idx * 2));
        uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(loaded.val[0]);
        data_for_min_u8x16 = comparable_u8x16;
        data_for_max_u8x16 = comparable_u8x16;
        idx += 16;
    }
    else if (stride_elements == 3 && idx + 16 <= count) {
        uint8x16x3_t loaded = vld3q_u8((nk_u8_t const *)(data_ptr + idx * 3));
        uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(loaded.val[0]);
        data_for_min_u8x16 = comparable_u8x16;
        data_for_max_u8x16 = comparable_u8x16;
        idx += 16;
    }
    else if (stride_elements == 4 && idx + 16 <= count) {
        uint8x16x4_t loaded = vld4q_u8((nk_u8_t const *)(data_ptr + idx * 4));
        uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(loaded.val[0]);
        data_for_min_u8x16 = comparable_u8x16;
        data_for_max_u8x16 = comparable_u8x16;
        idx += 16;
    }
    else if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_strided_load_b8x16_serial_(data_ptr + idx * stride_elements, stride_elements, &tail_vec, count - idx);
        uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(tail_vec.u8x16);
        uint8x16_t valid_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((uint8_t)(count - idx)));
        data_for_min_u8x16 = vbslq_u8(valid_u8x16, comparable_u8x16, vdupq_n_u8(0xFF));
        data_for_max_u8x16 = vbslq_u8(valid_u8x16, comparable_u8x16, vdupq_n_u8(0x00));
        idx = count;
    }
    else {
        nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
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
        *min_value_ptr = nk_comparable_to_fp6_(min_comparable), *min_index_ptr = min_idx;
        *max_value_ptr = nk_comparable_to_fp6_(max_comparable), *max_index_ptr = max_idx;
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
    goto nk_reduce_minmax_e3m2_neon_cycle;
}

NK_PUBLIC void nk_reduce_minmax_e3m2_neon(                              //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e3m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    int aligned = (stride_bytes % sizeof(nk_e3m2_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_E3M2_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_E3M2_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_e3m2_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (count > (nk_size_t)256 * 16) {
        nk_size_t left_count = count / 2;
        nk_e3m2_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_e3m2_neon(data_ptr, left_count, stride_bytes, &left_min_value, &left_min_index,
                                   &left_max_value, &left_max_index);
        nk_reduce_minmax_e3m2_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                   &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (nk_e3m2_order_serial(right_min_value, left_min_value) < 0)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (nk_e3m2_order_serial(right_max_value, left_max_value) > 0)
            *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e3m2_neon_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                               max_index_ptr);
    else if (stride_elements <= 4)
        nk_reduce_minmax_e3m2_neon_strided_(data_ptr, count, stride_elements, min_value_ptr, min_index_ptr,
                                            max_value_ptr, max_index_ptr);
    else
        nk_reduce_minmax_e3m2_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_e4m3_neon_contiguous_( //
    nk_e4m3_t const *data_ptr, nk_size_t count,           //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0), sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data_ptr + idx));
        float16x8_t half_lo_f16x8, half_hi_f16x8;
        nk_e4m3x16_to_f16x8x2_neon_(raw_u8x16, &half_lo_f16x8, &half_hi_f16x8);
        float32x4_t a_f32x4 = vcvt_f32_f16(vget_low_f16(half_lo_f16x8));
        float32x4_t b_f32x4 = vcvt_high_f32_f16(half_lo_f16x8);
        float32x4_t c_f32x4 = vcvt_f32_f16(vget_low_f16(half_hi_f16x8));
        float32x4_t d_f32x4 = vcvt_high_f32_f16(half_hi_f16x8);
        sum_f32x4 = vaddq_f32(vaddq_f32(sum_f32x4, vaddq_f32(a_f32x4, b_f32x4)), vaddq_f32(c_f32x4, d_f32x4));
        sumsq_f32x4 = vfmaq_f32(vfmaq_f32(vfmaq_f32(vfmaq_f32( //
                                                        sumsq_f32x4, a_f32x4, a_f32x4),
                                                    b_f32x4, b_f32x4),
                                          c_f32x4, c_f32x4),
                                d_f32x4, d_f32x4);
    }
    nk_f32_t sum = vaddvq_f32(sum_f32x4), sumsq = vaddvq_f32(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t value_f32;
        nk_e4m3_to_f32_serial(&data_ptr[idx], &value_f32);
        sum += value_f32, sumsq += value_f32 * value_f32;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_e4m3_neon_strided_(                     //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0), sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x2_t loaded_u8x16x2 = vld2q_u8((nk_u8_t const *)(data_ptr + idx * 2));
            float16x8_t half_lo_f16x8, half_hi_f16x8;
            nk_e4m3x16_to_f16x8x2_neon_(loaded_u8x16x2.val[0], &half_lo_f16x8, &half_hi_f16x8);
            float32x4_t a_f32x4 = vcvt_f32_f16(vget_low_f16(half_lo_f16x8));
            float32x4_t b_f32x4 = vcvt_high_f32_f16(half_lo_f16x8);
            float32x4_t c_f32x4 = vcvt_f32_f16(vget_low_f16(half_hi_f16x8));
            float32x4_t d_f32x4 = vcvt_high_f32_f16(half_hi_f16x8);
            sum_f32x4 = vaddq_f32(vaddq_f32(sum_f32x4, vaddq_f32(a_f32x4, b_f32x4)), vaddq_f32(c_f32x4, d_f32x4));
            sumsq_f32x4 = vfmaq_f32(vfmaq_f32(vfmaq_f32(vfmaq_f32( //
                                                            sumsq_f32x4, a_f32x4, a_f32x4),
                                                        b_f32x4, b_f32x4),
                                              c_f32x4, c_f32x4),
                                    d_f32x4, d_f32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x3_t loaded_u8x16x3 = vld3q_u8((nk_u8_t const *)(data_ptr + idx * 3));
            float16x8_t half_lo_f16x8, half_hi_f16x8;
            nk_e4m3x16_to_f16x8x2_neon_(loaded_u8x16x3.val[0], &half_lo_f16x8, &half_hi_f16x8);
            float32x4_t a_f32x4 = vcvt_f32_f16(vget_low_f16(half_lo_f16x8));
            float32x4_t b_f32x4 = vcvt_high_f32_f16(half_lo_f16x8);
            float32x4_t c_f32x4 = vcvt_f32_f16(vget_low_f16(half_hi_f16x8));
            float32x4_t d_f32x4 = vcvt_high_f32_f16(half_hi_f16x8);
            sum_f32x4 = vaddq_f32(vaddq_f32(sum_f32x4, vaddq_f32(a_f32x4, b_f32x4)), vaddq_f32(c_f32x4, d_f32x4));
            sumsq_f32x4 = vfmaq_f32(vfmaq_f32(vfmaq_f32(vfmaq_f32( //
                                                            sumsq_f32x4, a_f32x4, a_f32x4),
                                                        b_f32x4, b_f32x4),
                                              c_f32x4, c_f32x4),
                                    d_f32x4, d_f32x4);
        }
    }
    else {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x4_t loaded_u8x16x4 = vld4q_u8((nk_u8_t const *)(data_ptr + idx * 4));
            float16x8_t half_lo_f16x8, half_hi_f16x8;
            nk_e4m3x16_to_f16x8x2_neon_(loaded_u8x16x4.val[0], &half_lo_f16x8, &half_hi_f16x8);
            float32x4_t a_f32x4 = vcvt_f32_f16(vget_low_f16(half_lo_f16x8));
            float32x4_t b_f32x4 = vcvt_high_f32_f16(half_lo_f16x8);
            float32x4_t c_f32x4 = vcvt_f32_f16(vget_low_f16(half_hi_f16x8));
            float32x4_t d_f32x4 = vcvt_high_f32_f16(half_hi_f16x8);
            sum_f32x4 = vaddq_f32(vaddq_f32(sum_f32x4, vaddq_f32(a_f32x4, b_f32x4)), vaddq_f32(c_f32x4, d_f32x4));
            sumsq_f32x4 = vfmaq_f32(vfmaq_f32(vfmaq_f32(vfmaq_f32( //
                                                            sumsq_f32x4, a_f32x4, a_f32x4),
                                                        b_f32x4, b_f32x4),
                                              c_f32x4, c_f32x4),
                                    d_f32x4, d_f32x4);
        }
    }
    nk_f32_t sum = vaddvq_f32(sum_f32x4), sumsq = vaddvq_f32(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_e4m3_to_f32_serial((nk_e4m3_t const *)(data_ptr + idx * stride_elements), &val);
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_e4m3_neon(                             //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e4m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e4m3_neon(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e4m3_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                    &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum, *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e4m3_neon_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_e4m3_neon_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e4m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e4m3_neon_contiguous_( //
    nk_e4m3_t const *data_ptr, nk_size_t count,          //
    nk_e4m3_t *min_value_ptr, nk_size_t *min_index_ptr,  //
    nk_e4m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    uint8x16_t min_u8x16 = vdupq_n_u8(0xFF), max_u8x16 = vdupq_n_u8(0);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data_ptr + idx));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(raw_u8x16);
        uint8x16_t is_nan_u8x16 = vorrq_u8(vceqq_u8(comparable_u8x16, vdupq_n_u8(0x00)),
                                           vceqq_u8(comparable_u8x16, vdupq_n_u8(0xFF)));
        uint8x16_t data_min_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0xFF), comparable_u8x16);
        uint8x16_t data_max_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0x00), comparable_u8x16);
        uint8x16_t less_u8x16 = vcltq_u8(data_min_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(data_max_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, data_min_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, data_max_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
        iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b8x16_serial_(data_ptr + idx, &tail_vec, remaining);
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(tail_vec.u8x16);
        uint8x16_t is_nan_u8x16 = vorrq_u8(vceqq_u8(comparable_u8x16, vdupq_n_u8(0x00)),
                                           vceqq_u8(comparable_u8x16, vdupq_n_u8(0xFF)));
        uint8x16_t nan_min_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0xFF), comparable_u8x16);
        uint8x16_t nan_max_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0x00), comparable_u8x16);
        uint8x16_t lane_indices_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                    vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
        uint8x16_t valid_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((uint8_t)remaining));
        uint8x16_t data_for_min_u8x16 = vbslq_u8(valid_u8x16, nan_min_u8x16, vdupq_n_u8(0xFF));
        uint8x16_t data_for_max_u8x16 = vbslq_u8(valid_u8x16, nan_max_u8x16, vdupq_n_u8(0));
        uint8x16_t less_u8x16 = vcltq_u8(data_for_min_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(data_for_max_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, data_for_min_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, data_for_max_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
    }
    nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
    // If min stayed at 0xFF, all values were NaN
    if (min_comparable == 0xFF) {
        *min_value_ptr = (nk_e4m3_t)NK_E4M3_MAX, *min_index_ptr = NK_SIZE_MAX;
        *max_value_ptr = (nk_e4m3_t)NK_E4M3_MIN, *max_index_ptr = NK_SIZE_MAX;
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

NK_INTERNAL void nk_reduce_minmax_e4m3_neon_strided_(                      //
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

nk_reduce_minmax_e4m3_neon_cycle:
    if (stride_elements == 2 && idx + 16 <= count) {
        uint8x16x2_t loaded = vld2q_u8((nk_u8_t const *)(data_ptr + idx * 2));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded.val[0]);
        uint8x16_t is_nan_u8x16 = vorrq_u8(vceqq_u8(comparable_u8x16, vdupq_n_u8(0x00)),
                                           vceqq_u8(comparable_u8x16, vdupq_n_u8(0xFF)));
        data_for_min_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0xFF), comparable_u8x16);
        data_for_max_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0x00), comparable_u8x16);
        idx += 16;
    }
    else if (stride_elements == 3 && idx + 16 <= count) {
        uint8x16x3_t loaded = vld3q_u8((nk_u8_t const *)(data_ptr + idx * 3));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded.val[0]);
        uint8x16_t is_nan_u8x16 = vorrq_u8(vceqq_u8(comparable_u8x16, vdupq_n_u8(0x00)),
                                           vceqq_u8(comparable_u8x16, vdupq_n_u8(0xFF)));
        data_for_min_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0xFF), comparable_u8x16);
        data_for_max_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0x00), comparable_u8x16);
        idx += 16;
    }
    else if (stride_elements == 4 && idx + 16 <= count) {
        uint8x16x4_t loaded = vld4q_u8((nk_u8_t const *)(data_ptr + idx * 4));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded.val[0]);
        uint8x16_t is_nan_u8x16 = vorrq_u8(vceqq_u8(comparable_u8x16, vdupq_n_u8(0x00)),
                                           vceqq_u8(comparable_u8x16, vdupq_n_u8(0xFF)));
        data_for_min_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0xFF), comparable_u8x16);
        data_for_max_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0x00), comparable_u8x16);
        idx += 16;
    }
    else if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_strided_load_b8x16_serial_(data_ptr + idx * stride_elements, stride_elements, &tail_vec, count - idx);
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(tail_vec.u8x16);
        uint8x16_t is_nan_u8x16 = vorrq_u8(vceqq_u8(comparable_u8x16, vdupq_n_u8(0x00)),
                                           vceqq_u8(comparable_u8x16, vdupq_n_u8(0xFF)));
        uint8x16_t valid_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((uint8_t)(count - idx)));
        uint8x16_t invalid_or_nan_u8x16 = vornq_u8(is_nan_u8x16, valid_u8x16);
        data_for_min_u8x16 = vbslq_u8(invalid_or_nan_u8x16, vdupq_n_u8(0xFF), comparable_u8x16);
        data_for_max_u8x16 = vbslq_u8(invalid_or_nan_u8x16, vdupq_n_u8(0x00), comparable_u8x16);
        idx = count;
    }
    else {
        nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
        if (min_comparable == 0xFF) {
            *min_value_ptr = (nk_e4m3_t)NK_E4M3_MAX, *min_index_ptr = NK_SIZE_MAX;
            *max_value_ptr = (nk_e4m3_t)NK_E4M3_MIN, *max_index_ptr = NK_SIZE_MAX;
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
    goto nk_reduce_minmax_e4m3_neon_cycle;
}

NK_PUBLIC void nk_reduce_minmax_e4m3_neon(                              //
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
        nk_reduce_minmax_e4m3_neon(data_ptr, left_count, stride_bytes, &left_min_value, &left_min_index,
                                   &left_max_value, &left_max_index);
        nk_reduce_minmax_e4m3_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                   &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (left_min_index == NK_SIZE_MAX)
            *min_value_ptr = right_min_value,
            *min_index_ptr = right_min_index == NK_SIZE_MAX ? NK_SIZE_MAX : left_count + right_min_index;
        else if (right_min_index == NK_SIZE_MAX || nk_e4m3_order_serial(left_min_value, right_min_value) <= 0)
            *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        else *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        if (left_max_index == NK_SIZE_MAX)
            *max_value_ptr = right_max_value,
            *max_index_ptr = right_max_index == NK_SIZE_MAX ? NK_SIZE_MAX : left_count + right_max_index;
        else if (right_max_index == NK_SIZE_MAX || nk_e4m3_order_serial(right_max_value, left_max_value) <= 0)
            *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
        else *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e4m3_neon_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                               max_index_ptr);
    else if (stride_elements <= 4)
        nk_reduce_minmax_e4m3_neon_strided_(data_ptr, count, stride_elements, min_value_ptr, min_index_ptr,
                                            max_value_ptr, max_index_ptr);
    else
        nk_reduce_minmax_e4m3_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_e5m2_neon_contiguous_( //
    nk_e5m2_t const *data_ptr, nk_size_t count,           //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0), sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        uint8x8_t raw_u8x8 = vld1_u8((nk_u8_t const *)(data_ptr + idx));
        float16x8_t half_f16x8 = nk_e5m2x8_to_f16x8_neon_(raw_u8x8);
        float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(half_f16x8));
        float32x4_t hi_f32x4 = vcvt_high_f32_f16(half_f16x8);
        sum_f32x4 = vaddq_f32(sum_f32x4, vaddq_f32(lo_f32x4, hi_f32x4));
        sumsq_f32x4 = vfmaq_f32(vfmaq_f32(sumsq_f32x4, lo_f32x4, lo_f32x4), hi_f32x4, hi_f32x4);
    }
    nk_f32_t sum = vaddvq_f32(sum_f32x4), sumsq = vaddvq_f32(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t value_f32;
        nk_e5m2_to_f32_serial(&data_ptr[idx], &value_f32);
        sum += value_f32, sumsq += value_f32 * value_f32;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_e5m2_neon_strided_(                     //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0), sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint8x8x2_t loaded_u8x8x2 = vld2_u8((nk_u8_t const *)(data_ptr + idx * 2));
            float16x8_t half_f16x8 = nk_e5m2x8_to_f16x8_neon_(loaded_u8x8x2.val[0]);
            float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(half_f16x8));
            float32x4_t hi_f32x4 = vcvt_high_f32_f16(half_f16x8);
            sum_f32x4 = vaddq_f32(sum_f32x4, vaddq_f32(lo_f32x4, hi_f32x4));
            sumsq_f32x4 = vfmaq_f32(vfmaq_f32(sumsq_f32x4, lo_f32x4, lo_f32x4), hi_f32x4, hi_f32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint8x8x3_t loaded_u8x8x3 = vld3_u8((nk_u8_t const *)(data_ptr + idx * 3));
            float16x8_t half_f16x8 = nk_e5m2x8_to_f16x8_neon_(loaded_u8x8x3.val[0]);
            float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(half_f16x8));
            float32x4_t hi_f32x4 = vcvt_high_f32_f16(half_f16x8);
            sum_f32x4 = vaddq_f32(sum_f32x4, vaddq_f32(lo_f32x4, hi_f32x4));
            sumsq_f32x4 = vfmaq_f32(vfmaq_f32(sumsq_f32x4, lo_f32x4, lo_f32x4), hi_f32x4, hi_f32x4);
        }
    }
    else {
        for (; idx + 8 <= count; idx += 8) {
            uint8x8x4_t loaded_u8x8x4 = vld4_u8((nk_u8_t const *)(data_ptr + idx * 4));
            float16x8_t half_f16x8 = nk_e5m2x8_to_f16x8_neon_(loaded_u8x8x4.val[0]);
            float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(half_f16x8));
            float32x4_t hi_f32x4 = vcvt_high_f32_f16(half_f16x8);
            sum_f32x4 = vaddq_f32(sum_f32x4, vaddq_f32(lo_f32x4, hi_f32x4));
            sumsq_f32x4 = vfmaq_f32(vfmaq_f32(sumsq_f32x4, lo_f32x4, lo_f32x4), hi_f32x4, hi_f32x4);
        }
    }
    nk_f32_t sum = vaddvq_f32(sum_f32x4), sumsq = vaddvq_f32(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_e5m2_to_f32_serial((nk_e5m2_t const *)(data_ptr + idx * stride_elements), &val);
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_e5m2_neon(                             //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e5m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e5m2_neon(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e5m2_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                    &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum, *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e5m2_neon_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_e5m2_neon_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e5m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e5m2_neon_contiguous_( //
    nk_e5m2_t const *data_ptr, nk_size_t count,          //
    nk_e5m2_t *min_value_ptr, nk_size_t *min_index_ptr,  //
    nk_e5m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    uint8x16_t min_u8x16 = vdupq_n_u8(0xFF), max_u8x16 = vdupq_n_u8(0);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data_ptr + idx));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(raw_u8x16);
        uint8x16_t is_nan_low_u8x16 = vcleq_u8(comparable_u8x16, vdupq_n_u8(0x02));
        uint8x16_t is_nan_high_u8x16 = vcgeq_u8(comparable_u8x16, vdupq_n_u8(0xFD));
        uint8x16_t is_nan_u8x16 = vorrq_u8(is_nan_low_u8x16, is_nan_high_u8x16);
        uint8x16_t data_min_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0xFF), comparable_u8x16);
        uint8x16_t data_max_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0x00), comparable_u8x16);
        uint8x16_t less_u8x16 = vcltq_u8(data_min_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(data_max_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, data_min_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, data_max_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
        iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b8x16_serial_(data_ptr + idx, &tail_vec, remaining);
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(tail_vec.u8x16);
        uint8x16_t is_nan_low_u8x16 = vcleq_u8(comparable_u8x16, vdupq_n_u8(0x02));
        uint8x16_t is_nan_high_u8x16 = vcgeq_u8(comparable_u8x16, vdupq_n_u8(0xFD));
        uint8x16_t is_nan_u8x16 = vorrq_u8(is_nan_low_u8x16, is_nan_high_u8x16);
        uint8x16_t nan_min_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0xFF), comparable_u8x16);
        uint8x16_t nan_max_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0x00), comparable_u8x16);
        uint8x16_t lane_indices_u8x16 = vcombine_u8(vreinterpret_u8_u64(vcreate_u64(0x0706050403020100ULL)),
                                                    vreinterpret_u8_u64(vcreate_u64(0x0F0E0D0C0B0A0908ULL)));
        uint8x16_t valid_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((uint8_t)remaining));
        uint8x16_t data_for_min_u8x16 = vbslq_u8(valid_u8x16, nan_min_u8x16, vdupq_n_u8(0xFF));
        uint8x16_t data_for_max_u8x16 = vbslq_u8(valid_u8x16, nan_max_u8x16, vdupq_n_u8(0));
        uint8x16_t less_u8x16 = vcltq_u8(data_for_min_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(data_for_max_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, data_for_min_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, data_for_max_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
    }
    nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
    // If min stayed at 0xFF, all values were NaN
    if (min_comparable == 0xFF) {
        *min_value_ptr = (nk_e5m2_t)NK_E5M2_MAX, *min_index_ptr = NK_SIZE_MAX;
        *max_value_ptr = (nk_e5m2_t)NK_E5M2_MIN, *max_index_ptr = NK_SIZE_MAX;
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

NK_INTERNAL void nk_reduce_minmax_e5m2_neon_strided_(                      //
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

nk_reduce_minmax_e5m2_neon_cycle:
    if (stride_elements == 2 && idx + 16 <= count) {
        uint8x16x2_t loaded = vld2q_u8((nk_u8_t const *)(data_ptr + idx * 2));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded.val[0]);
        uint8x16_t is_nan_u8x16 = vorrq_u8(vcleq_u8(comparable_u8x16, vdupq_n_u8(0x02)),
                                           vcgeq_u8(comparable_u8x16, vdupq_n_u8(0xFD)));
        data_for_min_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0xFF), comparable_u8x16);
        data_for_max_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0x00), comparable_u8x16);
        idx += 16;
    }
    else if (stride_elements == 3 && idx + 16 <= count) {
        uint8x16x3_t loaded = vld3q_u8((nk_u8_t const *)(data_ptr + idx * 3));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded.val[0]);
        uint8x16_t is_nan_u8x16 = vorrq_u8(vcleq_u8(comparable_u8x16, vdupq_n_u8(0x02)),
                                           vcgeq_u8(comparable_u8x16, vdupq_n_u8(0xFD)));
        data_for_min_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0xFF), comparable_u8x16);
        data_for_max_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0x00), comparable_u8x16);
        idx += 16;
    }
    else if (stride_elements == 4 && idx + 16 <= count) {
        uint8x16x4_t loaded = vld4q_u8((nk_u8_t const *)(data_ptr + idx * 4));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded.val[0]);
        uint8x16_t is_nan_u8x16 = vorrq_u8(vcleq_u8(comparable_u8x16, vdupq_n_u8(0x02)),
                                           vcgeq_u8(comparable_u8x16, vdupq_n_u8(0xFD)));
        data_for_min_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0xFF), comparable_u8x16);
        data_for_max_u8x16 = vbslq_u8(is_nan_u8x16, vdupq_n_u8(0x00), comparable_u8x16);
        idx += 16;
    }
    else if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_strided_load_b8x16_serial_(data_ptr + idx * stride_elements, stride_elements, &tail_vec, count - idx);
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(tail_vec.u8x16);
        uint8x16_t is_nan_u8x16 = vorrq_u8(vcleq_u8(comparable_u8x16, vdupq_n_u8(0x02)),
                                           vcgeq_u8(comparable_u8x16, vdupq_n_u8(0xFD)));
        uint8x16_t valid_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((uint8_t)(count - idx)));
        uint8x16_t invalid_or_nan_u8x16 = vornq_u8(is_nan_u8x16, valid_u8x16);
        data_for_min_u8x16 = vbslq_u8(invalid_or_nan_u8x16, vdupq_n_u8(0xFF), comparable_u8x16);
        data_for_max_u8x16 = vbslq_u8(invalid_or_nan_u8x16, vdupq_n_u8(0x00), comparable_u8x16);
        idx = count;
    }
    else {
        nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
        if (min_comparable == 0xFF) {
            *min_value_ptr = (nk_e5m2_t)NK_E5M2_MAX, *min_index_ptr = NK_SIZE_MAX;
            *max_value_ptr = (nk_e5m2_t)NK_E5M2_MIN, *max_index_ptr = NK_SIZE_MAX;
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
    goto nk_reduce_minmax_e5m2_neon_cycle;
}

NK_PUBLIC void nk_reduce_minmax_e5m2_neon(                              //
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
        nk_reduce_minmax_e5m2_neon(data_ptr, left_count, stride_bytes, &left_min_value, &left_min_index,
                                   &left_max_value, &left_max_index);
        nk_reduce_minmax_e5m2_neon(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                   &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (left_min_index == NK_SIZE_MAX)
            *min_value_ptr = right_min_value,
            *min_index_ptr = right_min_index == NK_SIZE_MAX ? NK_SIZE_MAX : left_count + right_min_index;
        else if (right_min_index == NK_SIZE_MAX || nk_e5m2_order_serial(left_min_value, right_min_value) <= 0)
            *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        else *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        if (left_max_index == NK_SIZE_MAX)
            *max_value_ptr = right_max_value,
            *max_index_ptr = right_max_index == NK_SIZE_MAX ? NK_SIZE_MAX : left_count + right_max_index;
        else if (right_max_index == NK_SIZE_MAX || nk_e5m2_order_serial(right_max_value, left_max_value) <= 0)
            *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
        else *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e5m2_neon_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                               max_index_ptr);
    else if (stride_elements <= 4)
        nk_reduce_minmax_e5m2_neon_strided_(data_ptr, count, stride_elements, min_value_ptr, min_index_ptr,
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

#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_
#endif // NK_REDUCE_NEON_H
