/**
 *  @brief SIMD-accelerated Reductions for WASM.
 *  @file include/numkong/reduce/v128relaxed.h
 *  @author Ash Vardanian
 *  @date February 13, 2026
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_V128RELAXED_H
#define NK_REDUCE_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/reduce/serial.h"
#include "numkong/cast/v128relaxed.h" // `nk_bf16x4_to_f32x4_v128relaxed_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/** @brief Horizontal sum of 4 floats using shuffle tree. */
NK_INTERNAL nk_f32_t nk_reduce_add_f32x4_v128relaxed_(v128_t vec_f32x4) {
    v128_t high_f32x4 = wasm_i32x4_shuffle(vec_f32x4, vec_f32x4, 2, 3, 0, 0);
    v128_t sum1_f32x4 = wasm_f32x4_add(vec_f32x4, high_f32x4);
    v128_t high2_f32x4 = wasm_i32x4_shuffle(sum1_f32x4, sum1_f32x4, 1, 0, 0, 0);
    v128_t sum2_f32x4 = wasm_f32x4_add(sum1_f32x4, high2_f32x4);
    return wasm_f32x4_extract_lane(sum2_f32x4, 0);
}

/** @brief Horizontal sum of 2 doubles using single shuffle. */
NK_INTERNAL nk_f64_t nk_reduce_add_f64x2_v128relaxed_(v128_t vec_f64x2) {
    v128_t high_f64x2 = wasm_i64x2_shuffle(vec_f64x2, vec_f64x2, 1, 0);
    v128_t sum_f64x2 = wasm_f64x2_add(vec_f64x2, high_f64x2);
    return wasm_f64x2_extract_lane(sum_f64x2, 0);
}

/** @brief Horizontal sum of 4 signed 32-bit integers using shuffle tree. */
NK_INTERNAL nk_i32_t nk_reduce_add_i32x4_v128relaxed_(v128_t vec_i32x4) {
    v128_t high_i32x4 = wasm_i32x4_shuffle(vec_i32x4, vec_i32x4, 2, 3, 0, 0);
    v128_t sum1_i32x4 = wasm_i32x4_add(vec_i32x4, high_i32x4);
    v128_t high2_i32x4 = wasm_i32x4_shuffle(sum1_i32x4, sum1_i32x4, 1, 0, 0, 0);
    v128_t sum2_i32x4 = wasm_i32x4_add(sum1_i32x4, high2_i32x4);
    return wasm_i32x4_extract_lane(sum2_i32x4, 0);
}

/** @brief Horizontal sum of 4 unsigned 32-bit integers using shuffle tree. */
NK_INTERNAL nk_u32_t nk_reduce_add_u32x4_v128relaxed_(v128_t vec_u32x4) {
    v128_t high_u32x4 = wasm_i32x4_shuffle(vec_u32x4, vec_u32x4, 2, 3, 0, 0);
    v128_t sum1_u32x4 = wasm_i32x4_add(vec_u32x4, high_u32x4);
    v128_t high2_u32x4 = wasm_i32x4_shuffle(sum1_u32x4, sum1_u32x4, 1, 0, 0, 0);
    v128_t sum2_u32x4 = wasm_i32x4_add(sum1_u32x4, high2_u32x4);
    return (nk_u32_t)wasm_i32x4_extract_lane(sum2_u32x4, 0);
}

/** @brief  Horizontal sum of 16 unsigned 8-bit integers using pairwise widening. */
NK_INTERNAL nk_u32_t nk_reduce_add_u8x16_v128relaxed_(v128_t vec_u8x16) {
    v128_t sum_u16x8 = wasm_u16x8_extadd_pairwise_u8x16(vec_u8x16);
    v128_t sum_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(sum_u16x8);
    return nk_reduce_add_u32x4_v128relaxed_(sum_u32x4);
}

NK_INTERNAL nk_i64_t nk_reduce_add_i64x2_v128relaxed_(v128_t vec_i64x2) {
    v128_t high_i64x2 = wasm_i64x2_shuffle(vec_i64x2, vec_i64x2, 1, 0);
    v128_t sum_i64x2 = wasm_i64x2_add(vec_i64x2, high_i64x2);
    return (nk_i64_t)wasm_i64x2_extract_lane(sum_i64x2, 0);
}

NK_INTERNAL nk_u64_t nk_reduce_add_u64x2_v128relaxed_(v128_t vec_u64x2) {
    v128_t high_u64x2 = wasm_i64x2_shuffle(vec_u64x2, vec_u64x2, 1, 0);
    v128_t sum_u64x2 = wasm_i64x2_add(vec_u64x2, high_u64x2);
    return (nk_u64_t)wasm_i64x2_extract_lane(sum_u64x2, 0);
}

NK_INTERNAL nk_i32_t nk_reduce_add_i16x8_v128relaxed_(v128_t vec_i16x8) {
    v128_t pairwise_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(vec_i16x8);
    return nk_reduce_add_i32x4_v128relaxed_(pairwise_i32x4);
}

NK_INTERNAL nk_i64_t nk_reduce_add_i32x4_to_i64_v128relaxed_(v128_t vec_i32x4) {
    v128_t low_i64x2 = wasm_i64x2_extend_low_i32x4(vec_i32x4);
    v128_t high_i64x2 = wasm_i64x2_extend_high_i32x4(vec_i32x4);
    v128_t sum_i64x2 = wasm_i64x2_add(low_i64x2, high_i64x2);
    return nk_reduce_add_i64x2_v128relaxed_(sum_i64x2);
}

NK_INTERNAL nk_u64_t nk_reduce_add_u32x4_to_u64_v128relaxed_(v128_t vec_u32x4) {
    v128_t low_u64x2 = wasm_u64x2_extend_low_u32x4(vec_u32x4);
    v128_t high_u64x2 = wasm_u64x2_extend_high_u32x4(vec_u32x4);
    v128_t sum_u64x2 = wasm_i64x2_add(low_u64x2, high_u64x2);
    return nk_reduce_add_u64x2_v128relaxed_(sum_u64x2);
}

NK_INTERNAL v128_t nk_u64_sadd_epi64_v128relaxed_(v128_t a_u64x2, v128_t b_u64x2) {
    v128_t result_u64x2 = wasm_i64x2_add(a_u64x2, b_u64x2);
    v128_t sign_bit_i64x2 = wasm_i64x2_splat((nk_i64_t)0x8000000000000000LL);
    v128_t a_biased_i64x2 = wasm_v128_xor(a_u64x2, sign_bit_i64x2);
    v128_t result_biased_i64x2 = wasm_v128_xor(result_u64x2, sign_bit_i64x2);
    v128_t overflow_u64x2 = wasm_i64x2_gt(a_biased_i64x2, result_biased_i64x2);
    return wasm_v128_or(result_u64x2, overflow_u64x2);
}

NK_INTERNAL v128_t nk_i64_smul_sq_epi64_v128relaxed_(v128_t val_i64x2) {
    v128_t sign_i64x2 = wasm_i64x2_gt(wasm_i64x2_splat(0), val_i64x2);
    v128_t abs_val_u64x2 = wasm_i64x2_sub(wasm_v128_xor(val_i64x2, sign_i64x2), sign_i64x2);
    v128_t low_halves_i32x4 = wasm_i32x4_shuffle(abs_val_u64x2, abs_val_u64x2, 0, 2, 0, 0);
    v128_t low_squared_u64x2 = wasm_u64x2_extmul_low_u32x4(low_halves_i32x4, low_halves_i32x4);
    v128_t high_bits_u64x2 = wasm_u64x2_shr(abs_val_u64x2, 32);
    v128_t is_small_u64x2 = wasm_i64x2_eq(high_bits_u64x2, wasm_i64x2_splat(0));
    v128_t saturated_u64x2 = wasm_i64x2_splat(NK_I64_MAX);
    return wasm_i32x4_relaxed_laneselect(low_squared_u64x2, saturated_u64x2, is_small_u64x2);
}

NK_INTERNAL v128_t nk_u64_smul_sq_epi64_v128relaxed_(v128_t val_u64x2) {
    v128_t low_halves_i32x4 = wasm_i32x4_shuffle(val_u64x2, val_u64x2, 0, 2, 0, 0);
    v128_t low_squared_u64x2 = wasm_u64x2_extmul_low_u32x4(low_halves_i32x4, low_halves_i32x4);
    v128_t high_bits_u64x2 = wasm_u64x2_shr(val_u64x2, 32);
    v128_t is_small_u64x2 = wasm_i64x2_eq(high_bits_u64x2, wasm_i64x2_splat(0));
    v128_t saturated_u64x2 = wasm_i64x2_splat((nk_i64_t)-1);
    return wasm_i32x4_relaxed_laneselect(low_squared_u64x2, saturated_u64x2, is_small_u64x2);
}

NK_INTERNAL nk_u64_t nk_reduce_sadd_u64x2_v128relaxed_(v128_t v_u64x2) {
    v128_t swapped_u64x2 = wasm_i64x2_shuffle(v_u64x2, v_u64x2, 1, 0);
    v128_t sum_u64x2 = wasm_i64x2_add(v_u64x2, swapped_u64x2);
    v128_t sign_bit_i64x2 = wasm_i64x2_splat((nk_i64_t)0x8000000000000000LL);
    v128_t v_biased_i64x2 = wasm_v128_xor(v_u64x2, sign_bit_i64x2);
    v128_t sum_biased_i64x2 = wasm_v128_xor(sum_u64x2, sign_bit_i64x2);
    v128_t overflow_u64x2 = wasm_i64x2_gt(v_biased_i64x2, sum_biased_i64x2);
    sum_u64x2 = wasm_v128_or(sum_u64x2, overflow_u64x2);
    return (nk_u64_t)wasm_i64x2_extract_lane(sum_u64x2, 0);
}

NK_INTERNAL void nk_reduce_moments_f32_v128relaxed_contiguous_( //
    nk_f32_t const *data, nk_size_t count,                      //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    v128_t sum_f64x2 = wasm_f64x2_splat(0), sumsq_f64x2 = wasm_f64x2_splat(0);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        v128_t data_f32x4 = wasm_v128_load(data + idx);
        v128_t low_f64x2 = wasm_f64x2_promote_low_f32x4(data_f32x4);
        v128_t high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(data_f32x4, data_f32x4, 2, 3, 0, 1));
        sum_f64x2 = wasm_f64x2_add(wasm_f64x2_add(sum_f64x2, low_f64x2), high_f64x2);
        sumsq_f64x2 = wasm_f64x2_relaxed_madd(low_f64x2, low_f64x2, sumsq_f64x2);
        sumsq_f64x2 = wasm_f64x2_relaxed_madd(high_f64x2, high_f64x2, sumsq_f64x2);
    }
    nk_f64_t sum = nk_reduce_add_f64x2_v128relaxed_(sum_f64x2);
    nk_f64_t sumsq = nk_reduce_add_f64x2_v128relaxed_(sumsq_f64x2);
    for (; idx < count; ++idx) {
        nk_f64_t val = (nk_f64_t)data[idx];
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_f32_v128relaxed(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f32_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 4) {
        nk_size_t left_count = count / 2;
        nk_f64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f32_v128relaxed(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f32_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                          &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f32_v128relaxed_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_f32_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_f64_v128relaxed_contiguous_( //
    nk_f64_t const *data, nk_size_t count,                      //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    v128_t sum_f64x2 = wasm_f64x2_splat(0);
    v128_t sum_comp_f64x2 = wasm_f64x2_splat(0);
    v128_t sumsq_f64x2 = wasm_f64x2_splat(0);
    v128_t sumsq_comp_f64x2 = wasm_f64x2_splat(0);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        v128_t val_f64x2 = wasm_v128_load(data + idx);
        v128_t tentative_f64x2 = wasm_f64x2_add(sum_f64x2, val_f64x2);
        v128_t round_f64x2 = wasm_f64x2_sub(tentative_f64x2, sum_f64x2);
        v128_t corr_f64x2 = wasm_f64x2_add(wasm_f64x2_sub(sum_f64x2, wasm_f64x2_sub(tentative_f64x2, round_f64x2)),
                                           wasm_f64x2_sub(val_f64x2, round_f64x2));
        sum_comp_f64x2 = wasm_f64x2_add(sum_comp_f64x2, corr_f64x2);
        sum_f64x2 = tentative_f64x2;
        v128_t sq_f64x2 = wasm_f64x2_mul(val_f64x2, val_f64x2);
        v128_t tentative_sq_f64x2 = wasm_f64x2_add(sumsq_f64x2, sq_f64x2);
        v128_t round_sq_f64x2 = wasm_f64x2_sub(tentative_sq_f64x2, sumsq_f64x2);
        v128_t corr_sq_f64x2 = wasm_f64x2_add(
            wasm_f64x2_sub(sumsq_f64x2, wasm_f64x2_sub(tentative_sq_f64x2, round_sq_f64x2)),
            wasm_f64x2_sub(sq_f64x2, round_sq_f64x2));
        sumsq_comp_f64x2 = wasm_f64x2_add(sumsq_comp_f64x2, corr_sq_f64x2);
        sumsq_f64x2 = tentative_sq_f64x2;
    }
    nk_f64_t sum = nk_reduce_add_f64x2_v128relaxed_(wasm_f64x2_add(sum_f64x2, sum_comp_f64x2));
    nk_f64_t sumsq = nk_reduce_add_f64x2_v128relaxed_(wasm_f64x2_add(sumsq_f64x2, sumsq_comp_f64x2));
    for (; idx < count; ++idx) {
        nk_f64_t val = data[idx];
        sum += val;
        sumsq += val * val;
    }
    *sum_ptr = sum;
    *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_f64_v128relaxed(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f64_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 2) {
        nk_size_t left_count = count / 2;
        nk_f64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f64_v128relaxed(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f64_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                          &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f64_v128relaxed_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_f64_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_bf16_v128relaxed_contiguous_( //
    nk_bf16_t const *data, nk_size_t count,                      //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0);
    v128_t sumsq_f32x4 = wasm_f32x4_splat(0);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        nk_b64_vec_t raw;
        raw.u64 = *(nk_u64_t const *)(data + idx);
        v128_t data_f32x4 = nk_bf16x4_to_f32x4_v128relaxed_(raw).v128;
        sum_f32x4 = wasm_f32x4_add(sum_f32x4, data_f32x4);
        sumsq_f32x4 = wasm_f32x4_relaxed_madd(data_f32x4, data_f32x4, sumsq_f32x4);
    }
    nk_f32_t sum = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
    nk_f32_t sumsq = nk_reduce_add_f32x4_v128relaxed_(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_bf16_to_f32_serial(data + idx, &val);
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_bf16_v128relaxed(                  //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_bf16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 4) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_bf16_v128relaxed(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_bf16_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                           &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_bf16_v128relaxed_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_bf16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_f16_v128relaxed_contiguous_( //
    nk_f16_t const *data, nk_size_t count,                      //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0);
    v128_t sumsq_f32x4 = wasm_f32x4_splat(0);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        nk_b64_vec_t raw;
        raw.u64 = *(nk_u64_t const *)(data + idx);
        v128_t data_f32x4 = nk_f16x4_to_f32x4_v128relaxed_(raw).v128;
        sum_f32x4 = wasm_f32x4_add(sum_f32x4, data_f32x4);
        sumsq_f32x4 = wasm_f32x4_relaxed_madd(data_f32x4, data_f32x4, sumsq_f32x4);
    }
    nk_f32_t sum = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
    nk_f32_t sumsq = nk_reduce_add_f32x4_v128relaxed_(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_f16_to_f32_serial(data + idx, &val);
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_f16_v128relaxed(                  //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 4) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f16_v128relaxed(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f16_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                          &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f16_v128relaxed_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_f16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_f32_v128relaxed_contiguous_( //
    nk_f32_t const *data, nk_size_t count,                     //
    nk_f32_t *min_value_ptr, nk_size_t *min_index_ptr,         //
    nk_f32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    v128_t min_f32x4 = wasm_f32x4_splat(NK_F32_MAX), max_f32x4 = wasm_f32x4_splat(NK_F32_MIN);
    v128_t min_iter_u32x4 = wasm_i32x4_splat(0), max_iter_u32x4 = wasm_i32x4_splat(0);
    v128_t iter_u32x4 = wasm_i32x4_splat(0), one_u32x4 = wasm_i32x4_splat(1);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        v128_t data_f32x4 = wasm_v128_load(data + idx);
        v128_t less_b32x4 = wasm_f32x4_lt(data_f32x4, min_f32x4);
        v128_t greater_b32x4 = wasm_f32x4_gt(data_f32x4, max_f32x4);
        min_f32x4 = wasm_i32x4_relaxed_laneselect(data_f32x4, min_f32x4, less_b32x4);
        max_f32x4 = wasm_i32x4_relaxed_laneselect(data_f32x4, max_f32x4, greater_b32x4);
        min_iter_u32x4 = wasm_i32x4_relaxed_laneselect(iter_u32x4, min_iter_u32x4, less_b32x4);
        max_iter_u32x4 = wasm_i32x4_relaxed_laneselect(iter_u32x4, max_iter_u32x4, greater_b32x4);
        iter_u32x4 = wasm_i32x4_add(iter_u32x4, one_u32x4);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_iters_vec, max_iters_vec;
    min_values_vec.v128 = min_f32x4;
    max_values_vec.v128 = max_f32x4;
    min_iters_vec.v128 = min_iter_u32x4;
    max_iters_vec.v128 = max_iter_u32x4;
    nk_f32_t min_value = min_values_vec.f32s[0];
    nk_size_t min_idx = (nk_size_t)min_iters_vec.u32s[0] * 4;
    for (int i = 1; i < 4; ++i) {
        nk_size_t abs_idx = (nk_size_t)min_iters_vec.u32s[i] * 4 + (nk_size_t)i;
        if (min_values_vec.f32s[i] < min_value || (min_values_vec.f32s[i] == min_value && abs_idx < min_idx))
            min_value = min_values_vec.f32s[i], min_idx = abs_idx;
    }
    nk_f32_t max_value = max_values_vec.f32s[0];
    nk_size_t max_idx = (nk_size_t)max_iters_vec.u32s[0] * 4;
    for (int i = 1; i < 4; ++i) {
        nk_size_t abs_idx = (nk_size_t)max_iters_vec.u32s[i] * 4 + (nk_size_t)i;
        if (max_values_vec.f32s[i] > max_value || (max_values_vec.f32s[i] == max_value && abs_idx < max_idx))
            max_value = max_values_vec.f32s[i], max_idx = abs_idx;
    }
    for (; idx < count; ++idx) {
        nk_f32_t val = data[idx];
        if (val < min_value) min_value = val, min_idx = idx;
        if (val > max_value) max_value = val, max_idx = idx;
    }
    if (min_value == NK_F32_MAX && max_value == NK_F32_MIN) {
        *min_value_ptr = NK_F32_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F32_MIN,
        *max_index_ptr = NK_SIZE_MAX;
        return;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_f32_v128relaxed(                   //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_f32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_F32_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F32_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f32_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)NK_U32_MAX * 4) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_f32_v128relaxed(data, left_count, stride_bytes, &left_min_value, &left_min_index,
                                         &left_max_value, &left_max_index);
        nk_reduce_minmax_f32_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                         &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (right_min_value < left_min_value)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (right_max_value > left_max_value)
            *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_f32_v128relaxed_contiguous_(data, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                     max_index_ptr);
    else
        nk_reduce_minmax_f32_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_f64_v128relaxed_contiguous_( //
    nk_f64_t const *data, nk_size_t count,                     //
    nk_f64_t *min_value_ptr, nk_size_t *min_index_ptr,         //
    nk_f64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    v128_t min_f64x2 = wasm_f64x2_splat(NK_F64_MAX), max_f64x2 = wasm_f64x2_splat(NK_F64_MIN);
    v128_t min_iter_u64x2 = wasm_i64x2_splat(0), max_iter_u64x2 = wasm_i64x2_splat(0);
    v128_t iter_u64x2 = wasm_i64x2_splat(0), one_u64x2 = wasm_i64x2_splat(1);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        v128_t data_f64x2 = wasm_v128_load(data + idx);
        v128_t less_b64x2 = wasm_f64x2_lt(data_f64x2, min_f64x2);
        v128_t greater_b64x2 = wasm_f64x2_gt(data_f64x2, max_f64x2);
        min_f64x2 = wasm_i64x2_relaxed_laneselect(data_f64x2, min_f64x2, less_b64x2);
        max_f64x2 = wasm_i64x2_relaxed_laneselect(data_f64x2, max_f64x2, greater_b64x2);
        min_iter_u64x2 = wasm_i64x2_relaxed_laneselect(iter_u64x2, min_iter_u64x2, less_b64x2);
        max_iter_u64x2 = wasm_i64x2_relaxed_laneselect(iter_u64x2, max_iter_u64x2, greater_b64x2);
        iter_u64x2 = wasm_i64x2_add(iter_u64x2, one_u64x2);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_iters_vec, max_iters_vec;
    min_values_vec.v128 = min_f64x2;
    max_values_vec.v128 = max_f64x2;
    min_iters_vec.v128 = min_iter_u64x2;
    max_iters_vec.v128 = max_iter_u64x2;
    nk_f64_t min_value = min_values_vec.f64s[0];
    nk_size_t min_idx = (nk_size_t)min_iters_vec.u64s[0] * 2;
    if (min_values_vec.f64s[1] < min_value ||
        (min_values_vec.f64s[1] == min_value && (nk_size_t)min_iters_vec.u64s[1] * 2 + 1 < min_idx))
        min_value = min_values_vec.f64s[1], min_idx = (nk_size_t)min_iters_vec.u64s[1] * 2 + 1;
    nk_f64_t max_value = max_values_vec.f64s[0];
    nk_size_t max_idx = (nk_size_t)max_iters_vec.u64s[0] * 2;
    if (max_values_vec.f64s[1] > max_value ||
        (max_values_vec.f64s[1] == max_value && (nk_size_t)max_iters_vec.u64s[1] * 2 + 1 < max_idx))
        max_value = max_values_vec.f64s[1], max_idx = (nk_size_t)max_iters_vec.u64s[1] * 2 + 1;
    for (; idx < count; ++idx) {
        nk_f64_t val = data[idx];
        if (val < min_value) min_value = val, min_idx = idx;
        if (val > max_value) max_value = val, max_idx = idx;
    }
    if (min_value == NK_F64_MAX && max_value == NK_F64_MIN) {
        *min_value_ptr = NK_F64_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F64_MIN,
        *max_index_ptr = NK_SIZE_MAX;
        return;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_f64_v128relaxed(                   //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_f64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_F64_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F64_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f64_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_f64_v128relaxed_contiguous_(data, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                     max_index_ptr);
    else
        nk_reduce_minmax_f64_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_bf16_v128relaxed_contiguous_( //
    nk_bf16_t const *data, nk_size_t count,                     //
    nk_bf16_t *min_value_ptr, nk_size_t *min_index_ptr,         //
    nk_bf16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    v128_t abs_mask_u16x8 = wasm_i16x8_splat(0x7FFF);
    v128_t nan_threshold_u16x8 = wasm_i16x8_splat((short)0x7F80);
    v128_t min_cmp_i16x8 = wasm_i16x8_splat(0x7F80);        // +inf comparable
    v128_t max_cmp_i16x8 = wasm_i16x8_splat((short)0x807F); // -inf comparable
    v128_t min_iter_u16x8 = wasm_i16x8_splat(0), max_iter_u16x8 = wasm_i16x8_splat(0);
    v128_t iter_u16x8 = wasm_i16x8_splat(0), one_u16x8 = wasm_i16x8_splat(1);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        v128_t raw_u16x8 = wasm_v128_load(data + idx);
        // Convert to comparable i16: sign = srai(raw, 15), flip = srli(sign, 1), cmp = raw ^ flip
        v128_t sign_i16x8 = wasm_i16x8_shr(raw_u16x8, 15);
        v128_t flip_u16x8 = wasm_u16x8_shr(sign_i16x8, 1);
        v128_t cmp_i16x8 = wasm_v128_xor(raw_u16x8, flip_u16x8);
        // Filter NaN: (raw & 0x7FFF) <= 0x7F80 (both sides non-negative, so signed LE works)
        v128_t abs_u16x8 = wasm_v128_and(raw_u16x8, abs_mask_u16x8);
        v128_t not_nan_i16x8 = wasm_i16x8_le(abs_u16x8, nan_threshold_u16x8);
        // Compare as signed i16, masked by not-NaN
        v128_t less_i16x8 = wasm_v128_and(wasm_i16x8_lt(cmp_i16x8, min_cmp_i16x8), not_nan_i16x8);
        v128_t greater_i16x8 = wasm_v128_and(wasm_i16x8_gt(cmp_i16x8, max_cmp_i16x8), not_nan_i16x8);
        min_cmp_i16x8 = wasm_i16x8_relaxed_laneselect(cmp_i16x8, min_cmp_i16x8, less_i16x8);
        max_cmp_i16x8 = wasm_i16x8_relaxed_laneselect(cmp_i16x8, max_cmp_i16x8, greater_i16x8);
        min_iter_u16x8 = wasm_i16x8_relaxed_laneselect(iter_u16x8, min_iter_u16x8, less_i16x8);
        max_iter_u16x8 = wasm_i16x8_relaxed_laneselect(iter_u16x8, max_iter_u16x8, greater_i16x8);
        iter_u16x8 = wasm_i16x8_add(iter_u16x8, one_u16x8);
    }
    // Horizontal reduction over 8 lanes
    nk_b128_vec_t min_cmp_vec, max_cmp_vec, min_iters_vec, max_iters_vec;
    min_cmp_vec.v128 = min_cmp_i16x8;
    max_cmp_vec.v128 = max_cmp_i16x8;
    min_iters_vec.v128 = min_iter_u16x8;
    max_iters_vec.v128 = max_iter_u16x8;
    nk_i16_t min_comparable = min_cmp_vec.i16s[0];
    nk_size_t min_idx = (nk_size_t)min_iters_vec.u16s[0] * 8;
    for (int i = 1; i < 8; ++i) {
        nk_size_t abs_idx = (nk_size_t)min_iters_vec.u16s[i] * 8 + (nk_size_t)i;
        if (min_cmp_vec.i16s[i] < min_comparable || (min_cmp_vec.i16s[i] == min_comparable && abs_idx < min_idx))
            min_comparable = min_cmp_vec.i16s[i], min_idx = abs_idx;
    }
    nk_i16_t max_comparable = max_cmp_vec.i16s[0];
    nk_size_t max_idx = (nk_size_t)max_iters_vec.u16s[0] * 8;
    for (int i = 1; i < 8; ++i) {
        nk_size_t abs_idx = (nk_size_t)max_iters_vec.u16s[i] * 8 + (nk_size_t)i;
        if (max_cmp_vec.i16s[i] > max_comparable || (max_cmp_vec.i16s[i] == max_comparable && abs_idx < max_idx))
            max_comparable = max_cmp_vec.i16s[i], max_idx = abs_idx;
    }
    // Scalar tail
    for (; idx < count; ++idx) {
        nk_u16_t raw = *(nk_u16_t const *)(data + idx);
        if ((raw & 0x7FFF) > 0x7F80) continue; // skip NaN
        nk_i16_t comparable = (raw & 0x8000) ? (nk_i16_t)(raw ^ 0x7FFF) : (nk_i16_t)raw;
        if (comparable < min_comparable) min_comparable = comparable, min_idx = idx;
        if (comparable > max_comparable) max_comparable = comparable, max_idx = idx;
    }
    if (min_comparable == 0x7F80 && max_comparable == (nk_i16_t)0x807F) {
        *min_value_ptr = NK_BF16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_BF16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
        return;
    }
    // Convert comparable back to raw bf16
    nk_i16_t min_sign = min_comparable >> 15;
    nk_u16_t min_raw = (nk_u16_t)min_comparable ^ ((nk_u16_t)min_sign >> 1);
    *(nk_u16_t *)min_value_ptr = min_raw, *min_index_ptr = min_idx;
    nk_i16_t max_sign = max_comparable >> 15;
    nk_u16_t max_raw = (nk_u16_t)max_comparable ^ ((nk_u16_t)max_sign >> 1);
    *(nk_u16_t *)max_value_ptr = max_raw, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_bf16_v128relaxed(                   //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_bf16_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_bf16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_BF16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_BF16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_bf16_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_bf16_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_bf16_v128relaxed(data, left_count, stride_bytes, &left_min_value, &left_min_index,
                                          &left_max_value, &left_max_index);
        nk_reduce_minmax_bf16_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                          &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (nk_bf16_order_serial(right_min_value, left_min_value) < 0)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (nk_bf16_order_serial(right_max_value, left_max_value) > 0)
            *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_bf16_v128relaxed_contiguous_(data, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                      max_index_ptr);
    else
        nk_reduce_minmax_bf16_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_f16_v128relaxed_contiguous_( //
    nk_f16_t const *data, nk_size_t count,                     //
    nk_f16_t *min_value_ptr, nk_size_t *min_index_ptr,         //
    nk_f16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    v128_t min_f32x4 = wasm_f32x4_splat(NK_F32_MAX), max_f32x4 = wasm_f32x4_splat(NK_F32_MIN);
    v128_t min_iter_u32x4 = wasm_i32x4_splat(0), max_iter_u32x4 = wasm_i32x4_splat(0);
    v128_t iter_u32x4 = wasm_i32x4_splat(0), one_u32x4 = wasm_i32x4_splat(1);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        nk_b64_vec_t raw;
        raw.u64 = *(nk_u64_t const *)(data + idx);
        v128_t data_f32x4 = nk_f16x4_to_f32x4_v128relaxed_(raw).v128;
        v128_t less_b32x4 = wasm_f32x4_lt(data_f32x4, min_f32x4);
        v128_t greater_b32x4 = wasm_f32x4_gt(data_f32x4, max_f32x4);
        min_f32x4 = wasm_i32x4_relaxed_laneselect(data_f32x4, min_f32x4, less_b32x4);
        max_f32x4 = wasm_i32x4_relaxed_laneselect(data_f32x4, max_f32x4, greater_b32x4);
        min_iter_u32x4 = wasm_i32x4_relaxed_laneselect(iter_u32x4, min_iter_u32x4, less_b32x4);
        max_iter_u32x4 = wasm_i32x4_relaxed_laneselect(iter_u32x4, max_iter_u32x4, greater_b32x4);
        iter_u32x4 = wasm_i32x4_add(iter_u32x4, one_u32x4);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_iters_vec, max_iters_vec;
    min_values_vec.v128 = min_f32x4;
    max_values_vec.v128 = max_f32x4;
    min_iters_vec.v128 = min_iter_u32x4;
    max_iters_vec.v128 = max_iter_u32x4;
    nk_f32_t min_value_f32 = min_values_vec.f32s[0];
    nk_size_t min_idx = (nk_size_t)min_iters_vec.u32s[0] * 4;
    for (int i = 1; i < 4; ++i) {
        nk_size_t abs_idx = (nk_size_t)min_iters_vec.u32s[i] * 4 + (nk_size_t)i;
        if (min_values_vec.f32s[i] < min_value_f32 || (min_values_vec.f32s[i] == min_value_f32 && abs_idx < min_idx))
            min_value_f32 = min_values_vec.f32s[i], min_idx = abs_idx;
    }
    nk_f32_t max_value_f32 = max_values_vec.f32s[0];
    nk_size_t max_idx = (nk_size_t)max_iters_vec.u32s[0] * 4;
    for (int i = 1; i < 4; ++i) {
        nk_size_t abs_idx = (nk_size_t)max_iters_vec.u32s[i] * 4 + (nk_size_t)i;
        if (max_values_vec.f32s[i] > max_value_f32 || (max_values_vec.f32s[i] == max_value_f32 && abs_idx < max_idx))
            max_value_f32 = max_values_vec.f32s[i], max_idx = abs_idx;
    }
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_f16_to_f32_serial(data + idx, &val);
        if (val < min_value_f32) min_value_f32 = val, min_idx = idx;
        if (val > max_value_f32) max_value_f32 = val, max_idx = idx;
    }
    if (min_value_f32 == NK_F32_MAX && max_value_f32 == NK_F32_MIN) {
        *min_value_ptr = nk_f16_from_u16_(NK_F16_MAX), *min_index_ptr = NK_SIZE_MAX,
        *max_value_ptr = nk_f16_from_u16_(NK_F16_MIN), *max_index_ptr = NK_SIZE_MAX;
        return;
    }
    *min_value_ptr = data[min_idx], *min_index_ptr = min_idx;
    *max_value_ptr = data[max_idx], *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_f16_v128relaxed(                   //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f16_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_f16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0)
        *min_value_ptr = nk_f16_from_u16_(NK_F16_MAX), *min_index_ptr = NK_SIZE_MAX,
        *max_value_ptr = nk_f16_from_u16_(NK_F16_MIN), *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f16_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)NK_U32_MAX * 4) {
        nk_size_t left_count = count / 2;
        nk_f16_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_f16_v128relaxed(data, left_count, stride_bytes, &left_min_value, &left_min_index,
                                         &left_max_value, &left_max_index);
        nk_reduce_minmax_f16_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                         &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (nk_f16_order_serial(right_min_value, left_min_value) < 0)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (nk_f16_order_serial(right_max_value, left_max_value) > 0)
            *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_f16_v128relaxed_contiguous_(data, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                     max_index_ptr);
    else
        nk_reduce_minmax_f16_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_i8_v128relaxed_contiguous_( //
    nk_i8_t const *data, nk_size_t count,                      //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    v128_t sum_i32x4 = wasm_i32x4_splat(0);
    v128_t sumsq_u64x2 = wasm_i64x2_splat(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        v128_t data_i8x16 = wasm_v128_load(data + idx);
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
    nk_i64_t sum = nk_reduce_add_i32x4_v128relaxed_(sum_i32x4);
    nk_u64_t sumsq = nk_reduce_add_u64x2_v128relaxed_(sumsq_u64x2);
    for (; idx < count; ++idx) {
        nk_i64_t val = (nk_i64_t)data[idx];
        sum += val, sumsq += (nk_u64_t)(val * val);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i8_v128relaxed(                  //
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
        nk_reduce_moments_i8_v128relaxed(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i8_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                         &right_sum, &right_sumsq);
        *sum = nk_i64_saturating_add_serial(left_sum, right_sum);
        *sumsq = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i8_v128relaxed_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_i8_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_u8_v128relaxed_contiguous_( //
    nk_u8_t const *data, nk_size_t count,                      //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    v128_t sum_u32x4 = wasm_i32x4_splat(0);
    v128_t sumsq_u64x2 = wasm_i64x2_splat(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        v128_t data_u8x16 = wasm_v128_load(data + idx);
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
    nk_u64_t sum = nk_reduce_add_u32x4_v128relaxed_(sum_u32x4);
    nk_u64_t sumsq = nk_reduce_add_u64x2_v128relaxed_(sumsq_u64x2);
    for (; idx < count; ++idx) {
        nk_u64_t val = (nk_u64_t)data[idx];
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u8_v128relaxed(                  //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u8_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u8_v128relaxed(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u8_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                         &right_sum, &right_sumsq);
        *sum = nk_u64_saturating_add_serial(left_sum, right_sum);
        *sumsq = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u8_v128relaxed_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_u8_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_i16_v128relaxed_contiguous_( //
    nk_i16_t const *data, nk_size_t count,                      //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    v128_t sum_i64x2 = wasm_i64x2_splat(0);
    v128_t sumsq_u64x2 = wasm_i64x2_splat(0);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        v128_t data_i16x8 = wasm_v128_load(data + idx);
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
    nk_i64_t sum = nk_reduce_add_i64x2_v128relaxed_(sum_i64x2);
    nk_u64_t sumsq = nk_reduce_add_u64x2_v128relaxed_(sumsq_u64x2);
    for (; idx < count; ++idx) {
        nk_i64_t val = (nk_i64_t)data[idx];
        sum += val, sumsq += (nk_u64_t)(val * val);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i16_v128relaxed(                  //
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
        nk_reduce_moments_i16_v128relaxed(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i16_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                          &right_sum, &right_sumsq);
        *sum = nk_i64_saturating_add_serial(left_sum, right_sum);
        *sumsq = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i16_v128relaxed_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_i16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_u16_v128relaxed_contiguous_( //
    nk_u16_t const *data, nk_size_t count,                      //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    v128_t sum_u64x2 = wasm_i64x2_splat(0);
    v128_t sumsq_u64x2 = wasm_i64x2_splat(0);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        v128_t data_u16x8 = wasm_v128_load(data + idx);
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
    nk_u64_t sum = nk_reduce_add_u64x2_v128relaxed_(sum_u64x2);
    nk_u64_t sumsq = nk_reduce_add_u64x2_v128relaxed_(sumsq_u64x2);
    for (; idx < count; ++idx) {
        nk_u64_t val = (nk_u64_t)data[idx];
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u16_v128relaxed(                  //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u16_v128relaxed(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u16_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                          &right_sum, &right_sumsq);
        *sum = nk_u64_saturating_add_serial(left_sum, right_sum);
        *sumsq = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u16_v128relaxed_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_u16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_i32_v128relaxed_contiguous_( //
    nk_i32_t const *data, nk_size_t count,                      //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    v128_t sum_lower_u64x2 = wasm_i64x2_splat(0);
    v128_t sum_upper_i64x2 = wasm_i64x2_splat(0);
    v128_t sumsq_u64x2 = wasm_i64x2_splat(0);
    v128_t sumsq_overflow_u64x2 = wasm_i64x2_splat(0);
    v128_t sign_bit_i64x2 = wasm_i64x2_splat((nk_i64_t)0x8000000000000000LL);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        v128_t data_i32x4 = wasm_v128_load(data + idx);
        v128_t data_low_i64x2 = wasm_i64x2_extend_low_i32x4(data_i32x4);
        v128_t before_u64x2 = sum_lower_u64x2;
        sum_lower_u64x2 = wasm_i64x2_add(sum_lower_u64x2, data_low_i64x2);
        v128_t result_biased_i64x2 = wasm_v128_xor(sum_lower_u64x2, sign_bit_i64x2);
        v128_t before_biased_i64x2 = wasm_v128_xor(before_u64x2, sign_bit_i64x2);
        v128_t carry_u64x2 = wasm_i64x2_gt(before_biased_i64x2, result_biased_i64x2);
        sum_upper_i64x2 = wasm_i64x2_sub(sum_upper_i64x2, carry_u64x2);
        sum_upper_i64x2 = wasm_i64x2_add(sum_upper_i64x2, wasm_i64x2_shr(data_low_i64x2, 63));
        v128_t data_high_i64x2 = wasm_i64x2_extend_high_i32x4(data_i32x4);
        before_u64x2 = sum_lower_u64x2;
        sum_lower_u64x2 = wasm_i64x2_add(sum_lower_u64x2, data_high_i64x2);
        result_biased_i64x2 = wasm_v128_xor(sum_lower_u64x2, sign_bit_i64x2);
        before_biased_i64x2 = wasm_v128_xor(before_u64x2, sign_bit_i64x2);
        carry_u64x2 = wasm_i64x2_gt(before_biased_i64x2, result_biased_i64x2);
        sum_upper_i64x2 = wasm_i64x2_sub(sum_upper_i64x2, carry_u64x2);
        sum_upper_i64x2 = wasm_i64x2_add(sum_upper_i64x2, wasm_i64x2_shr(data_high_i64x2, 63));
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
    nk_u64_t sumsq = sumsq_overflow ? NK_U64_MAX : nk_reduce_sadd_u64x2_v128relaxed_(sumsq_u64x2);
    nk_b128_vec_t lower_vec, upper_vec;
    lower_vec.v128 = sum_lower_u64x2;
    upper_vec.v128 = sum_upper_i64x2;
    nk_u64_t sum_lower = 0;
    nk_i64_t sum_upper = 0;
    nk_u64_t sum_before = sum_lower;
    sum_lower += lower_vec.u64s[0], sum_upper += (sum_lower < sum_before) + upper_vec.i64s[0];
    sum_before = sum_lower;
    sum_lower += lower_vec.u64s[1], sum_upper += (sum_lower < sum_before) + upper_vec.i64s[1];
    for (; idx < count; ++idx) {
        nk_i64_t val = (nk_i64_t)data[idx];
        sum_before = sum_lower;
        sum_lower += (nk_u64_t)val;
        if (sum_lower < sum_before) sum_upper++;
        sum_upper += (val >> 63);
        nk_u64_t product = (nk_u64_t)(val * val);
        sumsq = nk_u64_saturating_add_serial(sumsq, product);
    }
    nk_i64_t sum_lower_signed = (nk_i64_t)sum_lower;
    if (sum_upper == (sum_lower_signed >> 63)) *sum_ptr = sum_lower_signed;
    else if (sum_upper >= 0) *sum_ptr = NK_I64_MAX;
    else *sum_ptr = NK_I64_MIN;
    *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i32_v128relaxed(                  //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i32_serial(data, count, stride_bytes, sum, sumsq);
    else if (stride_elements == 1) nk_reduce_moments_i32_v128relaxed_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_i32_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_u32_v128relaxed_contiguous_( //
    nk_u32_t const *data, nk_size_t count,                      //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    v128_t sum_u64x2 = wasm_i64x2_splat(0);
    v128_t sumsq_u64x2 = wasm_i64x2_splat(0);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        v128_t data_u32x4 = wasm_v128_load(data + idx);
        sum_u64x2 = wasm_i64x2_add(sum_u64x2, wasm_u64x2_extend_low_u32x4(data_u32x4));
        sum_u64x2 = wasm_i64x2_add(sum_u64x2, wasm_u64x2_extend_high_u32x4(data_u32x4));
        v128_t sq_low_u64x2 = wasm_u64x2_extmul_low_u32x4(data_u32x4, data_u32x4);
        v128_t sq_high_u64x2 = wasm_u64x2_extmul_high_u32x4(data_u32x4, data_u32x4);
        sumsq_u64x2 = nk_u64_sadd_epi64_v128relaxed_(sumsq_u64x2, sq_low_u64x2);
        sumsq_u64x2 = nk_u64_sadd_epi64_v128relaxed_(sumsq_u64x2, sq_high_u64x2);
    }
    nk_u64_t sum = nk_reduce_add_u64x2_v128relaxed_(sum_u64x2);
    nk_u64_t sumsq = nk_reduce_sadd_u64x2_v128relaxed_(sumsq_u64x2);
    for (; idx < count; ++idx) {
        nk_u64_t val = (nk_u64_t)data[idx];
        sum += val;
        nk_u64_t product = val * val;
        sumsq = nk_u64_saturating_add_serial(sumsq, product);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u32_v128relaxed(                  //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u32_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 4) {
        nk_size_t left_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u32_v128relaxed(data, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u32_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                          &right_sum, &right_sumsq);
        *sum = nk_u64_saturating_add_serial(left_sum, right_sum);
        *sumsq = nk_u64_saturating_add_serial(left_sumsq, right_sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u32_v128relaxed_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_u32_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_i64_v128relaxed_contiguous_( //
    nk_i64_t const *data, nk_size_t count,                      //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    v128_t sum_lower_u64x2 = wasm_i64x2_splat(0);
    v128_t sum_upper_i64x2 = wasm_i64x2_splat(0);
    v128_t sumsq_u64x2 = wasm_i64x2_splat(0);
    v128_t sumsq_overflow_u64x2 = wasm_i64x2_splat(0);
    v128_t sign_bit_i64x2 = wasm_i64x2_splat((nk_i64_t)0x8000000000000000LL);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        v128_t data_i64x2 = wasm_v128_load(data + idx);
        v128_t sq_u64x2 = nk_i64_smul_sq_epi64_v128relaxed_(data_i64x2);
        v128_t sq_before_u64x2 = sumsq_u64x2;
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, sq_u64x2);
        sumsq_overflow_u64x2 = wasm_v128_or(
            sumsq_overflow_u64x2,
            wasm_i64x2_gt(wasm_v128_xor(sq_before_u64x2, sign_bit_i64x2), wasm_v128_xor(sumsq_u64x2, sign_bit_i64x2)));
        v128_t before_u64x2 = sum_lower_u64x2;
        sum_lower_u64x2 = wasm_i64x2_add(sum_lower_u64x2, data_i64x2);
        v128_t carry_u64x2 = wasm_i64x2_gt(wasm_v128_xor(before_u64x2, sign_bit_i64x2),
                                           wasm_v128_xor(sum_lower_u64x2, sign_bit_i64x2));
        sum_upper_i64x2 = wasm_i64x2_sub(sum_upper_i64x2, carry_u64x2);
        sum_upper_i64x2 = wasm_i64x2_add(sum_upper_i64x2, wasm_i64x2_shr(data_i64x2, 63));
    }
    int sumsq_overflow = (int)(wasm_i64x2_extract_lane(sumsq_overflow_u64x2, 0) |
                               wasm_i64x2_extract_lane(sumsq_overflow_u64x2, 1));
    nk_u64_t sumsq = sumsq_overflow ? NK_U64_MAX : nk_reduce_sadd_u64x2_v128relaxed_(sumsq_u64x2);
    nk_u64_t sum_lower = (nk_u64_t)wasm_i64x2_extract_lane(sum_lower_u64x2, 0);
    nk_i64_t sum_upper = wasm_i64x2_extract_lane(sum_upper_i64x2, 0);
    {
        nk_u64_t sum_before = sum_lower;
        sum_lower += (nk_u64_t)wasm_i64x2_extract_lane(sum_lower_u64x2, 1);
        if (sum_lower < sum_before) sum_upper++;
        sum_upper += wasm_i64x2_extract_lane(sum_upper_i64x2, 1);
    }
    for (; idx < count; ++idx) {
        nk_i64_t val = data[idx];
        nk_u64_t unsigned_product = (nk_u64_t)nk_i64_saturating_mul_serial(val, val);
        sumsq = nk_u64_saturating_add_serial(sumsq, unsigned_product);
        nk_u64_t sum_before = sum_lower;
        sum_lower += (nk_u64_t)val;
        if (sum_lower < sum_before) sum_upper++;
        sum_upper += (val >> 63);
    }
    nk_i64_t sum_lower_signed = (nk_i64_t)sum_lower;
    if (sum_upper == (sum_lower_signed >> 63)) *sum_ptr = sum_lower_signed;
    else if (sum_upper >= 0) *sum_ptr = NK_I64_MAX;
    else *sum_ptr = NK_I64_MIN;
    *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i64_v128relaxed(                  //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i64_serial(data, count, stride_bytes, sum, sumsq);
    else if (stride_elements == 1) nk_reduce_moments_i64_v128relaxed_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_i64_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_u64_v128relaxed_contiguous_( //
    nk_u64_t const *data, nk_size_t count,                      //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    v128_t sum_u64x2 = wasm_i64x2_splat(0);
    v128_t sumsq_u64x2 = wasm_i64x2_splat(0);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        v128_t data_u64x2 = wasm_v128_load(data + idx);
        sum_u64x2 = nk_u64_sadd_epi64_v128relaxed_(sum_u64x2, data_u64x2);
        v128_t sq_u64x2 = nk_u64_smul_sq_epi64_v128relaxed_(data_u64x2);
        sumsq_u64x2 = nk_u64_sadd_epi64_v128relaxed_(sumsq_u64x2, sq_u64x2);
    }
    nk_u64_t sum = nk_reduce_sadd_u64x2_v128relaxed_(sum_u64x2);
    nk_u64_t sumsq = nk_reduce_sadd_u64x2_v128relaxed_(sumsq_u64x2);
    for (; idx < count; ++idx) {
        nk_u64_t val = data[idx];
        sum = nk_u64_saturating_add_serial(sum, val);
        nk_u64_t product = nk_u64_saturating_mul_serial(val, val);
        sumsq = nk_u64_saturating_add_serial(sumsq, product);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u64_v128relaxed(                  //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u64_serial(data, count, stride_bytes, sum, sumsq);
    else if (stride_elements == 1) nk_reduce_moments_u64_v128relaxed_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_u64_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_i8_v128relaxed_contiguous_( //
    nk_i8_t const *data, nk_size_t count,                     //
    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr,         //
    nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    v128_t min_i8x16 = wasm_i8x16_splat(NK_I8_MAX), max_i8x16 = wasm_i8x16_splat(NK_I8_MIN);
    v128_t min_iter_u8x16 = wasm_i8x16_splat(0), max_iter_u8x16 = wasm_i8x16_splat(0);
    v128_t iter_u8x16 = wasm_i8x16_splat(0), one_u8x16 = wasm_i8x16_splat(1);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        v128_t data_i8x16 = wasm_v128_load(data + idx);
        v128_t less_b8x16 = wasm_i8x16_lt(data_i8x16, min_i8x16);
        v128_t greater_b8x16 = wasm_i8x16_gt(data_i8x16, max_i8x16);
        min_i8x16 = wasm_i8x16_relaxed_laneselect(data_i8x16, min_i8x16, less_b8x16);
        max_i8x16 = wasm_i8x16_relaxed_laneselect(data_i8x16, max_i8x16, greater_b8x16);
        min_iter_u8x16 = wasm_i8x16_relaxed_laneselect(iter_u8x16, min_iter_u8x16, less_b8x16);
        max_iter_u8x16 = wasm_i8x16_relaxed_laneselect(iter_u8x16, max_iter_u8x16, greater_b8x16);
        iter_u8x16 = wasm_i8x16_add(iter_u8x16, one_u8x16);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_iters_vec, max_iters_vec;
    min_values_vec.v128 = min_i8x16;
    max_values_vec.v128 = max_i8x16;
    min_iters_vec.v128 = min_iter_u8x16;
    max_iters_vec.v128 = max_iter_u8x16;
    nk_i8_t min_value = min_values_vec.i8s[0];
    nk_size_t min_idx = (nk_size_t)min_iters_vec.u8s[0] * 16;
    for (int i = 1; i < 16; ++i) {
        nk_size_t abs_idx = (nk_size_t)min_iters_vec.u8s[i] * 16 + (nk_size_t)i;
        if (min_values_vec.i8s[i] < min_value || (min_values_vec.i8s[i] == min_value && abs_idx < min_idx))
            min_value = min_values_vec.i8s[i], min_idx = abs_idx;
    }
    nk_i8_t max_value = max_values_vec.i8s[0];
    nk_size_t max_idx = (nk_size_t)max_iters_vec.u8s[0] * 16;
    for (int i = 1; i < 16; ++i) {
        nk_size_t abs_idx = (nk_size_t)max_iters_vec.u8s[i] * 16 + (nk_size_t)i;
        if (max_values_vec.i8s[i] > max_value || (max_values_vec.i8s[i] == max_value && abs_idx < max_idx))
            max_value = max_values_vec.i8s[i], max_idx = abs_idx;
    }
    for (; idx < count; ++idx) {
        nk_i8_t val = data[idx];
        if (val < min_value) min_value = val, min_idx = idx;
        if (val > max_value) max_value = val, max_idx = idx;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i8_v128relaxed(                   //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_I8_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I8_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i8_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                   max_index_ptr);
    else if (count > (nk_size_t)(NK_U8_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_i8_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_idx, right_min_idx, left_max_idx, right_max_idx;
        nk_reduce_minmax_i8_v128relaxed(data, left_count, stride_bytes, &left_min, &left_min_idx, &left_max,
                                        &left_max_idx);
        nk_reduce_minmax_i8_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                        &right_min, &right_min_idx, &right_max, &right_max_idx);
        if (right_min < left_min) *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_idx;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_idx;
        if (right_max > left_max) *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_idx;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_idx;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i8_v128relaxed_contiguous_(data, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                    max_index_ptr);
    else
        nk_reduce_minmax_i8_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                   max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_u8_v128relaxed_contiguous_( //
    nk_u8_t const *data, nk_size_t count,                     //
    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,         //
    nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    v128_t min_u8x16 = wasm_i8x16_splat((nk_i8_t)NK_U8_MAX), max_u8x16 = wasm_i8x16_splat(0);
    v128_t min_iter_u8x16 = wasm_i8x16_splat(0), max_iter_u8x16 = wasm_i8x16_splat(0);
    v128_t iter_u8x16 = wasm_i8x16_splat(0), one_u8x16 = wasm_i8x16_splat(1);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        v128_t data_u8x16 = wasm_v128_load(data + idx);
        v128_t less_b8x16 = wasm_u8x16_lt(data_u8x16, min_u8x16);
        v128_t greater_b8x16 = wasm_u8x16_gt(data_u8x16, max_u8x16);
        min_u8x16 = wasm_i8x16_relaxed_laneselect(data_u8x16, min_u8x16, less_b8x16);
        max_u8x16 = wasm_i8x16_relaxed_laneselect(data_u8x16, max_u8x16, greater_b8x16);
        min_iter_u8x16 = wasm_i8x16_relaxed_laneselect(iter_u8x16, min_iter_u8x16, less_b8x16);
        max_iter_u8x16 = wasm_i8x16_relaxed_laneselect(iter_u8x16, max_iter_u8x16, greater_b8x16);
        iter_u8x16 = wasm_i8x16_add(iter_u8x16, one_u8x16);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_iters_vec, max_iters_vec;
    min_values_vec.v128 = min_u8x16;
    max_values_vec.v128 = max_u8x16;
    min_iters_vec.v128 = min_iter_u8x16;
    max_iters_vec.v128 = max_iter_u8x16;
    nk_u8_t min_value = min_values_vec.u8s[0];
    nk_size_t min_idx = (nk_size_t)min_iters_vec.u8s[0] * 16;
    for (int i = 1; i < 16; ++i) {
        nk_size_t abs_idx = (nk_size_t)min_iters_vec.u8s[i] * 16 + (nk_size_t)i;
        if (min_values_vec.u8s[i] < min_value || (min_values_vec.u8s[i] == min_value && abs_idx < min_idx))
            min_value = min_values_vec.u8s[i], min_idx = abs_idx;
    }
    nk_u8_t max_value = max_values_vec.u8s[0];
    nk_size_t max_idx = (nk_size_t)max_iters_vec.u8s[0] * 16;
    for (int i = 1; i < 16; ++i) {
        nk_size_t abs_idx = (nk_size_t)max_iters_vec.u8s[i] * 16 + (nk_size_t)i;
        if (max_values_vec.u8s[i] > max_value || (max_values_vec.u8s[i] == max_value && abs_idx < max_idx))
            max_value = max_values_vec.u8s[i], max_idx = abs_idx;
    }
    for (; idx < count; ++idx) {
        nk_u8_t val = data[idx];
        if (val < min_value) min_value = val, min_idx = idx;
        if (val > max_value) max_value = val, max_idx = idx;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u8_v128relaxed(                   //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_U8_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = 0, *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u8_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                   max_index_ptr);
    else if (count > (nk_size_t)(NK_U8_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_u8_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_idx, right_min_idx, left_max_idx, right_max_idx;
        nk_reduce_minmax_u8_v128relaxed(data, left_count, stride_bytes, &left_min, &left_min_idx, &left_max,
                                        &left_max_idx);
        nk_reduce_minmax_u8_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                        &right_min, &right_min_idx, &right_max, &right_max_idx);
        if (right_min < left_min) *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_idx;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_idx;
        if (right_max > left_max) *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_idx;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_idx;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u8_v128relaxed_contiguous_(data, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                    max_index_ptr);
    else
        nk_reduce_minmax_u8_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                   max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_i16_v128relaxed_contiguous_( //
    nk_i16_t const *data, nk_size_t count,                     //
    nk_i16_t *min_value_ptr, nk_size_t *min_index_ptr,         //
    nk_i16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    v128_t min_i16x8 = wasm_i16x8_splat(NK_I16_MAX), max_i16x8 = wasm_i16x8_splat(NK_I16_MIN);
    v128_t min_iter_u16x8 = wasm_i16x8_splat(0), max_iter_u16x8 = wasm_i16x8_splat(0);
    v128_t iter_u16x8 = wasm_i16x8_splat(0), one_u16x8 = wasm_i16x8_splat(1);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        v128_t data_i16x8 = wasm_v128_load(data + idx);
        v128_t less_b16x8 = wasm_i16x8_lt(data_i16x8, min_i16x8);
        v128_t greater_b16x8 = wasm_i16x8_gt(data_i16x8, max_i16x8);
        min_i16x8 = wasm_i16x8_relaxed_laneselect(data_i16x8, min_i16x8, less_b16x8);
        max_i16x8 = wasm_i16x8_relaxed_laneselect(data_i16x8, max_i16x8, greater_b16x8);
        min_iter_u16x8 = wasm_i16x8_relaxed_laneselect(iter_u16x8, min_iter_u16x8, less_b16x8);
        max_iter_u16x8 = wasm_i16x8_relaxed_laneselect(iter_u16x8, max_iter_u16x8, greater_b16x8);
        iter_u16x8 = wasm_i16x8_add(iter_u16x8, one_u16x8);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_iters_vec, max_iters_vec;
    min_values_vec.v128 = min_i16x8;
    max_values_vec.v128 = max_i16x8;
    min_iters_vec.v128 = min_iter_u16x8;
    max_iters_vec.v128 = max_iter_u16x8;
    nk_i16_t min_value = min_values_vec.i16s[0];
    nk_size_t min_idx = (nk_size_t)min_iters_vec.u16s[0] * 8;
    for (int i = 1; i < 8; ++i) {
        nk_size_t abs_idx = (nk_size_t)min_iters_vec.u16s[i] * 8 + (nk_size_t)i;
        if (min_values_vec.i16s[i] < min_value || (min_values_vec.i16s[i] == min_value && abs_idx < min_idx))
            min_value = min_values_vec.i16s[i], min_idx = abs_idx;
    }
    nk_i16_t max_value = max_values_vec.i16s[0];
    nk_size_t max_idx = (nk_size_t)max_iters_vec.u16s[0] * 8;
    for (int i = 1; i < 8; ++i) {
        nk_size_t abs_idx = (nk_size_t)max_iters_vec.u16s[i] * 8 + (nk_size_t)i;
        if (max_values_vec.i16s[i] > max_value || (max_values_vec.i16s[i] == max_value && abs_idx < max_idx))
            max_value = max_values_vec.i16s[i], max_idx = abs_idx;
    }
    for (; idx < count; ++idx) {
        nk_i16_t val = data[idx];
        if (val < min_value) min_value = val, min_idx = idx;
        if (val > max_value) max_value = val, max_idx = idx;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i16_v128relaxed(                   //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_i16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_I16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i16_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_i16_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_idx, right_min_idx, left_max_idx, right_max_idx;
        nk_reduce_minmax_i16_v128relaxed(data, left_count, stride_bytes, &left_min, &left_min_idx, &left_max,
                                         &left_max_idx);
        nk_reduce_minmax_i16_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                         &right_min, &right_min_idx, &right_max, &right_max_idx);
        if (right_min < left_min) *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_idx;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_idx;
        if (right_max > left_max) *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_idx;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_idx;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i16_v128relaxed_contiguous_(data, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                     max_index_ptr);
    else
        nk_reduce_minmax_i16_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_u16_v128relaxed_contiguous_( //
    nk_u16_t const *data, nk_size_t count,                     //
    nk_u16_t *min_value_ptr, nk_size_t *min_index_ptr,         //
    nk_u16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    v128_t min_u16x8 = wasm_i16x8_splat((nk_i16_t)NK_U16_MAX), max_u16x8 = wasm_i16x8_splat(0);
    v128_t min_iter_u16x8 = wasm_i16x8_splat(0), max_iter_u16x8 = wasm_i16x8_splat(0);
    v128_t iter_u16x8 = wasm_i16x8_splat(0), one_u16x8 = wasm_i16x8_splat(1);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        v128_t data_u16x8 = wasm_v128_load(data + idx);
        v128_t less_b16x8 = wasm_u16x8_lt(data_u16x8, min_u16x8);
        v128_t greater_b16x8 = wasm_u16x8_gt(data_u16x8, max_u16x8);
        min_u16x8 = wasm_i16x8_relaxed_laneselect(data_u16x8, min_u16x8, less_b16x8);
        max_u16x8 = wasm_i16x8_relaxed_laneselect(data_u16x8, max_u16x8, greater_b16x8);
        min_iter_u16x8 = wasm_i16x8_relaxed_laneselect(iter_u16x8, min_iter_u16x8, less_b16x8);
        max_iter_u16x8 = wasm_i16x8_relaxed_laneselect(iter_u16x8, max_iter_u16x8, greater_b16x8);
        iter_u16x8 = wasm_i16x8_add(iter_u16x8, one_u16x8);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_iters_vec, max_iters_vec;
    min_values_vec.v128 = min_u16x8;
    max_values_vec.v128 = max_u16x8;
    min_iters_vec.v128 = min_iter_u16x8;
    max_iters_vec.v128 = max_iter_u16x8;
    nk_u16_t min_value = min_values_vec.u16s[0];
    nk_size_t min_idx = (nk_size_t)min_iters_vec.u16s[0] * 8;
    for (int i = 1; i < 8; ++i) {
        nk_size_t abs_idx = (nk_size_t)min_iters_vec.u16s[i] * 8 + (nk_size_t)i;
        if (min_values_vec.u16s[i] < min_value || (min_values_vec.u16s[i] == min_value && abs_idx < min_idx))
            min_value = min_values_vec.u16s[i], min_idx = abs_idx;
    }
    nk_u16_t max_value = max_values_vec.u16s[0];
    nk_size_t max_idx = (nk_size_t)max_iters_vec.u16s[0] * 8;
    for (int i = 1; i < 8; ++i) {
        nk_size_t abs_idx = (nk_size_t)max_iters_vec.u16s[i] * 8 + (nk_size_t)i;
        if (max_values_vec.u16s[i] > max_value || (max_values_vec.u16s[i] == max_value && abs_idx < max_idx))
            max_value = max_values_vec.u16s[i], max_idx = abs_idx;
    }
    for (; idx < count; ++idx) {
        nk_u16_t val = data[idx];
        if (val < min_value) min_value = val, min_idx = idx;
        if (val > max_value) max_value = val, max_idx = idx;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u16_v128relaxed(                   //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_u16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_U16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = 0, *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u16_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_u16_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_idx, right_min_idx, left_max_idx, right_max_idx;
        nk_reduce_minmax_u16_v128relaxed(data, left_count, stride_bytes, &left_min, &left_min_idx, &left_max,
                                         &left_max_idx);
        nk_reduce_minmax_u16_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                         &right_min, &right_min_idx, &right_max, &right_max_idx);
        if (right_min < left_min) *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_idx;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_idx;
        if (right_max > left_max) *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_idx;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_idx;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u16_v128relaxed_contiguous_(data, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                     max_index_ptr);
    else
        nk_reduce_minmax_u16_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_i32_v128relaxed_contiguous_( //
    nk_i32_t const *data, nk_size_t count,                     //
    nk_i32_t *min_value_ptr, nk_size_t *min_index_ptr,         //
    nk_i32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    v128_t min_i32x4 = wasm_i32x4_splat(NK_I32_MAX), max_i32x4 = wasm_i32x4_splat(NK_I32_MIN);
    v128_t min_iter_u32x4 = wasm_i32x4_splat(0), max_iter_u32x4 = wasm_i32x4_splat(0);
    v128_t iter_u32x4 = wasm_i32x4_splat(0), one_u32x4 = wasm_i32x4_splat(1);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        v128_t data_i32x4 = wasm_v128_load(data + idx);
        v128_t less_b32x4 = wasm_i32x4_lt(data_i32x4, min_i32x4);
        v128_t greater_b32x4 = wasm_i32x4_gt(data_i32x4, max_i32x4);
        min_i32x4 = wasm_i32x4_relaxed_laneselect(data_i32x4, min_i32x4, less_b32x4);
        max_i32x4 = wasm_i32x4_relaxed_laneselect(data_i32x4, max_i32x4, greater_b32x4);
        min_iter_u32x4 = wasm_i32x4_relaxed_laneselect(iter_u32x4, min_iter_u32x4, less_b32x4);
        max_iter_u32x4 = wasm_i32x4_relaxed_laneselect(iter_u32x4, max_iter_u32x4, greater_b32x4);
        iter_u32x4 = wasm_i32x4_add(iter_u32x4, one_u32x4);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_iters_vec, max_iters_vec;
    min_values_vec.v128 = min_i32x4;
    max_values_vec.v128 = max_i32x4;
    min_iters_vec.v128 = min_iter_u32x4;
    max_iters_vec.v128 = max_iter_u32x4;
    nk_i32_t min_value = min_values_vec.i32s[0];
    nk_size_t min_idx = (nk_size_t)min_iters_vec.u32s[0] * 4;
    for (int i = 1; i < 4; ++i) {
        nk_size_t abs_idx = (nk_size_t)min_iters_vec.u32s[i] * 4 + (nk_size_t)i;
        if (min_values_vec.i32s[i] < min_value || (min_values_vec.i32s[i] == min_value && abs_idx < min_idx))
            min_value = min_values_vec.i32s[i], min_idx = abs_idx;
    }
    nk_i32_t max_value = max_values_vec.i32s[0];
    nk_size_t max_idx = (nk_size_t)max_iters_vec.u32s[0] * 4;
    for (int i = 1; i < 4; ++i) {
        nk_size_t abs_idx = (nk_size_t)max_iters_vec.u32s[i] * 4 + (nk_size_t)i;
        if (max_values_vec.i32s[i] > max_value || (max_values_vec.i32s[i] == max_value && abs_idx < max_idx))
            max_value = max_values_vec.i32s[i], max_idx = abs_idx;
    }
    for (; idx < count; ++idx) {
        nk_i32_t val = data[idx];
        if (val < min_value) min_value = val, min_idx = idx;
        if (val > max_value) max_value = val, max_idx = idx;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i32_v128relaxed(                   //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_i32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_I32_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I32_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i32_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)NK_U32_MAX * 4) {
        nk_size_t left_count = count / 2;
        nk_i32_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_idx, right_min_idx, left_max_idx, right_max_idx;
        nk_reduce_minmax_i32_v128relaxed(data, left_count, stride_bytes, &left_min, &left_min_idx, &left_max,
                                         &left_max_idx);
        nk_reduce_minmax_i32_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                         &right_min, &right_min_idx, &right_max, &right_max_idx);
        if (right_min < left_min) *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_idx;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_idx;
        if (right_max > left_max) *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_idx;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_idx;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i32_v128relaxed_contiguous_(data, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                     max_index_ptr);
    else
        nk_reduce_minmax_i32_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_u32_v128relaxed_contiguous_( //
    nk_u32_t const *data, nk_size_t count,                     //
    nk_u32_t *min_value_ptr, nk_size_t *min_index_ptr,         //
    nk_u32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    v128_t min_u32x4 = wasm_i32x4_splat((nk_i32_t)NK_U32_MAX), max_u32x4 = wasm_i32x4_splat(0);
    v128_t min_iter_u32x4 = wasm_i32x4_splat(0), max_iter_u32x4 = wasm_i32x4_splat(0);
    v128_t iter_u32x4 = wasm_i32x4_splat(0), one_u32x4 = wasm_i32x4_splat(1);
    v128_t sign_bit_i32x4 = wasm_i32x4_splat((nk_i32_t)0x80000000);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        v128_t data_u32x4 = wasm_v128_load(data + idx);
        v128_t data_biased_i32x4 = wasm_v128_xor(data_u32x4, sign_bit_i32x4);
        v128_t min_biased_i32x4 = wasm_v128_xor(min_u32x4, sign_bit_i32x4);
        v128_t max_biased_i32x4 = wasm_v128_xor(max_u32x4, sign_bit_i32x4);
        v128_t less_b32x4 = wasm_i32x4_lt(data_biased_i32x4, min_biased_i32x4);
        v128_t greater_b32x4 = wasm_i32x4_gt(data_biased_i32x4, max_biased_i32x4);
        min_u32x4 = wasm_i32x4_relaxed_laneselect(data_u32x4, min_u32x4, less_b32x4);
        max_u32x4 = wasm_i32x4_relaxed_laneselect(data_u32x4, max_u32x4, greater_b32x4);
        min_iter_u32x4 = wasm_i32x4_relaxed_laneselect(iter_u32x4, min_iter_u32x4, less_b32x4);
        max_iter_u32x4 = wasm_i32x4_relaxed_laneselect(iter_u32x4, max_iter_u32x4, greater_b32x4);
        iter_u32x4 = wasm_i32x4_add(iter_u32x4, one_u32x4);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_iters_vec, max_iters_vec;
    min_values_vec.v128 = min_u32x4;
    max_values_vec.v128 = max_u32x4;
    min_iters_vec.v128 = min_iter_u32x4;
    max_iters_vec.v128 = max_iter_u32x4;
    nk_u32_t min_value = min_values_vec.u32s[0];
    nk_size_t min_idx = (nk_size_t)min_iters_vec.u32s[0] * 4;
    for (int i = 1; i < 4; ++i) {
        nk_size_t abs_idx = (nk_size_t)min_iters_vec.u32s[i] * 4 + (nk_size_t)i;
        if (min_values_vec.u32s[i] < min_value || (min_values_vec.u32s[i] == min_value && abs_idx < min_idx))
            min_value = min_values_vec.u32s[i], min_idx = abs_idx;
    }
    nk_u32_t max_value = max_values_vec.u32s[0];
    nk_size_t max_idx = (nk_size_t)max_iters_vec.u32s[0] * 4;
    for (int i = 1; i < 4; ++i) {
        nk_size_t abs_idx = (nk_size_t)max_iters_vec.u32s[i] * 4 + (nk_size_t)i;
        if (max_values_vec.u32s[i] > max_value || (max_values_vec.u32s[i] == max_value && abs_idx < max_idx))
            max_value = max_values_vec.u32s[i], max_idx = abs_idx;
    }
    for (; idx < count; ++idx) {
        nk_u32_t val = data[idx];
        if (val < min_value) min_value = val, min_idx = idx;
        if (val > max_value) max_value = val, max_idx = idx;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u32_v128relaxed(                   //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_u32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_U32_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = 0, *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u32_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)NK_U32_MAX * 4) {
        nk_size_t left_count = count / 2;
        nk_u32_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_idx, right_min_idx, left_max_idx, right_max_idx;
        nk_reduce_minmax_u32_v128relaxed(data, left_count, stride_bytes, &left_min, &left_min_idx, &left_max,
                                         &left_max_idx);
        nk_reduce_minmax_u32_v128relaxed(data + left_count * stride_elements, count - left_count, stride_bytes,
                                         &right_min, &right_min_idx, &right_max, &right_max_idx);
        if (right_min < left_min) *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_idx;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_idx;
        if (right_max > left_max) *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_idx;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_idx;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u32_v128relaxed_contiguous_(data, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                     max_index_ptr);
    else
        nk_reduce_minmax_u32_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_i64_v128relaxed_contiguous_( //
    nk_i64_t const *data, nk_size_t count,                     //
    nk_i64_t *min_value_ptr, nk_size_t *min_index_ptr,         //
    nk_i64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    v128_t min_i64x2 = wasm_i64x2_splat(NK_I64_MAX), max_i64x2 = wasm_i64x2_splat(NK_I64_MIN);
    v128_t min_iter_u64x2 = wasm_i64x2_splat(0), max_iter_u64x2 = wasm_i64x2_splat(0);
    v128_t iter_u64x2 = wasm_i64x2_splat(0), one_u64x2 = wasm_i64x2_splat(1);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        v128_t data_i64x2 = wasm_v128_load(data + idx);
        v128_t less_b64x2 = wasm_i64x2_gt(min_i64x2, data_i64x2);
        v128_t greater_b64x2 = wasm_i64x2_gt(data_i64x2, max_i64x2);
        min_i64x2 = wasm_i64x2_relaxed_laneselect(data_i64x2, min_i64x2, less_b64x2);
        max_i64x2 = wasm_i64x2_relaxed_laneselect(data_i64x2, max_i64x2, greater_b64x2);
        min_iter_u64x2 = wasm_i64x2_relaxed_laneselect(iter_u64x2, min_iter_u64x2, less_b64x2);
        max_iter_u64x2 = wasm_i64x2_relaxed_laneselect(iter_u64x2, max_iter_u64x2, greater_b64x2);
        iter_u64x2 = wasm_i64x2_add(iter_u64x2, one_u64x2);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_iters_vec, max_iters_vec;
    min_values_vec.v128 = min_i64x2;
    max_values_vec.v128 = max_i64x2;
    min_iters_vec.v128 = min_iter_u64x2;
    max_iters_vec.v128 = max_iter_u64x2;
    nk_i64_t min_value = min_values_vec.i64s[0];
    nk_size_t min_idx = (nk_size_t)min_iters_vec.u64s[0] * 2;
    if (min_values_vec.i64s[1] < min_value ||
        (min_values_vec.i64s[1] == min_value && (nk_size_t)min_iters_vec.u64s[1] * 2 + 1 < min_idx))
        min_value = min_values_vec.i64s[1], min_idx = (nk_size_t)min_iters_vec.u64s[1] * 2 + 1;
    nk_i64_t max_value = max_values_vec.i64s[0];
    nk_size_t max_idx = (nk_size_t)max_iters_vec.u64s[0] * 2;
    if (max_values_vec.i64s[1] > max_value ||
        (max_values_vec.i64s[1] == max_value && (nk_size_t)max_iters_vec.u64s[1] * 2 + 1 < max_idx))
        max_value = max_values_vec.i64s[1], max_idx = (nk_size_t)max_iters_vec.u64s[1] * 2 + 1;
    for (; idx < count; ++idx) {
        nk_i64_t val = data[idx];
        if (val < min_value) min_value = val, min_idx = idx;
        if (val > max_value) max_value = val, max_idx = idx;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i64_v128relaxed(                   //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_i64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_I64_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I64_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i64_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_i64_v128relaxed_contiguous_(data, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                     max_index_ptr);
    else
        nk_reduce_minmax_i64_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_u64_v128relaxed_contiguous_( //
    nk_u64_t const *data, nk_size_t count,                     //
    nk_u64_t *min_value_ptr, nk_size_t *min_index_ptr,         //
    nk_u64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    v128_t min_u64x2 = wasm_i64x2_splat((nk_i64_t)NK_U64_MAX), max_u64x2 = wasm_i64x2_splat(0);
    v128_t min_iter_u64x2 = wasm_i64x2_splat(0), max_iter_u64x2 = wasm_i64x2_splat(0);
    v128_t iter_u64x2 = wasm_i64x2_splat(0), one_u64x2 = wasm_i64x2_splat(1);
    v128_t sign_bit_i64x2 = wasm_i64x2_splat((nk_i64_t)0x8000000000000000LL);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        v128_t data_u64x2 = wasm_v128_load(data + idx);
        v128_t data_biased_i64x2 = wasm_v128_xor(data_u64x2, sign_bit_i64x2);
        v128_t min_biased_i64x2 = wasm_v128_xor(min_u64x2, sign_bit_i64x2);
        v128_t max_biased_i64x2 = wasm_v128_xor(max_u64x2, sign_bit_i64x2);
        v128_t less_b64x2 = wasm_i64x2_gt(min_biased_i64x2, data_biased_i64x2);
        v128_t greater_b64x2 = wasm_i64x2_gt(data_biased_i64x2, max_biased_i64x2);
        min_u64x2 = wasm_i64x2_relaxed_laneselect(data_u64x2, min_u64x2, less_b64x2);
        max_u64x2 = wasm_i64x2_relaxed_laneselect(data_u64x2, max_u64x2, greater_b64x2);
        min_iter_u64x2 = wasm_i64x2_relaxed_laneselect(iter_u64x2, min_iter_u64x2, less_b64x2);
        max_iter_u64x2 = wasm_i64x2_relaxed_laneselect(iter_u64x2, max_iter_u64x2, greater_b64x2);
        iter_u64x2 = wasm_i64x2_add(iter_u64x2, one_u64x2);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_iters_vec, max_iters_vec;
    min_values_vec.v128 = min_u64x2;
    max_values_vec.v128 = max_u64x2;
    min_iters_vec.v128 = min_iter_u64x2;
    max_iters_vec.v128 = max_iter_u64x2;
    nk_u64_t min_value = min_values_vec.u64s[0];
    nk_size_t min_idx = (nk_size_t)min_iters_vec.u64s[0] * 2;
    if (min_values_vec.u64s[1] < min_value ||
        (min_values_vec.u64s[1] == min_value && (nk_size_t)min_iters_vec.u64s[1] * 2 + 1 < min_idx))
        min_value = min_values_vec.u64s[1], min_idx = (nk_size_t)min_iters_vec.u64s[1] * 2 + 1;
    nk_u64_t max_value = max_values_vec.u64s[0];
    nk_size_t max_idx = (nk_size_t)max_iters_vec.u64s[0] * 2;
    if (max_values_vec.u64s[1] > max_value ||
        (max_values_vec.u64s[1] == max_value && (nk_size_t)max_iters_vec.u64s[1] * 2 + 1 < max_idx))
        max_value = max_values_vec.u64s[1], max_idx = (nk_size_t)max_iters_vec.u64s[1] * 2 + 1;
    for (; idx < count; ++idx) {
        nk_u64_t val = data[idx];
        if (val < min_value) min_value = val, min_idx = idx;
        if (val > max_value) max_value = val, max_idx = idx;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u64_v128relaxed(                   //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_u64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);
    if (count == 0)
        *min_value_ptr = NK_U64_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = 0, *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u64_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_u64_v128relaxed_contiguous_(data, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                     max_index_ptr);
    else
        nk_reduce_minmax_u64_serial(data, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_e4m3_v128relaxed_contiguous_( //
    nk_e4m3_t const *data_ptr, nk_size_t count,                  //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0), sumsq_f32x4 = wasm_f32x4_splat(0);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        nk_b32_vec_t raw;
        nk_load_b32_serial_(data_ptr + idx, &raw);
        v128_t data_f32x4 = nk_e4m3x4_to_f32x4_v128relaxed_(raw).v128;
        sum_f32x4 = wasm_f32x4_add(sum_f32x4, data_f32x4);
        sumsq_f32x4 = wasm_f32x4_relaxed_madd(data_f32x4, data_f32x4, sumsq_f32x4);
    }
    nk_f32_t sum = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
    nk_f32_t sumsq = nk_reduce_add_f32x4_v128relaxed_(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_e4m3_to_f32_serial(&data_ptr[idx], &val);
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_e4m3_v128relaxed(                      //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e4m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_e4m3_v128relaxed_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e4m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_e2m3_v128relaxed_contiguous_( //
    nk_e2m3_t const *data_ptr, nk_size_t count,                  //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    v128_t const lut_low_u8x16 = wasm_i8x16_const(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    v128_t const lut_high_u8x16 = wasm_i8x16_const(32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120);
    v128_t const magnitude_mask_u8x16 = wasm_i8x16_splat(0x1F);
    v128_t const sign_mask_u8x16 = wasm_i8x16_splat(0x20);
    v128_t const sixteen_u8x16 = wasm_i8x16_splat(16);
    v128_t sum_i32x4 = wasm_i32x4_splat(0);
    v128_t sumsq_u64x2 = wasm_i64x2_splat(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        v128_t raw_u8x16 = wasm_v128_load(data_ptr + idx);
        v128_t magnitude_u8x16 = wasm_v128_and(raw_u8x16, magnitude_mask_u8x16);
        v128_t from_low_u8x16 = wasm_i8x16_relaxed_swizzle(lut_low_u8x16, magnitude_u8x16);
        v128_t high_indices_u8x16 = wasm_i8x16_sub(magnitude_u8x16, sixteen_u8x16);
        v128_t from_high_u8x16 = wasm_i8x16_relaxed_swizzle(lut_high_u8x16, high_indices_u8x16);
        v128_t in_high_b8x16 = wasm_u8x16_ge(magnitude_u8x16, sixteen_u8x16);
        v128_t unsigned_u8x16 = wasm_i8x16_relaxed_laneselect(from_high_u8x16, from_low_u8x16, in_high_b8x16);
        v128_t is_negative_b8x16 = wasm_i8x16_eq(wasm_v128_and(raw_u8x16, sign_mask_u8x16), sign_mask_u8x16);
        v128_t negated_i8x16 = wasm_i8x16_sub(wasm_i8x16_splat(0), unsigned_u8x16);
        v128_t scaled_i8x16 = wasm_i8x16_relaxed_laneselect(negated_i8x16, unsigned_u8x16, is_negative_b8x16);
        v128_t pairwise_i16x8 = wasm_i16x8_extadd_pairwise_i8x16(scaled_i8x16);
        v128_t pairwise_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(pairwise_i16x8);
        sum_i32x4 = wasm_i32x4_add(sum_i32x4, pairwise_i32x4);
        v128_t sq_low_i16x8 = wasm_i16x8_extmul_low_i8x16(scaled_i8x16, scaled_i8x16);
        v128_t sq_high_i16x8 = wasm_i16x8_extmul_high_i8x16(scaled_i8x16, scaled_i8x16);
        v128_t sq_low_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(sq_low_i16x8);
        v128_t sq_high_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(sq_high_i16x8);
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_low_u32x4(sq_low_u32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_high_u32x4(sq_low_u32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_low_u32x4(sq_high_u32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_high_u32x4(sq_high_u32x4));
    }
    nk_i64_t sum = nk_reduce_add_i32x4_v128relaxed_(sum_i32x4);
    nk_u64_t sumsq = nk_reduce_add_u64x2_v128relaxed_(sumsq_u64x2);
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_e2m3_to_f32_serial(&data_ptr[idx], &val);
        sum += (nk_i64_t)(val * 16.0f), sumsq += (nk_u64_t)(nk_i64_t)(val * val * 256.0f);
    }
    *sum_ptr = (nk_f32_t)sum / 16.0f, *sumsq_ptr = (nk_f32_t)sumsq / 256.0f;
}

NK_PUBLIC void nk_reduce_moments_e2m3_v128relaxed(                      //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    int aligned = (stride_bytes % sizeof(nk_e2m3_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e2m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e2m3_v128relaxed(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e2m3_v128relaxed(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                           &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum, *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e2m3_v128relaxed_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e2m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_e3m2_v128relaxed_contiguous_( //
    nk_e3m2_t const *data_ptr, nk_size_t count,                  //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    v128_t const lut_low_u8x16 = wasm_i8x16_const(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28);
    v128_t const lut_high_u8x16 = wasm_i8x16_const(32, 40, 48, 56, 64, 80, 96, 112, (nk_i8_t)-128, (nk_i8_t)-96,
                                                   (nk_i8_t)-64, (nk_i8_t)-32, 0, 64, (nk_i8_t)-128, (nk_i8_t)-96);
    v128_t const magnitude_mask_u8x16 = wasm_i8x16_splat(0x1F);
    v128_t const sign_mask_u8x16 = wasm_i8x16_splat(0x20);
    v128_t const sixteen_u8x16 = wasm_i8x16_splat(16);
    v128_t const threshold_u8x16 = wasm_i8x16_splat(27);
    v128_t sum_i32x4 = wasm_i32x4_splat(0);
    v128_t sumsq_u64x2 = wasm_i64x2_splat(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        v128_t raw_u8x16 = wasm_v128_load(data_ptr + idx);
        v128_t magnitude_u8x16 = wasm_v128_and(raw_u8x16, magnitude_mask_u8x16);
        v128_t from_low_u8x16 = wasm_i8x16_relaxed_swizzle(lut_low_u8x16, magnitude_u8x16);
        v128_t high_indices_u8x16 = wasm_i8x16_sub(magnitude_u8x16, sixteen_u8x16);
        v128_t from_high_u8x16 = wasm_i8x16_relaxed_swizzle(lut_high_u8x16, high_indices_u8x16);
        v128_t in_high_b8x16 = wasm_u8x16_ge(magnitude_u8x16, sixteen_u8x16);
        v128_t low_byte_u8x16 = wasm_i8x16_relaxed_laneselect(from_high_u8x16, from_low_u8x16, in_high_b8x16);
        v128_t high_byte_u8x16 = wasm_v128_and(wasm_u8x16_gt(magnitude_u8x16, threshold_u8x16), wasm_i8x16_splat(1));
        v128_t is_negative_b8x16 = wasm_i8x16_eq(wasm_v128_and(raw_u8x16, sign_mask_u8x16), sign_mask_u8x16);
        v128_t unsigned_low_u16x8 = wasm_i8x16_shuffle(low_byte_u8x16, high_byte_u8x16, 0, 16, 1, 17, 2, 18, 3, 19, 4,
                                                       20, 5, 21, 6, 22, 7, 23);
        v128_t unsigned_high_u16x8 = wasm_i8x16_shuffle(low_byte_u8x16, high_byte_u8x16, 8, 24, 9, 25, 10, 26, 11, 27,
                                                        12, 28, 13, 29, 14, 30, 15, 31);
        v128_t is_neg_low_i16x8 = wasm_i16x8_extend_low_i8x16(is_negative_b8x16);
        v128_t is_neg_high_i16x8 = wasm_i16x8_extend_high_i8x16(is_negative_b8x16);
        v128_t neg_low_i16x8 = wasm_i16x8_neg(unsigned_low_u16x8);
        v128_t scaled_low_i16x8 = wasm_i16x8_relaxed_laneselect(neg_low_i16x8, unsigned_low_u16x8, is_neg_low_i16x8);
        v128_t neg_high_i16x8 = wasm_i16x8_neg(unsigned_high_u16x8);
        v128_t scaled_high_i16x8 = wasm_i16x8_relaxed_laneselect(neg_high_i16x8, unsigned_high_u16x8,
                                                                 is_neg_high_i16x8);
        v128_t sum_low_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(scaled_low_i16x8);
        v128_t sum_high_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(scaled_high_i16x8);
        sum_i32x4 = wasm_i32x4_add(sum_i32x4, sum_low_i32x4);
        sum_i32x4 = wasm_i32x4_add(sum_i32x4, sum_high_i32x4);
        v128_t sq_low_a_i32x4 = wasm_i32x4_extmul_low_i16x8(scaled_low_i16x8, scaled_low_i16x8);
        v128_t sq_low_b_i32x4 = wasm_i32x4_extmul_high_i16x8(scaled_low_i16x8, scaled_low_i16x8);
        v128_t sq_high_a_i32x4 = wasm_i32x4_extmul_low_i16x8(scaled_high_i16x8, scaled_high_i16x8);
        v128_t sq_high_b_i32x4 = wasm_i32x4_extmul_high_i16x8(scaled_high_i16x8, scaled_high_i16x8);
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_low_u32x4(sq_low_a_i32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_high_u32x4(sq_low_a_i32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_low_u32x4(sq_low_b_i32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_high_u32x4(sq_low_b_i32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_low_u32x4(sq_high_a_i32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_high_u32x4(sq_high_a_i32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_low_u32x4(sq_high_b_i32x4));
        sumsq_u64x2 = wasm_i64x2_add(sumsq_u64x2, wasm_u64x2_extend_high_u32x4(sq_high_b_i32x4));
    }
    nk_i64_t sum = nk_reduce_add_i32x4_v128relaxed_(sum_i32x4);
    nk_u64_t sumsq = nk_reduce_add_u64x2_v128relaxed_(sumsq_u64x2);
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_e3m2_to_f32_serial(&data_ptr[idx], &val);
        sum += (nk_i64_t)(val * 16.0f), sumsq += (nk_u64_t)(nk_i64_t)(val * val * 256.0f);
    }
    *sum_ptr = (nk_f32_t)sum / 16.0f, *sumsq_ptr = (nk_f32_t)sumsq / 256.0f;
}

NK_PUBLIC void nk_reduce_moments_e3m2_v128relaxed(                      //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    int aligned = (stride_bytes % sizeof(nk_e3m2_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e3m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e3m2_v128relaxed(data_ptr, left_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e3m2_v128relaxed(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                           &right_sum, &right_sumsq);
        *sum_ptr = left_sum + right_sum, *sumsq_ptr = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e3m2_v128relaxed_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e3m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_moments_e5m2_v128relaxed_contiguous_( //
    nk_e5m2_t const *data_ptr, nk_size_t count,                  //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0), sumsq_f32x4 = wasm_f32x4_splat(0);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        nk_b32_vec_t raw;
        nk_load_b32_serial_(data_ptr + idx, &raw);
        v128_t data_f32x4 = nk_e5m2x4_to_f32x4_v128relaxed_(raw).v128;
        sum_f32x4 = wasm_f32x4_add(sum_f32x4, data_f32x4);
        sumsq_f32x4 = wasm_f32x4_relaxed_madd(data_f32x4, data_f32x4, sumsq_f32x4);
    }
    nk_f32_t sum = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
    nk_f32_t sumsq = nk_reduce_add_f32x4_v128relaxed_(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_e5m2_to_f32_serial(&data_ptr[idx], &val);
        sum += val, sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_e5m2_v128relaxed(                      //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_e5m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_e5m2_v128relaxed_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e5m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL v128_t nk_fp8x16_to_comparable_v128relaxed_(v128_t raw_u8x16) {
    v128_t sign_mask_u8x16 = wasm_i8x16_splat((signed char)0x80);
    v128_t is_negative_b8x16 = wasm_i8x16_eq(wasm_v128_and(raw_u8x16, sign_mask_u8x16), sign_mask_u8x16);
    v128_t flip_positive_u8x16 = wasm_v128_xor(raw_u8x16, sign_mask_u8x16);
    v128_t flip_negative_u8x16 = wasm_v128_not(raw_u8x16);
    return wasm_i8x16_relaxed_laneselect(flip_negative_u8x16, flip_positive_u8x16, is_negative_b8x16);
}

NK_INTERNAL nk_u8_t nk_comparable_to_fp8_v128relaxed_(nk_u8_t comparable) {
    if (comparable >= 0x80) return comparable ^ 0x80;
    else return ~comparable;
}

NK_INTERNAL v128_t nk_fp6x16_to_comparable_v128relaxed_(v128_t raw_u8x16) {
    v128_t magnitude_u8x16 = wasm_v128_and(raw_u8x16, wasm_i8x16_splat(0x1F));
    v128_t sign_mask_u8x16 = wasm_i8x16_splat(0x20);
    v128_t is_negative_b8x16 = wasm_i8x16_eq(wasm_v128_and(raw_u8x16, sign_mask_u8x16), sign_mask_u8x16);
    v128_t positive_u8x16 = wasm_v128_or(magnitude_u8x16, sign_mask_u8x16);
    v128_t negative_u8x16 = wasm_i8x16_sub(wasm_i8x16_splat(0x1F), magnitude_u8x16);
    return wasm_i8x16_relaxed_laneselect(negative_u8x16, positive_u8x16, is_negative_b8x16);
}

NK_INTERNAL nk_u8_t nk_comparable_to_fp6_v128relaxed_(nk_u8_t comparable) {
    if (comparable >= 0x20) return comparable ^ 0x20;
    else return (0x1F - comparable) | 0x20;
}

NK_INTERNAL void nk_reduce_minmax_e4m3_v128relaxed_contiguous_( //
    nk_e4m3_t const *data_ptr, nk_size_t count,                 //
    nk_e4m3_t *min_value_ptr, nk_size_t *min_index_ptr,         //
    nk_e4m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    v128_t min_u8x16 = wasm_i8x16_splat((signed char)0xFF), max_u8x16 = wasm_i8x16_splat(0);
    v128_t min_iter_u8x16 = wasm_i8x16_splat(0), max_iter_u8x16 = wasm_i8x16_splat(0);
    v128_t iter_u8x16 = wasm_i8x16_splat(0), one_u8x16 = wasm_i8x16_splat(1);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        v128_t raw_u8x16 = wasm_v128_load(data_ptr + idx);
        v128_t comparable_u8x16 = nk_fp8x16_to_comparable_v128relaxed_(raw_u8x16);
        // E4M3 NaN: comparable == 0x00 (negative NaN) or comparable == 0xFF (positive NaN)
        v128_t is_nan_low_u8x16 = wasm_i8x16_eq(comparable_u8x16, wasm_i8x16_splat(0));
        v128_t is_nan_high_u8x16 = wasm_i8x16_eq(comparable_u8x16, wasm_i8x16_splat((signed char)0xFF));
        v128_t is_nan_u8x16 = wasm_v128_or(is_nan_low_u8x16, is_nan_high_u8x16);
        v128_t data_min_u8x16 = wasm_i8x16_relaxed_laneselect(wasm_i8x16_splat((signed char)0xFF), comparable_u8x16,
                                                              is_nan_u8x16);
        v128_t data_max_u8x16 = wasm_i8x16_relaxed_laneselect(wasm_i8x16_splat(0), comparable_u8x16, is_nan_u8x16);
        v128_t less_b8x16 = wasm_u8x16_lt(data_min_u8x16, min_u8x16);
        v128_t greater_b8x16 = wasm_u8x16_gt(data_max_u8x16, max_u8x16);
        min_u8x16 = wasm_i8x16_relaxed_laneselect(data_min_u8x16, min_u8x16, less_b8x16);
        max_u8x16 = wasm_i8x16_relaxed_laneselect(data_max_u8x16, max_u8x16, greater_b8x16);
        min_iter_u8x16 = wasm_i8x16_relaxed_laneselect(iter_u8x16, min_iter_u8x16, less_b8x16);
        max_iter_u8x16 = wasm_i8x16_relaxed_laneselect(iter_u8x16, max_iter_u8x16, greater_b8x16);
        iter_u8x16 = wasm_i8x16_add(iter_u8x16, one_u8x16);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_iters_vec, max_iters_vec;
    min_values_vec.v128 = min_u8x16;
    max_values_vec.v128 = max_u8x16;
    min_iters_vec.v128 = min_iter_u8x16;
    max_iters_vec.v128 = max_iter_u8x16;
    nk_u8_t min_comparable = min_values_vec.u8s[0];
    nk_size_t min_idx = (nk_size_t)min_iters_vec.u8s[0] * 16;
    for (int i = 1; i < 16; ++i) {
        nk_size_t abs_idx = (nk_size_t)min_iters_vec.u8s[i] * 16 + (nk_size_t)i;
        if (min_values_vec.u8s[i] < min_comparable || (min_values_vec.u8s[i] == min_comparable && abs_idx < min_idx))
            min_comparable = min_values_vec.u8s[i], min_idx = abs_idx;
    }
    nk_u8_t max_comparable = max_values_vec.u8s[0];
    nk_size_t max_idx = (nk_size_t)max_iters_vec.u8s[0] * 16;
    for (int i = 1; i < 16; ++i) {
        nk_size_t abs_idx = (nk_size_t)max_iters_vec.u8s[i] * 16 + (nk_size_t)i;
        if (max_values_vec.u8s[i] > max_comparable || (max_values_vec.u8s[i] == max_comparable && abs_idx < max_idx))
            max_comparable = max_values_vec.u8s[i], max_idx = abs_idx;
    }
    // Check if SIMD found only NaN (sentinels unchanged)
    if (min_comparable == 0xFF) min_idx = NK_SIZE_MAX;
    if (max_comparable == 0x00) max_idx = NK_SIZE_MAX;
    for (; idx < count; ++idx) {
        nk_u8_t raw = data_ptr[idx];
        nk_u8_t cmp = (raw & 0x80) ? (nk_u8_t)~raw : (raw ^ 0x80);
        if (cmp == 0x00 || cmp == 0xFF) continue;
        if (min_idx == NK_SIZE_MAX || cmp < min_comparable) min_comparable = cmp, min_idx = idx;
        if (max_idx == NK_SIZE_MAX || cmp > max_comparable) max_comparable = cmp, max_idx = idx;
    }
    if (min_idx == NK_SIZE_MAX) {
        *min_value_ptr = (nk_e4m3_t)NK_E4M3_MAX, *min_index_ptr = NK_SIZE_MAX;
        *max_value_ptr = (nk_e4m3_t)NK_E4M3_MIN, *max_index_ptr = NK_SIZE_MAX;
        return;
    }
    *min_value_ptr = nk_comparable_to_fp8_v128relaxed_(min_comparable), *min_index_ptr = min_idx;
    *max_value_ptr = nk_comparable_to_fp8_v128relaxed_(max_comparable), *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_e4m3_v128relaxed(                       //
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
        nk_e4m3_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_idx, right_min_idx, left_max_idx, right_max_idx;
        nk_reduce_minmax_e4m3_v128relaxed(data_ptr, left_count, stride_bytes, &left_min, &left_min_idx, &left_max,
                                          &left_max_idx);
        nk_reduce_minmax_e4m3_v128relaxed(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                          &right_min, &right_min_idx, &right_max, &right_max_idx);
        right_min_idx = (right_min_idx == NK_SIZE_MAX) ? NK_SIZE_MAX : left_count + right_min_idx;
        right_max_idx = (right_max_idx == NK_SIZE_MAX) ? NK_SIZE_MAX : left_count + right_max_idx;
        if (left_min_idx == NK_SIZE_MAX) *min_value_ptr = right_min, *min_index_ptr = right_min_idx;
        else if (right_min_idx == NK_SIZE_MAX || nk_e4m3_order_serial(left_min, right_min) <= 0)
            *min_value_ptr = left_min, *min_index_ptr = left_min_idx;
        else *min_value_ptr = right_min, *min_index_ptr = right_min_idx;
        if (left_max_idx == NK_SIZE_MAX) *max_value_ptr = right_max, *max_index_ptr = right_max_idx;
        else if (right_max_idx == NK_SIZE_MAX || nk_e4m3_order_serial(left_max, right_max) >= 0)
            *max_value_ptr = left_max, *max_index_ptr = left_max_idx;
        else *max_value_ptr = right_max, *max_index_ptr = right_max_idx;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e4m3_v128relaxed_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                      max_index_ptr);
    else
        nk_reduce_minmax_e4m3_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e5m2_v128relaxed_contiguous_( //
    nk_e5m2_t const *data_ptr, nk_size_t count,                 //
    nk_e5m2_t *min_value_ptr, nk_size_t *min_index_ptr,         //
    nk_e5m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    v128_t min_u8x16 = wasm_i8x16_splat((signed char)0xFF), max_u8x16 = wasm_i8x16_splat(0);
    v128_t min_iter_u8x16 = wasm_i8x16_splat(0), max_iter_u8x16 = wasm_i8x16_splat(0);
    v128_t iter_u8x16 = wasm_i8x16_splat(0), one_u8x16 = wasm_i8x16_splat(1);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        v128_t raw_u8x16 = wasm_v128_load(data_ptr + idx);
        v128_t comparable_u8x16 = nk_fp8x16_to_comparable_v128relaxed_(raw_u8x16);
        // E5M2 NaN: comparable <= 0x02 or comparable >= 0xFD
        v128_t low_bound_u8x16 = wasm_i8x16_splat(0x02);
        v128_t high_bound_u8x16 = wasm_i8x16_splat((signed char)0xFD);
        v128_t is_nan_low_u8x16 = wasm_u8x16_le(comparable_u8x16, low_bound_u8x16);
        v128_t is_nan_high_u8x16 = wasm_u8x16_ge(comparable_u8x16, high_bound_u8x16);
        v128_t is_nan_u8x16 = wasm_v128_or(is_nan_low_u8x16, is_nan_high_u8x16);
        v128_t data_min_u8x16 = wasm_i8x16_relaxed_laneselect(wasm_i8x16_splat((signed char)0xFF), comparable_u8x16,
                                                              is_nan_u8x16);
        v128_t data_max_u8x16 = wasm_i8x16_relaxed_laneselect(wasm_i8x16_splat(0), comparable_u8x16, is_nan_u8x16);
        v128_t less_b8x16 = wasm_u8x16_lt(data_min_u8x16, min_u8x16);
        v128_t greater_b8x16 = wasm_u8x16_gt(data_max_u8x16, max_u8x16);
        min_u8x16 = wasm_i8x16_relaxed_laneselect(data_min_u8x16, min_u8x16, less_b8x16);
        max_u8x16 = wasm_i8x16_relaxed_laneselect(data_max_u8x16, max_u8x16, greater_b8x16);
        min_iter_u8x16 = wasm_i8x16_relaxed_laneselect(iter_u8x16, min_iter_u8x16, less_b8x16);
        max_iter_u8x16 = wasm_i8x16_relaxed_laneselect(iter_u8x16, max_iter_u8x16, greater_b8x16);
        iter_u8x16 = wasm_i8x16_add(iter_u8x16, one_u8x16);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_iters_vec, max_iters_vec;
    min_values_vec.v128 = min_u8x16;
    max_values_vec.v128 = max_u8x16;
    min_iters_vec.v128 = min_iter_u8x16;
    max_iters_vec.v128 = max_iter_u8x16;
    nk_u8_t min_comparable = min_values_vec.u8s[0];
    nk_size_t min_idx = (nk_size_t)min_iters_vec.u8s[0] * 16;
    for (int i = 1; i < 16; ++i) {
        nk_size_t abs_idx = (nk_size_t)min_iters_vec.u8s[i] * 16 + (nk_size_t)i;
        if (min_values_vec.u8s[i] < min_comparable || (min_values_vec.u8s[i] == min_comparable && abs_idx < min_idx))
            min_comparable = min_values_vec.u8s[i], min_idx = abs_idx;
    }
    nk_u8_t max_comparable = max_values_vec.u8s[0];
    nk_size_t max_idx = (nk_size_t)max_iters_vec.u8s[0] * 16;
    for (int i = 1; i < 16; ++i) {
        nk_size_t abs_idx = (nk_size_t)max_iters_vec.u8s[i] * 16 + (nk_size_t)i;
        if (max_values_vec.u8s[i] > max_comparable || (max_values_vec.u8s[i] == max_comparable && abs_idx < max_idx))
            max_comparable = max_values_vec.u8s[i], max_idx = abs_idx;
    }
    // Check if SIMD found only NaN (sentinels unchanged)
    if (min_comparable == 0xFF) min_idx = NK_SIZE_MAX;
    if (max_comparable == 0x00) max_idx = NK_SIZE_MAX;
    for (; idx < count; ++idx) {
        nk_u8_t raw = data_ptr[idx];
        nk_u8_t cmp = (raw & 0x80) ? (nk_u8_t)~raw : (raw ^ 0x80);
        if (cmp <= 0x02 || cmp >= 0xFD) continue;
        if (min_idx == NK_SIZE_MAX || cmp < min_comparable) min_comparable = cmp, min_idx = idx;
        if (max_idx == NK_SIZE_MAX || cmp > max_comparable) max_comparable = cmp, max_idx = idx;
    }
    if (min_idx == NK_SIZE_MAX) {
        *min_value_ptr = (nk_e5m2_t)NK_E5M2_MAX, *min_index_ptr = NK_SIZE_MAX;
        *max_value_ptr = (nk_e5m2_t)NK_E5M2_MIN, *max_index_ptr = NK_SIZE_MAX;
        return;
    }
    *min_value_ptr = nk_comparable_to_fp8_v128relaxed_(min_comparable), *min_index_ptr = min_idx;
    *max_value_ptr = nk_comparable_to_fp8_v128relaxed_(max_comparable), *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_e5m2_v128relaxed(                       //
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
        nk_e5m2_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_idx, right_min_idx, left_max_idx, right_max_idx;
        nk_reduce_minmax_e5m2_v128relaxed(data_ptr, left_count, stride_bytes, &left_min, &left_min_idx, &left_max,
                                          &left_max_idx);
        nk_reduce_minmax_e5m2_v128relaxed(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                          &right_min, &right_min_idx, &right_max, &right_max_idx);
        right_min_idx = (right_min_idx == NK_SIZE_MAX) ? NK_SIZE_MAX : left_count + right_min_idx;
        right_max_idx = (right_max_idx == NK_SIZE_MAX) ? NK_SIZE_MAX : left_count + right_max_idx;
        if (left_min_idx == NK_SIZE_MAX) *min_value_ptr = right_min, *min_index_ptr = right_min_idx;
        else if (right_min_idx == NK_SIZE_MAX || nk_e5m2_order_serial(left_min, right_min) <= 0)
            *min_value_ptr = left_min, *min_index_ptr = left_min_idx;
        else *min_value_ptr = right_min, *min_index_ptr = right_min_idx;
        if (left_max_idx == NK_SIZE_MAX) *max_value_ptr = right_max, *max_index_ptr = right_max_idx;
        else if (right_max_idx == NK_SIZE_MAX || nk_e5m2_order_serial(left_max, right_max) >= 0)
            *max_value_ptr = left_max, *max_index_ptr = left_max_idx;
        else *max_value_ptr = right_max, *max_index_ptr = right_max_idx;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e5m2_v128relaxed_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                      max_index_ptr);
    else
        nk_reduce_minmax_e5m2_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e2m3_v128relaxed_contiguous_( //
    nk_e2m3_t const *data_ptr, nk_size_t count,                 //
    nk_e2m3_t *min_value_ptr, nk_size_t *min_index_ptr,         //
    nk_e2m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    v128_t min_u8x16 = wasm_i8x16_splat(0x3F), max_u8x16 = wasm_i8x16_splat(0);
    v128_t min_iter_u8x16 = wasm_i8x16_splat(0), max_iter_u8x16 = wasm_i8x16_splat(0);
    v128_t iter_u8x16 = wasm_i8x16_splat(0), one_u8x16 = wasm_i8x16_splat(1);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        v128_t raw_u8x16 = wasm_v128_load(data_ptr + idx);
        v128_t comparable_u8x16 = nk_fp6x16_to_comparable_v128relaxed_(raw_u8x16);
        v128_t less_b8x16 = wasm_u8x16_lt(comparable_u8x16, min_u8x16);
        v128_t greater_b8x16 = wasm_u8x16_gt(comparable_u8x16, max_u8x16);
        min_u8x16 = wasm_i8x16_relaxed_laneselect(comparable_u8x16, min_u8x16, less_b8x16);
        max_u8x16 = wasm_i8x16_relaxed_laneselect(comparable_u8x16, max_u8x16, greater_b8x16);
        min_iter_u8x16 = wasm_i8x16_relaxed_laneselect(iter_u8x16, min_iter_u8x16, less_b8x16);
        max_iter_u8x16 = wasm_i8x16_relaxed_laneselect(iter_u8x16, max_iter_u8x16, greater_b8x16);
        iter_u8x16 = wasm_i8x16_add(iter_u8x16, one_u8x16);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_iters_vec, max_iters_vec;
    min_values_vec.v128 = min_u8x16;
    max_values_vec.v128 = max_u8x16;
    min_iters_vec.v128 = min_iter_u8x16;
    max_iters_vec.v128 = max_iter_u8x16;
    nk_u8_t min_comparable = min_values_vec.u8s[0];
    nk_size_t min_idx = (nk_size_t)min_iters_vec.u8s[0] * 16;
    for (int i = 1; i < 16; ++i) {
        nk_size_t abs_idx = (nk_size_t)min_iters_vec.u8s[i] * 16 + (nk_size_t)i;
        if (min_values_vec.u8s[i] < min_comparable || (min_values_vec.u8s[i] == min_comparable && abs_idx < min_idx))
            min_comparable = min_values_vec.u8s[i], min_idx = abs_idx;
    }
    nk_u8_t max_comparable = max_values_vec.u8s[0];
    nk_size_t max_idx = (nk_size_t)max_iters_vec.u8s[0] * 16;
    for (int i = 1; i < 16; ++i) {
        nk_size_t abs_idx = (nk_size_t)max_iters_vec.u8s[i] * 16 + (nk_size_t)i;
        if (max_values_vec.u8s[i] > max_comparable || (max_values_vec.u8s[i] == max_comparable && abs_idx < max_idx))
            max_comparable = max_values_vec.u8s[i], max_idx = abs_idx;
    }
    for (; idx < count; ++idx) {
        nk_u8_t raw = data_ptr[idx] & 0x3F;
        nk_u8_t sign = raw >> 5;
        nk_u8_t mag = raw & 0x1F;
        nk_u8_t cmp = sign ? (0x1F - mag) : (mag | 0x20);
        if (cmp < min_comparable) min_comparable = cmp, min_idx = idx;
        if (cmp > max_comparable) max_comparable = cmp, max_idx = idx;
    }
    *min_value_ptr = nk_comparable_to_fp6_v128relaxed_(min_comparable), *min_index_ptr = min_idx;
    *max_value_ptr = nk_comparable_to_fp6_v128relaxed_(max_comparable), *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_e2m3_v128relaxed(                       //
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
        nk_e2m3_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_idx, right_min_idx, left_max_idx, right_max_idx;
        nk_reduce_minmax_e2m3_v128relaxed(data_ptr, left_count, stride_bytes, &left_min, &left_min_idx, &left_max,
                                          &left_max_idx);
        nk_reduce_minmax_e2m3_v128relaxed(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                          &right_min, &right_min_idx, &right_max, &right_max_idx);
        if (nk_e2m3_order_serial(right_min, left_min) < 0)
            *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_idx;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_idx;
        if (nk_e2m3_order_serial(right_max, left_max) > 0)
            *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_idx;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_idx;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e2m3_v128relaxed_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                      max_index_ptr);
    else
        nk_reduce_minmax_e2m3_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e3m2_v128relaxed_contiguous_( //
    nk_e3m2_t const *data_ptr, nk_size_t count,                 //
    nk_e3m2_t *min_value_ptr, nk_size_t *min_index_ptr,         //
    nk_e3m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    v128_t min_u8x16 = wasm_i8x16_splat(0x3F), max_u8x16 = wasm_i8x16_splat(0);
    v128_t min_iter_u8x16 = wasm_i8x16_splat(0), max_iter_u8x16 = wasm_i8x16_splat(0);
    v128_t iter_u8x16 = wasm_i8x16_splat(0), one_u8x16 = wasm_i8x16_splat(1);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        v128_t raw_u8x16 = wasm_v128_load(data_ptr + idx);
        v128_t comparable_u8x16 = nk_fp6x16_to_comparable_v128relaxed_(raw_u8x16);
        v128_t less_b8x16 = wasm_u8x16_lt(comparable_u8x16, min_u8x16);
        v128_t greater_b8x16 = wasm_u8x16_gt(comparable_u8x16, max_u8x16);
        min_u8x16 = wasm_i8x16_relaxed_laneselect(comparable_u8x16, min_u8x16, less_b8x16);
        max_u8x16 = wasm_i8x16_relaxed_laneselect(comparable_u8x16, max_u8x16, greater_b8x16);
        min_iter_u8x16 = wasm_i8x16_relaxed_laneselect(iter_u8x16, min_iter_u8x16, less_b8x16);
        max_iter_u8x16 = wasm_i8x16_relaxed_laneselect(iter_u8x16, max_iter_u8x16, greater_b8x16);
        iter_u8x16 = wasm_i8x16_add(iter_u8x16, one_u8x16);
    }
    nk_b128_vec_t min_values_vec, max_values_vec, min_iters_vec, max_iters_vec;
    min_values_vec.v128 = min_u8x16;
    max_values_vec.v128 = max_u8x16;
    min_iters_vec.v128 = min_iter_u8x16;
    max_iters_vec.v128 = max_iter_u8x16;
    nk_u8_t min_comparable = min_values_vec.u8s[0];
    nk_size_t min_idx = (nk_size_t)min_iters_vec.u8s[0] * 16;
    for (int i = 1; i < 16; ++i) {
        nk_size_t abs_idx = (nk_size_t)min_iters_vec.u8s[i] * 16 + (nk_size_t)i;
        if (min_values_vec.u8s[i] < min_comparable || (min_values_vec.u8s[i] == min_comparable && abs_idx < min_idx))
            min_comparable = min_values_vec.u8s[i], min_idx = abs_idx;
    }
    nk_u8_t max_comparable = max_values_vec.u8s[0];
    nk_size_t max_idx = (nk_size_t)max_iters_vec.u8s[0] * 16;
    for (int i = 1; i < 16; ++i) {
        nk_size_t abs_idx = (nk_size_t)max_iters_vec.u8s[i] * 16 + (nk_size_t)i;
        if (max_values_vec.u8s[i] > max_comparable || (max_values_vec.u8s[i] == max_comparable && abs_idx < max_idx))
            max_comparable = max_values_vec.u8s[i], max_idx = abs_idx;
    }
    for (; idx < count; ++idx) {
        nk_u8_t raw = data_ptr[idx] & 0x3F;
        nk_u8_t sign = raw >> 5;
        nk_u8_t mag = raw & 0x1F;
        nk_u8_t cmp = sign ? (0x1F - mag) : (mag | 0x20);
        if (cmp < min_comparable) min_comparable = cmp, min_idx = idx;
        if (cmp > max_comparable) max_comparable = cmp, max_idx = idx;
    }
    *min_value_ptr = nk_comparable_to_fp6_v128relaxed_(min_comparable), *min_index_ptr = min_idx;
    *max_value_ptr = nk_comparable_to_fp6_v128relaxed_(max_comparable), *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_e3m2_v128relaxed(                       //
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
        nk_e3m2_t left_min, right_min, left_max, right_max;
        nk_size_t left_min_idx, right_min_idx, left_max_idx, right_max_idx;
        nk_reduce_minmax_e3m2_v128relaxed(data_ptr, left_count, stride_bytes, &left_min, &left_min_idx, &left_max,
                                          &left_max_idx);
        nk_reduce_minmax_e3m2_v128relaxed(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                          &right_min, &right_min_idx, &right_max, &right_max_idx);
        if (nk_e3m2_order_serial(right_min, left_min) < 0)
            *min_value_ptr = right_min, *min_index_ptr = left_count + right_min_idx;
        else *min_value_ptr = left_min, *min_index_ptr = left_min_idx;
        if (nk_e3m2_order_serial(right_max, left_max) > 0)
            *max_value_ptr = right_max, *max_index_ptr = left_count + right_max_idx;
        else *max_value_ptr = left_max, *max_index_ptr = left_max_idx;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e3m2_v128relaxed_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                      max_index_ptr);
    else
        nk_reduce_minmax_e3m2_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_REDUCE_V128RELAXED_H
