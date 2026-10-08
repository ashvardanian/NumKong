/**
 *  @file include/numkong/spatial/v128.h
 *  @author Ash Vardanian
 *  @date September 12, 2026
 *  @brief SIMD-accelerated spatial similarity measures for WASM.
 *
 *  Contains:
 *  - Euclidean (L2) distance
 *  - Squared Euclidean (L2SQ) distance
 *  - Angular distance (1 - cosine similarity)
 *
 *  For dtypes:
 *  - 16-bit brain floating point (bf16)
 *  - 8-bit signed and unsigned integers (i8, u8)
 *
 *  Key improvements:
 *  - Parallel SIMD sqrt for normalization (computes both sqrts simultaneously)
 *  - Edge case handling (zero vectors, numerical stability)
 *  - Integer products widen to i16 and multiply-add adjacent pairs with `i32x4.dot_i16x8_s`
 */

#ifndef NUMKONG_SPATIAL_V128_H
#define NUMKONG_SPATIAL_V128_H

#if NUMKONG_ARCH_WASM_
#if NUMKONG_ARCH_WASM_V128_

#include "numkong/types.h"
#include "numkong/reduce/v128.h" // `nk_reduce_add_f32x4_v128_`, `nk_reduce_add_i32x4_to_i64_v128_`
#include "numkong/cast/serial.h"
#include "numkong/cast/v128.h" // `nk_load_b128_v128_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

NUMKONG_INLINE nk_f64_t nk_angular_normalize_f64_v128_(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {
    // Edge case: both vectors have zero magnitude
    if (a2 == 0.0 && b2 == 0.0) return 0.0;
    // Edge case: dot product is zero (perpendicular or one vector is zero)
    if (ab == 0.0) return 1.0;

    v128_t squares_f64x2 = wasm_f64x2_make(a2, b2);
    v128_t sqrts_f64x2 = wasm_f64x2_sqrt(squares_f64x2);
    nk_f64_t a_sqrt = wasm_f64x2_extract_lane(sqrts_f64x2, 0);
    nk_f64_t b_sqrt = wasm_f64x2_extract_lane(sqrts_f64x2, 1);
    nk_f64_t result = 1.0 - ab / (a_sqrt * b_sqrt);

    // Clamp negative results to 0 (can occur due to floating-point rounding)
    return result > 0.0 ? result : 0.0;
}

#pragma region BF16 Floats

/** Squared Euclidean distance between @p n BF16 values of @p a and @p b, accumulated in F32. */
NUMKONG_INLINE void nk_squared_distance_bf16_v128_(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                   nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t mask_high_u32x4 = wasm_i32x4_splat((int)0xFFFF0000);
    nk_bf16_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b128_vec_t a_bf16_vec, b_bf16_vec;

nk_sqeuclidean_bf16_v128_cycle:
    if (count_scalars < 8) {
        nk_partial_load_b16x8_serial_(a_scalars, &a_bf16_vec, count_scalars);
        nk_partial_load_b16x8_serial_(b_scalars, &b_bf16_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b128_v128_(a_scalars, &a_bf16_vec);
        nk_load_b128_v128_(b_scalars, &b_bf16_vec);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    v128_t a_even_f32x4 = wasm_i32x4_shl(a_bf16_vec.v128, 16);
    v128_t b_even_f32x4 = wasm_i32x4_shl(b_bf16_vec.v128, 16);
    v128_t diff_even_f32x4 = wasm_f32x4_sub(a_even_f32x4, b_even_f32x4);
    sum_f32x4 = wasm_f32x4_add(sum_f32x4, wasm_f32x4_mul(diff_even_f32x4, diff_even_f32x4));
    v128_t a_odd_f32x4 = wasm_v128_and(a_bf16_vec.v128, mask_high_u32x4);
    v128_t b_odd_f32x4 = wasm_v128_and(b_bf16_vec.v128, mask_high_u32x4);
    v128_t diff_odd_f32x4 = wasm_f32x4_sub(a_odd_f32x4, b_odd_f32x4);
    sum_f32x4 = wasm_f32x4_add(sum_f32x4, wasm_f32x4_mul(diff_odd_f32x4, diff_odd_f32x4));
    if (count_scalars) goto nk_sqeuclidean_bf16_v128_cycle;

    *result = nk_reduce_add_f32x4_v128_(sum_f32x4);
}

#if NUMKONG_TARGET_V128
NUMKONG_API nk_status_t nk_sqeuclidean_bf16_v128(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_bf16_v128_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_bf16_v128(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t l2sq;
    nk_squared_distance_bf16_v128_(a, b, n, &l2sq);
    *result = wasm_f32x4_extract_lane(wasm_f32x4_sqrt(wasm_f32x4_splat(l2sq)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_bf16_v128(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    v128_t ab_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t a2_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t b2_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t mask_high_u32x4 = wasm_i32x4_splat((int)0xFFFF0000);
    nk_bf16_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b128_vec_t a_bf16_vec, b_bf16_vec;

nk_angular_bf16_v128_cycle:
    if (count_scalars < 8) {
        nk_partial_load_b16x8_serial_(a_scalars, &a_bf16_vec, count_scalars);
        nk_partial_load_b16x8_serial_(b_scalars, &b_bf16_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b128_v128_(a_scalars, &a_bf16_vec);
        nk_load_b128_v128_(b_scalars, &b_bf16_vec);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    v128_t a_even_f32x4 = wasm_i32x4_shl(a_bf16_vec.v128, 16);
    v128_t b_even_f32x4 = wasm_i32x4_shl(b_bf16_vec.v128, 16);
    ab_f32x4 = wasm_f32x4_add(ab_f32x4, wasm_f32x4_mul(a_even_f32x4, b_even_f32x4));
    a2_f32x4 = wasm_f32x4_add(a2_f32x4, wasm_f32x4_mul(a_even_f32x4, a_even_f32x4));
    b2_f32x4 = wasm_f32x4_add(b2_f32x4, wasm_f32x4_mul(b_even_f32x4, b_even_f32x4));
    v128_t a_odd_f32x4 = wasm_v128_and(a_bf16_vec.v128, mask_high_u32x4);
    v128_t b_odd_f32x4 = wasm_v128_and(b_bf16_vec.v128, mask_high_u32x4);
    ab_f32x4 = wasm_f32x4_add(ab_f32x4, wasm_f32x4_mul(a_odd_f32x4, b_odd_f32x4));
    a2_f32x4 = wasm_f32x4_add(a2_f32x4, wasm_f32x4_mul(a_odd_f32x4, a_odd_f32x4));
    b2_f32x4 = wasm_f32x4_add(b2_f32x4, wasm_f32x4_mul(b_odd_f32x4, b_odd_f32x4));
    if (count_scalars) goto nk_angular_bf16_v128_cycle;

    nk_f32_t ab = nk_reduce_add_f32x4_v128_(ab_f32x4);
    nk_f32_t a2 = nk_reduce_add_f32x4_v128_(a2_f32x4);
    nk_f32_t b2 = nk_reduce_add_f32x4_v128_(b2_f32x4);
    *result = (nk_f32_t)nk_angular_normalize_f64_v128_((nk_f64_t)ab, (nk_f64_t)a2, (nk_f64_t)b2);
    return nk_success_k;
}

#pragma endregion BF16 Floats
#pragma region I8 and U8 Integers

/** Squared Euclidean distance between @p n U8 values of @p a and @p b, exact in U32. */
NUMKONG_INLINE void nk_squared_distance_u8_v128_(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    v128_t sum_u32x4 = wasm_u32x4_splat(0);
    nk_u8_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    v128_t a_u8x16, b_u8x16;

nk_sqeuclidean_u8_v128_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec = {0}, b_vec = {0};
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        a_u8x16 = a_vec.v128;
        b_u8x16 = b_vec.v128;
        count_scalars = 0;
    }
    else {
        a_u8x16 = wasm_v128_load(a_scalars);
        b_u8x16 = wasm_v128_load(b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }

    v128_t a_minus_b_u8x16 = wasm_u8x16_sub_sat(a_u8x16, b_u8x16);
    v128_t b_minus_a_u8x16 = wasm_u8x16_sub_sat(b_u8x16, a_u8x16);
    v128_t diff_u8x16 = wasm_v128_or(a_minus_b_u8x16, b_minus_a_u8x16); // |a-b|, one side saturates to zero
    v128_t diff_low_u16x8 = wasm_u16x8_extend_low_u8x16(diff_u8x16);
    v128_t diff_high_u16x8 = wasm_u16x8_extend_high_u8x16(diff_u8x16);
    sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_extmul_low_i16x8(diff_low_u16x8, diff_low_u16x8));
    sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_extmul_high_i16x8(diff_low_u16x8, diff_low_u16x8));
    sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_extmul_low_i16x8(diff_high_u16x8, diff_high_u16x8));
    sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_extmul_high_i16x8(diff_high_u16x8, diff_high_u16x8));
    if (count_scalars) goto nk_sqeuclidean_u8_v128_cycle;

    *result = nk_reduce_add_u32x4_v128_(sum_u32x4);
}

/** Squared Euclidean distance between @p n I8 values of @p a and @p b, exact in U32. */
NUMKONG_INLINE void nk_squared_distance_i8_v128_(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result) {
    v128_t sum_u32x4 = wasm_u32x4_splat(0);
    v128_t bias_u8x16 = wasm_u8x16_splat(0x80); // XOR flips i8 to u8, and |a-b|² is invariant under the shared offset
    nk_i8_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    v128_t a_u8x16, b_u8x16;

nk_sqeuclidean_i8_v128_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec = {0}, b_vec = {0};
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        a_u8x16 = wasm_v128_xor(a_vec.v128, bias_u8x16);
        b_u8x16 = wasm_v128_xor(b_vec.v128, bias_u8x16);
        count_scalars = 0;
    }
    else {
        a_u8x16 = wasm_v128_xor(wasm_v128_load(a_scalars), bias_u8x16);
        b_u8x16 = wasm_v128_xor(wasm_v128_load(b_scalars), bias_u8x16);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }

    v128_t diff_u8x16 = wasm_v128_or(wasm_u8x16_sub_sat(a_u8x16, b_u8x16), wasm_u8x16_sub_sat(b_u8x16, a_u8x16));
    v128_t diff_low_u16x8 = wasm_u16x8_extend_low_u8x16(diff_u8x16);
    v128_t diff_high_u16x8 = wasm_u16x8_extend_high_u8x16(diff_u8x16);
    sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_extmul_low_i16x8(diff_low_u16x8, diff_low_u16x8));
    sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_extmul_high_i16x8(diff_low_u16x8, diff_low_u16x8));
    sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_extmul_low_i16x8(diff_high_u16x8, diff_high_u16x8));
    sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_extmul_high_i16x8(diff_high_u16x8, diff_high_u16x8));
    if (count_scalars) goto nk_sqeuclidean_i8_v128_cycle;

    *result = nk_reduce_add_u32x4_v128_(sum_u32x4);
}

NUMKONG_API nk_status_t nk_sqeuclidean_u8_v128(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_u8_v128_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_u8_v128(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u32_t distance_sq;
    nk_squared_distance_u8_v128_(a, b, n, &distance_sq);
    *result = wasm_f32x4_extract_lane(wasm_f32x4_sqrt(wasm_f32x4_splat((nk_f32_t)distance_sq)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_u8_v128(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u64_t dot_ab_total = 0, dot_aa_total = 0, dot_bb_total = 0;
    nk_size_t i = 0;

    // Windowed accumulation loop
    while (i + 16 <= n) {
        v128_t dot_ab_u32x4 = wasm_i32x4_splat(0);
        v128_t dot_aa_u32x4 = wasm_i32x4_splat(0);
        v128_t dot_bb_u32x4 = wasm_i32x4_splat(0);

        nk_size_t cycle = 0;
        // u8 widened to u16 fits the signed pairwise dot, which adds at most 4 × 255² per lane, so 8191
        // iterations stay below 2³¹
        for (; cycle < 8191 && i + 16 <= n; ++cycle, i += 16) {
            v128_t a_u8x16 = wasm_v128_load(a + i);
            v128_t b_u8x16 = wasm_v128_load(b + i);
            v128_t a_low_u16x8 = wasm_u16x8_extend_low_u8x16(a_u8x16);
            v128_t a_high_u16x8 = wasm_u16x8_extend_high_u8x16(a_u8x16);
            v128_t b_low_u16x8 = wasm_u16x8_extend_low_u8x16(b_u8x16);
            v128_t b_high_u16x8 = wasm_u16x8_extend_high_u8x16(b_u8x16);
            dot_ab_u32x4 = wasm_i32x4_add(dot_ab_u32x4, wasm_i32x4_dot_i16x8(a_low_u16x8, b_low_u16x8));
            dot_ab_u32x4 = wasm_i32x4_add(dot_ab_u32x4, wasm_i32x4_dot_i16x8(a_high_u16x8, b_high_u16x8));
            dot_aa_u32x4 = wasm_i32x4_add(dot_aa_u32x4, wasm_i32x4_dot_i16x8(a_low_u16x8, a_low_u16x8));
            dot_aa_u32x4 = wasm_i32x4_add(dot_aa_u32x4, wasm_i32x4_dot_i16x8(a_high_u16x8, a_high_u16x8));
            dot_bb_u32x4 = wasm_i32x4_add(dot_bb_u32x4, wasm_i32x4_dot_i16x8(b_low_u16x8, b_low_u16x8));
            dot_bb_u32x4 = wasm_i32x4_add(dot_bb_u32x4, wasm_i32x4_dot_i16x8(b_high_u16x8, b_high_u16x8));
        }

        // Reduce window to scalars
        dot_ab_total += nk_reduce_add_u32x4_to_u64_v128_(dot_ab_u32x4);
        dot_aa_total += nk_reduce_add_u32x4_to_u64_v128_(dot_aa_u32x4);
        dot_bb_total += nk_reduce_add_u32x4_to_u64_v128_(dot_bb_u32x4);
    }

    // Scalar tail
    for (; i < n; i++) {
        dot_ab_total += (nk_u32_t)a[i] * (nk_u32_t)b[i];
        dot_aa_total += (nk_u32_t)a[i] * (nk_u32_t)a[i];
        dot_bb_total += (nk_u32_t)b[i] * (nk_u32_t)b[i];
    }

    *result = (nk_f32_t)nk_angular_normalize_f64_v128_((nk_f64_t)dot_ab_total, (nk_f64_t)dot_aa_total,
                                                       (nk_f64_t)dot_bb_total);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_i8_v128(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_i8_v128_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_i8_v128(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u32_t distance_sq;
    nk_squared_distance_i8_v128_(a, b, n, &distance_sq);
    *result = wasm_f32x4_extract_lane(wasm_f32x4_sqrt(wasm_f32x4_splat((nk_f32_t)distance_sq)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_i8_v128(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_i64_t dot_ab_total = 0, dot_aa_total = 0, dot_bb_total = 0;
    nk_size_t i = 0;

    // Windowed accumulation loop
    while (i + 16 <= n) {
        v128_t dot_ab_i32x4 = wasm_i32x4_splat(0);
        v128_t dot_aa_i32x4 = wasm_i32x4_splat(0);
        v128_t dot_bb_i32x4 = wasm_i32x4_splat(0);

        nk_size_t cycle = 0;
        // Two pairwise dots add at most 65536 per lane, so 32767 iterations fit i32
        for (; cycle < 32767 && i + 16 <= n; ++cycle, i += 16) {
            v128_t a_i8x16 = wasm_v128_load(a + i);
            v128_t b_i8x16 = wasm_v128_load(b + i);
            v128_t a_low_i16x8 = wasm_i16x8_extend_low_i8x16(a_i8x16);
            v128_t a_high_i16x8 = wasm_i16x8_extend_high_i8x16(a_i8x16);
            v128_t b_low_i16x8 = wasm_i16x8_extend_low_i8x16(b_i8x16);
            v128_t b_high_i16x8 = wasm_i16x8_extend_high_i8x16(b_i8x16);
            dot_ab_i32x4 = wasm_i32x4_add(dot_ab_i32x4, wasm_i32x4_dot_i16x8(a_low_i16x8, b_low_i16x8));
            dot_ab_i32x4 = wasm_i32x4_add(dot_ab_i32x4, wasm_i32x4_dot_i16x8(a_high_i16x8, b_high_i16x8));
            dot_aa_i32x4 = wasm_i32x4_add(dot_aa_i32x4, wasm_i32x4_dot_i16x8(a_low_i16x8, a_low_i16x8));
            dot_aa_i32x4 = wasm_i32x4_add(dot_aa_i32x4, wasm_i32x4_dot_i16x8(a_high_i16x8, a_high_i16x8));
            dot_bb_i32x4 = wasm_i32x4_add(dot_bb_i32x4, wasm_i32x4_dot_i16x8(b_low_i16x8, b_low_i16x8));
            dot_bb_i32x4 = wasm_i32x4_add(dot_bb_i32x4, wasm_i32x4_dot_i16x8(b_high_i16x8, b_high_i16x8));
        }

        // Reduce window to scalars
        dot_ab_total += nk_reduce_add_i32x4_to_i64_v128_(dot_ab_i32x4);
        dot_aa_total += nk_reduce_add_i32x4_to_i64_v128_(dot_aa_i32x4);
        dot_bb_total += nk_reduce_add_i32x4_to_i64_v128_(dot_bb_i32x4);
    }

    // Scalar tail
    for (; i < n; i++) {
        dot_ab_total += (nk_i32_t)a[i] * (nk_i32_t)b[i];
        dot_aa_total += (nk_i32_t)a[i] * (nk_i32_t)a[i];
        dot_bb_total += (nk_i32_t)b[i] * (nk_i32_t)b[i];
    }

    *result = (nk_f32_t)nk_angular_normalize_f64_v128_((nk_f64_t)dot_ab_total, (nk_f64_t)dot_aa_total,
                                                       (nk_f64_t)dot_bb_total);
    return nk_success_k;
}
#endif // NUMKONG_TARGET_V128

#pragma endregion I8 and U8 Integers
#pragma region Spatial From Dot Helpers

/** Angular from_dot: computes 1 − dot / (√q × √t) for 4 pairs in f32, where q is @p query_sumsq and
 *  t each target's sum of squares, with the rules of the serial variant. Separate square roots
 *  avoid overflowing the product of two finite-but-large norms. */
NUMKONG_INLINE void nk_angular_through_f32_from_dot_v128_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
                                                          nk_b128_vec_t const *target_sumsqs_vec,
                                                          nk_b128_vec_t *result_vec) {
    v128_t const zeros_f32x4 = wasm_f32x4_splat(0.0f), ones_f32x4 = wasm_f32x4_splat(1.0f), dots_f32x4 = dots_vec->v128;
    v128_t const query_sumsq_f32x4 = wasm_f32x4_splat(query_sumsq), target_sumsqs_f32x4 = target_sumsqs_vec->v128;
    v128_t const norm_f32x4 = wasm_f32x4_mul(wasm_f32x4_sqrt(query_sumsq_f32x4), wasm_f32x4_sqrt(target_sumsqs_f32x4));
    v128_t angular_f32x4 = wasm_f32x4_max(wasm_f32x4_sub(ones_f32x4, wasm_f32x4_div(dots_f32x4, norm_f32x4)),
                                          zeros_f32x4);
    v128_t const unit_f32x4 = wasm_v128_or(
        wasm_f32x4_eq(dots_f32x4, zeros_f32x4),
        wasm_v128_or(wasm_f32x4_eq(query_sumsq_f32x4, zeros_f32x4), wasm_f32x4_eq(target_sumsqs_f32x4, zeros_f32x4)));
    angular_f32x4 = wasm_v128_bitselect(ones_f32x4, angular_f32x4, unit_f32x4);
    angular_f32x4 = wasm_v128_andnot(
        angular_f32x4, wasm_f32x4_eq(wasm_f32x4_add(query_sumsq_f32x4, target_sumsqs_f32x4), zeros_f32x4));
    // A NaN dot outranks the zero-norm cases
    result_vec->v128 = wasm_v128_bitselect(dots_f32x4, angular_f32x4, wasm_f32x4_ne(dots_f32x4, dots_f32x4));
}

/** Euclidean from_dot: computes √(q + t − 2 × dot) for 4 pairs in f32, where q is @p query_sumsq
 *  and t each target's sum of squares. */
NUMKONG_INLINE void nk_euclidean_through_f32_from_dot_v128_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
                                                            nk_b128_vec_t const *target_sumsqs_vec,
                                                            nk_b128_vec_t *result_vec) {
    v128_t dots_f32x4 = dots_vec->v128;
    v128_t query_sumsq_f32x4 = wasm_f32x4_splat(query_sumsq);
    v128_t two_f32x4 = wasm_f32x4_splat(2.0f);
    v128_t sum_sq_f32x4 = wasm_f32x4_add(query_sumsq_f32x4, target_sumsqs_vec->v128);
    v128_t dist_sq_f32x4 = wasm_f32x4_sub(sum_sq_f32x4, wasm_f32x4_mul(two_f32x4, dots_f32x4));
    dist_sq_f32x4 = wasm_f32x4_max(dist_sq_f32x4, wasm_f32x4_splat(0.0f));
    result_vec->v128 = wasm_f32x4_sqrt(dist_sq_f32x4);
}

/** Angular from_dot for 4 i32 dots and u32 norms. The gap ab − d² is exact in 64-bit @c extmul
 *  products, as WASM has no fused multiply-add for an f32 TwoProduct and f64 cannot hold ab. */
NUMKONG_INLINE void nk_angular_through_i32_from_dot_v128_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                          nk_b128_vec_t const *target_sumsqs_vec,
                                                          nk_b128_vec_t *result_vec) {
    v128_t const zeros_i32x4 = wasm_i32x4_splat(0), ones_f32x4 = wasm_f32x4_splat(1.0f);
    v128_t const dots_i32x4 = dots_vec->v128, targets_u32x4 = target_sumsqs_vec->v128;
    v128_t const query_u32x4 = wasm_u32x4_splat(query_sumsq);
    // Cauchy–Schwarz keeps the gap non-negative
    v128_t const gap_low_u64x2 = wasm_i64x2_sub(wasm_u64x2_extmul_low_u32x4(query_u32x4, targets_u32x4),
                                                wasm_i64x2_extmul_low_i32x4(dots_i32x4, dots_i32x4));
    v128_t const gap_high_u64x2 = wasm_i64x2_sub(wasm_u64x2_extmul_high_u32x4(query_u32x4, targets_u32x4),
                                                 wasm_i64x2_extmul_high_i32x4(dots_i32x4, dots_i32x4));
    // WASM has no u64 → f32 conversion, so the 32-bit words convert apart and recombine
    v128_t const gap_words_low_u32x4 = wasm_i32x4_shuffle(gap_low_u64x2, gap_high_u64x2, 0, 2, 4, 6);
    v128_t const gap_words_high_u32x4 = wasm_i32x4_shuffle(gap_low_u64x2, gap_high_u64x2, 1, 3, 5, 7);
    v128_t const gap_f32x4 = wasm_f32x4_add(
        wasm_f32x4_mul(wasm_f32x4_convert_u32x4(gap_words_high_u32x4), wasm_f32x4_splat(4294967296.0f)),
        wasm_f32x4_convert_u32x4(gap_words_low_u32x4));
    v128_t const dots_f32x4 = wasm_f32x4_convert_i32x4(dots_i32x4);
    v128_t const product_f32x4 = wasm_f32x4_mul(wasm_f32x4_splat((nk_f32_t)query_sumsq),
                                                wasm_f32x4_convert_u32x4(targets_u32x4));
    v128_t const norm_f32x4 = wasm_f32x4_sqrt(product_f32x4);
    // A positive dot takes (ab − d²) / (ab + d × s), any other 1 + |d| / s
    v128_t const positive_i32x4 = wasm_i32x4_gt(dots_i32x4, zeros_i32x4);
    v128_t const numerator_f32x4 = wasm_v128_bitselect(gap_f32x4, wasm_f32x4_neg(dots_f32x4), positive_i32x4);
    v128_t const denominator_f32x4 = wasm_v128_bitselect(
        wasm_f32x4_add(product_f32x4, wasm_f32x4_mul(dots_f32x4, norm_f32x4)), norm_f32x4, positive_i32x4);
    v128_t angular_f32x4 = wasm_f32x4_add(wasm_f32x4_div(numerator_f32x4, denominator_f32x4),
                                          wasm_v128_andnot(ones_f32x4, positive_i32x4));
    // A zero norm gives 1, and two zero norms give 0
    v128_t const target_zero_i32x4 = wasm_i32x4_eq(targets_u32x4, zeros_i32x4);
    v128_t const query_zero_i32x4 = wasm_i32x4_eq(query_u32x4, zeros_i32x4);
    angular_f32x4 = wasm_v128_bitselect(ones_f32x4, angular_f32x4, wasm_v128_or(target_zero_i32x4, query_zero_i32x4));
    result_vec->v128 = wasm_v128_andnot(angular_f32x4, wasm_v128_and(target_zero_i32x4, query_zero_i32x4));
}

/** Euclidean from_dot for 4 i32 dots and u32 norms. The exact q + t − 2d spans 34 bits, so
 *  it sums as 16-bit halves that f32 holds exactly, rounding once without 64-bit lanes. */
NUMKONG_INLINE void nk_euclidean_through_i32_from_dot_v128_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                            nk_b128_vec_t const *target_sumsqs_vec,
                                                            nk_b128_vec_t *result_vec) {
    v128_t const low_mask_u32x4 = wasm_u32x4_splat(0xFFFF);
    v128_t const dots_i32x4 = dots_vec->v128, targets_u32x4 = target_sumsqs_vec->v128;
    // The arithmetic shift keeps d = (d >> 16) × 2¹⁶ + (d & 0xFFFF) for a negative dot
    v128_t const dots_high_i32x4 = wasm_i32x4_shr(dots_i32x4, 16);
    v128_t const dots_low_i32x4 = wasm_v128_and(dots_i32x4, low_mask_u32x4);
    v128_t const high_i32x4 = wasm_i32x4_sub(
        wasm_i32x4_add(wasm_u32x4_shr(targets_u32x4, 16), wasm_u32x4_splat(query_sumsq >> 16)),
        wasm_i32x4_add(dots_high_i32x4, dots_high_i32x4));
    v128_t const low_i32x4 = wasm_i32x4_sub(
        wasm_i32x4_add(wasm_v128_and(targets_u32x4, low_mask_u32x4), wasm_u32x4_splat(query_sumsq & 0xFFFF)),
        wasm_i32x4_add(dots_low_i32x4, dots_low_i32x4));
    v128_t const distance_sq_f32x4 = wasm_f32x4_add(
        wasm_f32x4_mul(wasm_f32x4_convert_i32x4(high_i32x4), wasm_f32x4_splat(65536.0f)),
        wasm_f32x4_convert_i32x4(low_i32x4));
    result_vec->v128 = wasm_f32x4_sqrt(distance_sq_f32x4);
}

/** Angular from_dot for 4 u32 dots and norms. The gap ab − d² is exact in 64-bit @c extmul
 *  products, as WASM has no fused multiply-add for an f32 TwoProduct and f64 cannot hold ab. */
NUMKONG_INLINE void nk_angular_through_u32_from_dot_v128_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                          nk_b128_vec_t const *target_sumsqs_vec,
                                                          nk_b128_vec_t *result_vec) {
    v128_t const zeros_u32x4 = wasm_u32x4_splat(0), ones_f32x4 = wasm_f32x4_splat(1.0f);
    v128_t const dots_u32x4 = dots_vec->v128, targets_u32x4 = target_sumsqs_vec->v128;
    v128_t const query_u32x4 = wasm_u32x4_splat(query_sumsq);
    // Cauchy–Schwarz keeps the gap non-negative
    v128_t const gap_low_u64x2 = wasm_i64x2_sub(wasm_u64x2_extmul_low_u32x4(query_u32x4, targets_u32x4),
                                                wasm_u64x2_extmul_low_u32x4(dots_u32x4, dots_u32x4));
    v128_t const gap_high_u64x2 = wasm_i64x2_sub(wasm_u64x2_extmul_high_u32x4(query_u32x4, targets_u32x4),
                                                 wasm_u64x2_extmul_high_u32x4(dots_u32x4, dots_u32x4));
    // WASM has no u64 → f32 conversion, so the 32-bit words convert apart and recombine
    v128_t const gap_words_low_u32x4 = wasm_i32x4_shuffle(gap_low_u64x2, gap_high_u64x2, 0, 2, 4, 6);
    v128_t const gap_words_high_u32x4 = wasm_i32x4_shuffle(gap_low_u64x2, gap_high_u64x2, 1, 3, 5, 7);
    v128_t const gap_f32x4 = wasm_f32x4_add(
        wasm_f32x4_mul(wasm_f32x4_convert_u32x4(gap_words_high_u32x4), wasm_f32x4_splat(4294967296.0f)),
        wasm_f32x4_convert_u32x4(gap_words_low_u32x4));
    v128_t const product_f32x4 = wasm_f32x4_mul(wasm_f32x4_splat((nk_f32_t)query_sumsq),
                                                wasm_f32x4_convert_u32x4(targets_u32x4));
    v128_t const norm_f32x4 = wasm_f32x4_sqrt(product_f32x4);
    // (ab − d²) / (ab + d × s) keeps the small angles that 1 − d / s cancels
    v128_t angular_f32x4 = wasm_f32x4_div(
        gap_f32x4, wasm_f32x4_add(product_f32x4, wasm_f32x4_mul(wasm_f32x4_convert_u32x4(dots_u32x4), norm_f32x4)));
    // A zero norm or a zero dot gives 1, and two zero norms give 0
    v128_t const target_zero_u32x4 = wasm_i32x4_eq(targets_u32x4, zeros_u32x4);
    v128_t const query_zero_u32x4 = wasm_i32x4_eq(query_u32x4, zeros_u32x4);
    v128_t const unit_u32x4 = wasm_v128_or(wasm_i32x4_eq(dots_u32x4, zeros_u32x4),
                                           wasm_v128_or(target_zero_u32x4, query_zero_u32x4));
    angular_f32x4 = wasm_v128_bitselect(ones_f32x4, angular_f32x4, unit_u32x4);
    result_vec->v128 = wasm_v128_andnot(angular_f32x4, wasm_v128_and(target_zero_u32x4, query_zero_u32x4));
}

/** Euclidean from_dot for 4 u32 dots and norms. The exact q + t − 2d spans 33 bits, so it sums
 *  as 16-bit halves that f32 holds exactly, rounding once without 64-bit lanes. */
NUMKONG_INLINE void nk_euclidean_through_u32_from_dot_v128_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                            nk_b128_vec_t const *target_sumsqs_vec,
                                                            nk_b128_vec_t *result_vec) {
    v128_t const low_mask_u32x4 = wasm_u32x4_splat(0xFFFF);
    v128_t const dots_u32x4 = dots_vec->v128, targets_u32x4 = target_sumsqs_vec->v128;
    v128_t const dots_high_u32x4 = wasm_u32x4_shr(dots_u32x4, 16);
    v128_t const dots_low_u32x4 = wasm_v128_and(dots_u32x4, low_mask_u32x4);
    v128_t const high_i32x4 = wasm_i32x4_sub(
        wasm_i32x4_add(wasm_u32x4_shr(targets_u32x4, 16), wasm_u32x4_splat(query_sumsq >> 16)),
        wasm_i32x4_add(dots_high_u32x4, dots_high_u32x4));
    v128_t const low_i32x4 = wasm_i32x4_sub(
        wasm_i32x4_add(wasm_v128_and(targets_u32x4, low_mask_u32x4), wasm_u32x4_splat(query_sumsq & 0xFFFF)),
        wasm_i32x4_add(dots_low_u32x4, dots_low_u32x4));
    v128_t const distance_sq_f32x4 = wasm_f32x4_add(
        wasm_f32x4_mul(wasm_f32x4_convert_i32x4(high_i32x4), wasm_f32x4_splat(65536.0f)),
        wasm_f32x4_convert_i32x4(low_i32x4));
    result_vec->v128 = wasm_f32x4_sqrt(distance_sq_f32x4);
}

#pragma endregion Spatial From Dot Helpers

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_WASM_V128_
#endif // NUMKONG_ARCH_WASM_
#endif // NUMKONG_SPATIAL_V128_H
