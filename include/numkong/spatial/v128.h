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

#ifndef NK_SPATIAL_V128_H
#define NK_SPATIAL_V128_H

#if NK_TARGET_V128

#include "numkong/types.h"
#include "numkong/scalar/v128.h" // `nk_f32_sqrt_v128`
#include "numkong/reduce/v128.h" // `nk_reduce_add_f32x4_v128_`, `nk_reduce_add_i32x4_to_i64_v128_`
#include "numkong/cast/serial.h"
#include "numkong/cast/v128.h" // `nk_load_b128_v128_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

NK_HELPER_INLINE nk_f64_t nk_angular_normalize_f64_v128_(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {
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

NK_API_COMPTIME void nk_sqeuclidean_bf16_v128(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
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

NK_API_COMPTIME void nk_euclidean_bf16_v128(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_f32_t l2sq;
    nk_sqeuclidean_bf16_v128(a, b, n, &l2sq);
    *result = nk_f32_sqrt_v128(l2sq);
}

NK_API_COMPTIME void nk_angular_bf16_v128(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
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
}

#pragma endregion BF16 Floats
#pragma region I8 and U8 Integers

NK_API_COMPTIME void nk_sqeuclidean_u8_v128(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
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

NK_API_COMPTIME void nk_euclidean_u8_v128(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq;
    nk_sqeuclidean_u8_v128(a, b, n, &distance_sq);
    *result = nk_f32_sqrt_v128((nk_f32_t)distance_sq);
}

NK_API_COMPTIME void nk_angular_u8_v128(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
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
}

NK_API_COMPTIME void nk_sqeuclidean_i8_v128(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result) {
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

NK_API_COMPTIME void nk_euclidean_i8_v128(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq;
    nk_sqeuclidean_i8_v128(a, b, n, &distance_sq);
    *result = nk_f32_sqrt_v128((nk_f32_t)distance_sq);
}

NK_API_COMPTIME void nk_angular_i8_v128(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
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
}

#pragma endregion I8 and U8 Integers
#pragma region Spatial From Dot Helpers

/** @brief Angular from_dot: computes 1 − dot / (√query_sumsq × √target_sumsq) for 4 pairs in f32.
 *  Separate square roots avoid overflowing the product of two finite-but-large norms. */
NK_HELPER_INLINE void nk_angular_through_f32_from_dot_v128_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
                                                            nk_b128_vec_t const *target_sumsqs_vec,
                                                            nk_b128_vec_t *result_vec) {
    v128_t dots_f32x4 = dots_vec->v128;
    v128_t query_sqrt_f32x4 = wasm_f32x4_sqrt(wasm_f32x4_splat(query_sumsq));
    v128_t target_sqrt_f32x4 = wasm_f32x4_sqrt(target_sumsqs_vec->v128);
    v128_t norm_f32x4 = wasm_f32x4_mul(query_sqrt_f32x4, target_sqrt_f32x4);
    v128_t normalized_f32x4 = wasm_f32x4_div(dots_f32x4, norm_f32x4);
    v128_t angular_f32x4 = wasm_f32x4_sub(wasm_f32x4_splat(1.0f), normalized_f32x4);
    result_vec->v128 = wasm_f32x4_max(angular_f32x4, wasm_f32x4_splat(0.0f));
}

/** @brief Euclidean from_dot: computes √(query_sumsq + target_sumsq − 2 × dot) for 4 pairs in f32. */
NK_HELPER_INLINE void nk_euclidean_through_f32_from_dot_v128_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
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

/** @brief Angular from_dot for i32 accumulators: cast to f32, separate-sqrt normalization. 4 pairs. */
NK_HELPER_INLINE void nk_angular_through_i32_from_dot_v128_(nk_b128_vec_t const *dots_vec, nk_i32_t query_sumsq,
                                                            nk_b128_vec_t const *target_sumsqs_vec,
                                                            nk_b128_vec_t *result_vec) {
    v128_t dots_f32x4 = wasm_f32x4_convert_i32x4(dots_vec->v128);
    v128_t query_sqrt_f32x4 = wasm_f32x4_sqrt(wasm_f32x4_splat((nk_f32_t)query_sumsq));
    v128_t target_sqrt_f32x4 = wasm_f32x4_sqrt(wasm_f32x4_convert_i32x4(target_sumsqs_vec->v128));
    v128_t norm_f32x4 = wasm_f32x4_mul(query_sqrt_f32x4, target_sqrt_f32x4);
    v128_t normalized_f32x4 = wasm_f32x4_div(dots_f32x4, norm_f32x4);
    v128_t angular_f32x4 = wasm_f32x4_sub(wasm_f32x4_splat(1.0f), normalized_f32x4);
    result_vec->v128 = wasm_f32x4_max(angular_f32x4, wasm_f32x4_splat(0.0f));
}

/** @brief Euclidean from_dot for i32 accumulators: cast to f32, then √(a² + b² − 2ab). 4 pairs. */
NK_HELPER_INLINE void nk_euclidean_through_i32_from_dot_v128_(nk_b128_vec_t const *dots_vec, nk_i32_t query_sumsq,
                                                              nk_b128_vec_t const *target_sumsqs_vec,
                                                              nk_b128_vec_t *result_vec) {
    v128_t dots_f32x4 = wasm_f32x4_convert_i32x4(dots_vec->v128);
    v128_t query_sumsq_f32x4 = wasm_f32x4_splat((nk_f32_t)query_sumsq);
    v128_t two_f32x4 = wasm_f32x4_splat(2.0f);
    v128_t sum_sq_f32x4 = wasm_f32x4_add(query_sumsq_f32x4, wasm_f32x4_convert_i32x4(target_sumsqs_vec->v128));
    v128_t dist_sq_f32x4 = wasm_f32x4_sub(sum_sq_f32x4, wasm_f32x4_mul(two_f32x4, dots_f32x4));
    dist_sq_f32x4 = wasm_f32x4_max(dist_sq_f32x4, wasm_f32x4_splat(0.0f));
    result_vec->v128 = wasm_f32x4_sqrt(dist_sq_f32x4);
}

/** @brief Angular from_dot for u32 accumulators: cast to f32, separate-sqrt normalization. 4 pairs. */
NK_HELPER_INLINE void nk_angular_through_u32_from_dot_v128_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                            nk_b128_vec_t const *target_sumsqs_vec,
                                                            nk_b128_vec_t *result_vec) {
    v128_t dots_f32x4 = wasm_f32x4_convert_u32x4(dots_vec->v128);
    v128_t query_sqrt_f32x4 = wasm_f32x4_sqrt(wasm_f32x4_splat((nk_f32_t)query_sumsq));
    v128_t target_sqrt_f32x4 = wasm_f32x4_sqrt(wasm_f32x4_convert_u32x4(target_sumsqs_vec->v128));
    v128_t norm_f32x4 = wasm_f32x4_mul(query_sqrt_f32x4, target_sqrt_f32x4);
    v128_t normalized_f32x4 = wasm_f32x4_div(dots_f32x4, norm_f32x4);
    v128_t angular_f32x4 = wasm_f32x4_sub(wasm_f32x4_splat(1.0f), normalized_f32x4);
    result_vec->v128 = wasm_f32x4_max(angular_f32x4, wasm_f32x4_splat(0.0f));
}

/** @brief Euclidean from_dot for u32 accumulators: cast to f32, then √(a² + b² − 2ab). 4 pairs. */
NK_HELPER_INLINE void nk_euclidean_through_u32_from_dot_v128_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                              nk_b128_vec_t const *target_sumsqs_vec,
                                                              nk_b128_vec_t *result_vec) {
    v128_t dots_f32x4 = wasm_f32x4_convert_u32x4(dots_vec->v128);
    v128_t query_sumsq_f32x4 = wasm_f32x4_splat((nk_f32_t)query_sumsq);
    v128_t two_f32x4 = wasm_f32x4_splat(2.0f);
    v128_t sum_sq_f32x4 = wasm_f32x4_add(query_sumsq_f32x4, wasm_f32x4_convert_u32x4(target_sumsqs_vec->v128));
    v128_t dist_sq_f32x4 = wasm_f32x4_sub(sum_sq_f32x4, wasm_f32x4_mul(two_f32x4, dots_f32x4));
    dist_sq_f32x4 = wasm_f32x4_max(dist_sq_f32x4, wasm_f32x4_splat(0.0f));
    result_vec->v128 = wasm_f32x4_sqrt(dist_sq_f32x4);
}

#pragma endregion Spatial From Dot Helpers

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128
#endif // NK_SPATIAL_V128_H
