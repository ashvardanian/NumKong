/**
 *  @brief SIMD-accelerated Dot Products for WASM.
 *  @file include/numkong/dot/v128.h
 *  @author Ash Vardanian
 *  @date September 12, 2026
 *
 *  Requires Emscripten 3.1.27+ or a WASI SDK with the `-msimd128` flag.
 *
 *  Key optimizations:
 *  - i8/u8 dot products widen to i16 and multiply-add adjacent pairs with `i32x4.dot_i16x8_s`
 *  - bf16 dot products shift the even and odd halves into f32 lanes without a separate upcast
 *  - u1 dot products accumulate popcounts in u8 for 31 iterations before widening
 */

#ifndef NK_DOT_V128_H
#define NK_DOT_V128_H

#if NK_TARGET_V128

#include "numkong/types.h"
#include "numkong/reduce/v128.h" // `nk_reduce_add_i32x4_to_i64_v128_`, `nk_reduce_add_u8x16_v128_`
#include "numkong/cast/serial.h"
#include "numkong/cast/v128.h" // `nk_load_b128_v128_`, `nk_bf16x4_to_f32x4_v128_`, `nk_e5m2x4_to_f32x4_v128_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

NK_HELPER_INLINE nk_f64_t nk_dot_stable_sum_f64x2_v128_(v128_t sum_f64x2, v128_t compensation_f64x2) {
    v128_t tentative_sum_f64x2 = wasm_f64x2_add(sum_f64x2, compensation_f64x2);
    v128_t virtual_addend_f64x2 = wasm_f64x2_sub(tentative_sum_f64x2, sum_f64x2);
    v128_t rounding_error_f64x2 = wasm_f64x2_add(
        wasm_f64x2_sub(sum_f64x2, wasm_f64x2_sub(tentative_sum_f64x2, virtual_addend_f64x2)),
        wasm_f64x2_sub(compensation_f64x2, virtual_addend_f64x2));
    nk_f64_t lower_sum = wasm_f64x2_extract_lane(tentative_sum_f64x2, 0);
    nk_f64_t upper_sum = wasm_f64x2_extract_lane(tentative_sum_f64x2, 1);
    nk_f64_t lower_error = wasm_f64x2_extract_lane(rounding_error_f64x2, 0);
    nk_f64_t upper_error = wasm_f64x2_extract_lane(rounding_error_f64x2, 1);
    nk_f64_t tentative_sum = lower_sum + upper_sum;
    nk_f64_t virtual_addend = tentative_sum - lower_sum;
    nk_f64_t rounding_error = (lower_sum - (tentative_sum - virtual_addend)) + (upper_sum - virtual_addend);
    return tentative_sum + (lower_error + upper_error + rounding_error);
}

NK_API_COMPTIME void nk_dot_bf16_v128(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t mask_high_u32x4 = wasm_i32x4_splat((int)0xFFFF0000);
    nk_bf16_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b128_vec_t a_bf16_vec, b_bf16_vec;

nk_dot_bf16_v128_cycle:
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
    v128_t a_even_f32x4 = wasm_i32x4_shl(a_bf16_vec.v128, 16); // even bf16 lanes into the f32 high half
    v128_t b_even_f32x4 = wasm_i32x4_shl(b_bf16_vec.v128, 16);
    sum_f32x4 = wasm_f32x4_add(sum_f32x4, wasm_f32x4_mul(a_even_f32x4, b_even_f32x4));
    v128_t a_odd_f32x4 = wasm_v128_and(a_bf16_vec.v128, mask_high_u32x4); // odd lanes already sit there
    v128_t b_odd_f32x4 = wasm_v128_and(b_bf16_vec.v128, mask_high_u32x4);
    sum_f32x4 = wasm_f32x4_add(sum_f32x4, wasm_f32x4_mul(a_odd_f32x4, b_odd_f32x4));
    if (count_scalars) goto nk_dot_bf16_v128_cycle;

    *result = nk_reduce_add_f32x4_v128_(sum_f32x4);
}

NK_API_COMPTIME void nk_dot_i8_v128(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result) {
    nk_i64_t sum_total = 0;
    nk_size_t i = 0;

    // Windowed accumulation loop
    while (i + 16 <= n) {
        v128_t sum_i32x4 = wasm_i32x4_splat(0);

        nk_size_t cycle = 0;
        // Two pairwise dots add at most 65536 per lane, so 32767 iterations fit i32
        for (; cycle < 32767 && i + 16 <= n; ++cycle, i += 16) {
            v128_t a_i8x16 = wasm_v128_load(a + i);
            v128_t b_i8x16 = wasm_v128_load(b + i);

            v128_t a_low_i16x8 = wasm_i16x8_extend_low_i8x16(a_i8x16);
            v128_t a_high_i16x8 = wasm_i16x8_extend_high_i8x16(a_i8x16);
            v128_t b_low_i16x8 = wasm_i16x8_extend_low_i8x16(b_i8x16);
            v128_t b_high_i16x8 = wasm_i16x8_extend_high_i8x16(b_i8x16);
            sum_i32x4 = wasm_i32x4_add(sum_i32x4, wasm_i32x4_dot_i16x8(a_low_i16x8, b_low_i16x8));
            sum_i32x4 = wasm_i32x4_add(sum_i32x4, wasm_i32x4_dot_i16x8(a_high_i16x8, b_high_i16x8));
        }

        // Reduce window to scalar
        sum_total += nk_reduce_add_i32x4_to_i64_v128_(sum_i32x4);
    }

    // Handle tail elements
    for (; i < n; i++) { sum_total += (nk_i32_t)a[i] * (nk_i32_t)b[i]; }

    *result = (nk_i32_t)sum_total;
}

NK_API_COMPTIME void nk_dot_u8_v128(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_u64_t sum_total = 0;
    nk_size_t i = 0;

    // Windowed accumulation loop
    while (i + 16 <= n) {
        v128_t sum_u32x4 = wasm_i32x4_splat(0);

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
            sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_dot_i16x8(a_low_u16x8, b_low_u16x8));
            sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_dot_i16x8(a_high_u16x8, b_high_u16x8));
        }

        // Reduce window to scalar
        sum_total += nk_reduce_add_u32x4_to_u64_v128_(sum_u32x4);
    }

    // Handle tail elements
    for (; i < n; i++) { sum_total += (nk_u32_t)a[i] * (nk_u32_t)b[i]; }

    *result = (nk_u32_t)sum_total;
}

NK_API_COMPTIME void nk_dot_u1_v128(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result) {
    nk_u8_t const *a_bytes = (nk_u8_t const *)a;
    nk_u8_t const *b_bytes = (nk_u8_t const *)b;
    nk_size_t n_bytes = n_bits / NK_BITS_PER_BYTE;

    nk_u32_t dot = 0;
    nk_size_t i = 0;

    // Windowed accumulation loop
    while (i + 16 <= n_bytes) {
        v128_t popcount_u8x16 = wasm_i8x16_splat(0);

        nk_size_t cycle = 0;
        // Accumulate 31 iterations in u8 before widening: 31 × 8 = 248 < 255
        for (; cycle < 31 && i + 16 <= n_bytes; ++cycle, i += 16) {
            v128_t a_u8x16 = wasm_v128_load(a_bytes + i);
            v128_t b_u8x16 = wasm_v128_load(b_bytes + i);
            v128_t and_u8x16 = wasm_v128_and(a_u8x16, b_u8x16);
            v128_t popcnt_u8x16 = wasm_i8x16_popcnt(and_u8x16);
            popcount_u8x16 = wasm_i8x16_add(popcount_u8x16, popcnt_u8x16);
        }

        // Widen once per window: u8 → u16 → u32
        dot += nk_reduce_add_u8x16_v128_(popcount_u8x16);
    }

    // Handle tail bytes
    for (; i < n_bytes; i++) {
        nk_u8_t and_byte = a_bytes[i] & b_bytes[i];
        dot += nk_u1x8_popcount_(and_byte);
    }

    *result = dot;
}

NK_HELPER_INLINE void nk_load_e5m2x4_to_f32x4_v128_(void const *src, nk_b128_vec_t *dst) {
    nk_b32_vec_t raw;
    nk_copy_bytes_(&raw, src, 4);
    *dst = nk_e5m2x4_to_f32x4_v128_(raw);
}

NK_HELPER_INLINE void nk_partial_load_e5m2x4_to_f32x4_v128_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    nk_b32_vec_t raw = {0};
    nk_copy_bytes_(&raw, src, n * sizeof(nk_e5m2_t));
    *dst = nk_e5m2x4_to_f32x4_v128_(raw);
}

typedef struct nk_dot_bf16x8_state_v128_t {
    v128_t sum_f32x4;
} nk_dot_bf16x8_state_v128_t;

NK_HELPER_INLINE void nk_dot_bf16x8_init_v128(nk_dot_bf16x8_state_v128_t *state) {
    state->sum_f32x4 = wasm_f32x4_splat(0.0f);
}

NK_HELPER_INLINE void nk_dot_bf16x8_update_v128(nk_dot_bf16x8_state_v128_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                                nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    v128_t mask_high_u32x4 = wasm_i32x4_splat((int)0xFFFF0000);
    v128_t a_even_f32x4 = wasm_i32x4_shl(a.v128, 16);
    v128_t b_even_f32x4 = wasm_i32x4_shl(b.v128, 16);
    state->sum_f32x4 = wasm_f32x4_add(state->sum_f32x4, wasm_f32x4_mul(a_even_f32x4, b_even_f32x4));
    v128_t a_odd_f32x4 = wasm_v128_and(a.v128, mask_high_u32x4);
    v128_t b_odd_f32x4 = wasm_v128_and(b.v128, mask_high_u32x4);
    state->sum_f32x4 = wasm_f32x4_add(state->sum_f32x4, wasm_f32x4_mul(a_odd_f32x4, b_odd_f32x4));
}

NK_HELPER_INLINE void nk_dot_bf16x8_finalize_v128(                                        //
    nk_dot_bf16x8_state_v128_t const *state_a, nk_dot_bf16x8_state_v128_t const *state_b, //
    nk_dot_bf16x8_state_v128_t const *state_c, nk_dot_bf16x8_state_v128_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f32s[0] = nk_reduce_add_f32x4_v128_(state_a->sum_f32x4);
    result->f32s[1] = nk_reduce_add_f32x4_v128_(state_b->sum_f32x4);
    result->f32s[2] = nk_reduce_add_f32x4_v128_(state_c->sum_f32x4);
    result->f32s[3] = nk_reduce_add_f32x4_v128_(state_d->sum_f32x4);
}

typedef struct nk_dot_i8x16_state_v128_t {
    v128_t sum_i32x4; // two pairwise dots per step add at most 65536 per lane, so 32767 steps fit i32
} nk_dot_i8x16_state_v128_t;

NK_HELPER_INLINE void nk_dot_i8x16_init_v128(nk_dot_i8x16_state_v128_t *state) {
    state->sum_i32x4 = wasm_i32x4_splat(0);
}

NK_HELPER_INLINE void nk_dot_i8x16_update_v128(nk_dot_i8x16_state_v128_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                               nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    v128_t a_low_i16x8 = wasm_i16x8_extend_low_i8x16(a.v128);
    v128_t a_high_i16x8 = wasm_i16x8_extend_high_i8x16(a.v128);
    v128_t b_low_i16x8 = wasm_i16x8_extend_low_i8x16(b.v128);
    v128_t b_high_i16x8 = wasm_i16x8_extend_high_i8x16(b.v128);
    state->sum_i32x4 = wasm_i32x4_add(state->sum_i32x4, wasm_i32x4_dot_i16x8(a_low_i16x8, b_low_i16x8));
    state->sum_i32x4 = wasm_i32x4_add(state->sum_i32x4, wasm_i32x4_dot_i16x8(a_high_i16x8, b_high_i16x8));
}

NK_HELPER_INLINE void nk_dot_i8x16_finalize_v128(                                       //
    nk_dot_i8x16_state_v128_t const *state_a, nk_dot_i8x16_state_v128_t const *state_b, //
    nk_dot_i8x16_state_v128_t const *state_c, nk_dot_i8x16_state_v128_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->i32s[0] = nk_reduce_add_i32x4_v128_(state_a->sum_i32x4);
    result->i32s[1] = nk_reduce_add_i32x4_v128_(state_b->sum_i32x4);
    result->i32s[2] = nk_reduce_add_i32x4_v128_(state_c->sum_i32x4);
    result->i32s[3] = nk_reduce_add_i32x4_v128_(state_d->sum_i32x4);
}

typedef struct nk_dot_u8x16_state_v128_t {
    v128_t sum_u32x4; // two pairwise dots per step add at most 4 × 255² per lane, so 8191 steps stay below 2³¹
} nk_dot_u8x16_state_v128_t;

NK_HELPER_INLINE void nk_dot_u8x16_init_v128(nk_dot_u8x16_state_v128_t *state) {
    state->sum_u32x4 = wasm_i32x4_splat(0);
}

NK_HELPER_INLINE void nk_dot_u8x16_update_v128(nk_dot_u8x16_state_v128_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                               nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    v128_t a_low_u16x8 = wasm_u16x8_extend_low_u8x16(a.v128); // u16 fits the signed pairwise dot
    v128_t a_high_u16x8 = wasm_u16x8_extend_high_u8x16(a.v128);
    v128_t b_low_u16x8 = wasm_u16x8_extend_low_u8x16(b.v128);
    v128_t b_high_u16x8 = wasm_u16x8_extend_high_u8x16(b.v128);
    state->sum_u32x4 = wasm_i32x4_add(state->sum_u32x4, wasm_i32x4_dot_i16x8(a_low_u16x8, b_low_u16x8));
    state->sum_u32x4 = wasm_i32x4_add(state->sum_u32x4, wasm_i32x4_dot_i16x8(a_high_u16x8, b_high_u16x8));
}

NK_HELPER_INLINE void nk_dot_u8x16_finalize_v128(                                       //
    nk_dot_u8x16_state_v128_t const *state_a, nk_dot_u8x16_state_v128_t const *state_b, //
    nk_dot_u8x16_state_v128_t const *state_c, nk_dot_u8x16_state_v128_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->u32s[0] = nk_reduce_add_u32x4_v128_(state_a->sum_u32x4);
    result->u32s[1] = nk_reduce_add_u32x4_v128_(state_b->sum_u32x4);
    result->u32s[2] = nk_reduce_add_u32x4_v128_(state_c->sum_u32x4);
    result->u32s[3] = nk_reduce_add_u32x4_v128_(state_d->sum_u32x4);
}

typedef struct nk_sum_u8x16_state_v128_t {
    v128_t sum_u32x4;
} nk_sum_u8x16_state_v128_t;

NK_HELPER_INLINE void nk_sum_u8x16_init_v128(nk_sum_u8x16_state_v128_t *state) {
    state->sum_u32x4 = wasm_i32x4_splat(0);
}

NK_HELPER_INLINE void nk_sum_u8x16_update_v128(nk_sum_u8x16_state_v128_t *state, nk_b128_vec_t v) {
    v128_t sum_u16x8 = wasm_u16x8_extadd_pairwise_u8x16(v.v128);
    v128_t sum_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(sum_u16x8);
    state->sum_u32x4 = wasm_i32x4_add(state->sum_u32x4, sum_u32x4);
}

NK_HELPER_INLINE nk_u32_t nk_sum_u8x16_finalize_v128(nk_sum_u8x16_state_v128_t const *state, nk_size_t count) {
    nk_unused_(count);
    return nk_reduce_add_u32x4_v128_(state->sum_u32x4);
}

typedef struct nk_sum_i4x32_state_v128_t {
    v128_t sum_i32x4;
} nk_sum_i4x32_state_v128_t;

NK_HELPER_INLINE void nk_sum_i4x32_init_v128(nk_sum_i4x32_state_v128_t *state) {
    state->sum_i32x4 = wasm_i32x4_splat(0);
}

NK_HELPER_INLINE void nk_sum_i4x32_update_v128(nk_sum_i4x32_state_v128_t *state, nk_b128_vec_t v) {
    v128_t nibble_mask_u8x16 = wasm_u8x16_splat(0x0F);
    v128_t bias_mask_u8x16 = wasm_u8x16_splat(0x08);
    v128_t low_u8x16 = wasm_v128_xor(wasm_v128_and(v.v128, nibble_mask_u8x16), bias_mask_u8x16);
    v128_t high_u8x16 = wasm_v128_xor(wasm_v128_and(wasm_u16x8_shr(v.v128, 4), nibble_mask_u8x16), bias_mask_u8x16);
    v128_t sum_low_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(wasm_u16x8_extadd_pairwise_u8x16(low_u8x16));
    v128_t sum_high_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(wasm_u16x8_extadd_pairwise_u8x16(high_u8x16));
    v128_t signed_sum_i32x4 = wasm_i32x4_sub(wasm_i32x4_add(sum_low_u32x4, sum_high_u32x4), wasm_i32x4_splat(64));
    state->sum_i32x4 = wasm_i32x4_add(state->sum_i32x4, signed_sum_i32x4);
}

NK_HELPER_INLINE nk_i32_t nk_sum_i4x32_finalize_v128(nk_sum_i4x32_state_v128_t const *state, nk_size_t count) {
    nk_unused_(count);
    return nk_reduce_add_i32x4_v128_(state->sum_i32x4);
}

typedef struct nk_dot_u1x128_state_v128_t {
    v128_t dot_count_u32x4;
} nk_dot_u1x128_state_v128_t;

NK_HELPER_INLINE void nk_dot_u1x128_init_v128(nk_dot_u1x128_state_v128_t *state) {
    state->dot_count_u32x4 = wasm_u32x4_const(0, 0, 0, 0);
}

NK_HELPER_INLINE void nk_dot_u1x128_update_v128(nk_dot_u1x128_state_v128_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                                nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    v128_t and_u8x16 = wasm_v128_and(a.v128, b.v128);
    v128_t popcount_u8x16 = wasm_i8x16_popcnt(and_u8x16);
    v128_t popcount_u16x8 = wasm_u16x8_extadd_pairwise_u8x16(popcount_u8x16);
    v128_t popcount_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(popcount_u16x8);
    state->dot_count_u32x4 = wasm_i32x4_add(state->dot_count_u32x4, popcount_u32x4);
}

NK_HELPER_INLINE void nk_dot_u1x128_finalize_v128(                                        //
    nk_dot_u1x128_state_v128_t const *state_a, nk_dot_u1x128_state_v128_t const *state_b, //
    nk_dot_u1x128_state_v128_t const *state_c, nk_dot_u1x128_state_v128_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    v128_t a_u32x4 = state_a->dot_count_u32x4, b_u32x4 = state_b->dot_count_u32x4;
    v128_t c_u32x4 = state_c->dot_count_u32x4, d_u32x4 = state_d->dot_count_u32x4;
    v128_t ab_low_u32x4 = wasm_i32x4_shuffle(a_u32x4, b_u32x4, 0, 4, 1, 5);  // a0 b0 a1 b1
    v128_t ab_high_u32x4 = wasm_i32x4_shuffle(a_u32x4, b_u32x4, 2, 6, 3, 7); // a2 b2 a3 b3
    v128_t cd_low_u32x4 = wasm_i32x4_shuffle(c_u32x4, d_u32x4, 0, 4, 1, 5);  // c0 d0 c1 d1
    v128_t cd_high_u32x4 = wasm_i32x4_shuffle(c_u32x4, d_u32x4, 2, 6, 3, 7); // c2 d2 c3 d3
    v128_t sum_02_u32x4 = wasm_i32x4_add(ab_low_u32x4, ab_high_u32x4);       // a02 b02 a13 b13
    v128_t sum_13_u32x4 = wasm_i32x4_add(cd_low_u32x4, cd_high_u32x4);       // c02 d02 c13 d13
    v128_t even_u32x4 = wasm_i32x4_shuffle(sum_02_u32x4, sum_13_u32x4, 0, 1, 4, 5);
    v128_t odd_u32x4 = wasm_i32x4_shuffle(sum_02_u32x4, sum_13_u32x4, 2, 3, 6, 7);
    result->v128 = wasm_i32x4_add(even_u32x4, odd_u32x4); // [sum_a, sum_b, sum_c, sum_d]
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128
#endif // NK_DOT_V128_H
