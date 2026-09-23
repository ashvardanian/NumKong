/**
 *  @brief SIMD-accelerated Set Similarity Measures for WASM.
 *  @file include/numkong/set/v128.h
 *  @author Ash Vardanian
 *  @date September 12, 2026
 *
 *  Hamming and Jaccard over packed bits accumulate byte popcounts in u8 for 31 steps before widening once, since
 *  31 × 8 = 248 stays below 255; the u8, u16 and u32 kernels count equal lanes with a compare and a shift.
 */

#ifndef NK_SET_V128_H
#define NK_SET_V128_H

#if NK_TARGET_V128

#include "numkong/types.h"
#include "numkong/reduce/v128.h" // `nk_reduce_add_u8x16_v128_`, `nk_reduce_add_u32x4_v128_`
#include "numkong/set/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

#pragma region Binary Sets

NK_API_COMPTIME void nk_hamming_u1_v128(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_u8_t const *a_bytes = (nk_u8_t const *)a;
    nk_u8_t const *b_bytes = (nk_u8_t const *)b;
    nk_size_t n_bytes = n / NK_BITS_PER_BYTE;

    nk_u32_t differences = 0;
    nk_size_t i = 0;

    // Windowed accumulation loop
    while (i + 16 <= n_bytes) {
        v128_t popcount_u8x16 = wasm_i8x16_splat(0);

        nk_size_t cycle = 0;
        // Accumulate 31 iterations in u8 before widening: 31 × 8 = 248 < 255
        for (; cycle < 31 && i + 16 <= n_bytes; ++cycle, i += 16) {
            v128_t a_u8x16 = wasm_v128_load(a_bytes + i);
            v128_t b_u8x16 = wasm_v128_load(b_bytes + i);
            v128_t xor_u8x16 = wasm_v128_xor(a_u8x16, b_u8x16);
            v128_t popcnt_u8x16 = wasm_i8x16_popcnt(xor_u8x16);
            popcount_u8x16 = wasm_i8x16_add(popcount_u8x16, popcnt_u8x16);
        }

        // Widen once per window: u8 → u16 → u32
        differences += nk_reduce_add_u8x16_v128_(popcount_u8x16);
    }

    // Handle tail bytes
    for (; i < n_bytes; i++) {
        nk_u8_t xor_byte = a_bytes[i] ^ b_bytes[i];
        differences += nk_u1x8_popcount_(xor_byte);
    }

    *result = differences;
}

NK_API_COMPTIME void nk_jaccard_u1_v128(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u8_t const *a_bytes = (nk_u8_t const *)a;
    nk_u8_t const *b_bytes = (nk_u8_t const *)b;
    nk_size_t n_bytes = n / NK_BITS_PER_BYTE;

    nk_u32_t intersection = 0;
    nk_u32_t union_count = 0;
    nk_size_t i = 0;

    // Windowed accumulation loop
    while (i + 16 <= n_bytes) {
        v128_t popcount_and_u8x16 = wasm_i8x16_splat(0);
        v128_t popcount_or_u8x16 = wasm_i8x16_splat(0);

        nk_size_t cycle = 0;
        // Accumulate 31 iterations in u8 before widening: 31 × 8 = 248 < 255
        for (; cycle < 31 && i + 16 <= n_bytes; ++cycle, i += 16) {
            v128_t a_u8x16 = wasm_v128_load(a_bytes + i);
            v128_t b_u8x16 = wasm_v128_load(b_bytes + i);
            v128_t and_u8x16 = wasm_v128_and(a_u8x16, b_u8x16);
            v128_t popcnt_and_u8x16 = wasm_i8x16_popcnt(and_u8x16);
            popcount_and_u8x16 = wasm_i8x16_add(popcount_and_u8x16, popcnt_and_u8x16);
            v128_t or_u8x16 = wasm_v128_or(a_u8x16, b_u8x16);
            v128_t popcnt_or_u8x16 = wasm_i8x16_popcnt(or_u8x16);
            popcount_or_u8x16 = wasm_i8x16_add(popcount_or_u8x16, popcnt_or_u8x16);
        }

        // Widen once per window
        intersection += nk_reduce_add_u8x16_v128_(popcount_and_u8x16);
        union_count += nk_reduce_add_u8x16_v128_(popcount_or_u8x16);
    }

    // Handle tail bytes
    for (; i < n_bytes; i++) {
        nk_u8_t a_byte = a_bytes[i];
        nk_u8_t b_byte = b_bytes[i];
        intersection += nk_u1x8_popcount_(a_byte & b_byte);
        union_count += nk_u1x8_popcount_(a_byte | b_byte);
    }

    // Jaccard distance = 1 - (intersection / union)
    *result = union_count > 0 ? 1.0f - ((nk_f32_t)intersection / (nk_f32_t)union_count) : 0.0f;
}

#pragma endregion Binary Sets

#pragma region Integer Sets

NK_API_COMPTIME void nk_hamming_u8_v128(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_u32_t sum_total = 0;
    nk_size_t i = 0;

    // Windowed accumulation: outer loop for windows, inner loop for iterations within window
    while (i + 16 <= n) {
        v128_t sum_u8x16 = wasm_i8x16_splat(0);

        nk_size_t cycle = 0;
        // Accumulate up to 31 iterations in u8: 31 × 1 = 31 < 255
        for (; cycle < 31 && i + 16 <= n; ++cycle, i += 16) {
            v128_t a_u8x16 = wasm_v128_load(a + i);
            v128_t b_u8x16 = wasm_v128_load(b + i);
            v128_t neq_mask_u8x16 = wasm_i8x16_ne(a_u8x16, b_u8x16);
            v128_t neq_count_u8x16 = wasm_v128_and(neq_mask_u8x16, wasm_i8x16_splat(1));
            sum_u8x16 = wasm_i8x16_add(sum_u8x16, neq_count_u8x16);
        }

        // Widen and reduce once per window
        sum_total += nk_reduce_add_u8x16_v128_(sum_u8x16);
    }

    // Traditional tail loop: handle remaining bytes (0-15) scalar-style
    for (; i < n; i++) { sum_total += (a[i] != b[i]); }

    *result = sum_total;
}

NK_API_COMPTIME void nk_jaccard_u32_v128(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t matches = 0;
    nk_size_t i = 0;
    v128_t matches_u32x4 = wasm_i32x4_splat(0);

    for (; i + 4 <= n; i += 4) {
        v128_t a_u32x4 = wasm_v128_load(a + i);
        v128_t b_u32x4 = wasm_v128_load(b + i);
        v128_t eq_mask_u32x4 = wasm_i32x4_eq(a_u32x4, b_u32x4);
        v128_t match_u32x4 = wasm_u32x4_shr(eq_mask_u32x4, 31);
        matches_u32x4 = wasm_i32x4_add(matches_u32x4, match_u32x4);
    }

    matches += nk_reduce_add_u32x4_v128_(matches_u32x4);
    for (; i < n; ++i) matches += (a[i] == b[i]);

    *result = (n != 0) ? 1.0f - (nk_f32_t)matches / (nk_f32_t)n : 0.0f;
}

NK_API_COMPTIME void nk_jaccard_u16_v128(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t matches = 0;
    nk_size_t i = 0;
    v128_t matches_u32x4 = wasm_i32x4_splat(0);

    for (; i + 8 <= n; i += 8) {
        v128_t a_u16x8 = wasm_v128_load(a + i);
        v128_t b_u16x8 = wasm_v128_load(b + i);
        v128_t eq_mask_u16x8 = wasm_i16x8_eq(a_u16x8, b_u16x8);
        v128_t match_u16x8 = wasm_u16x8_shr(eq_mask_u16x8, 15);
        matches_u32x4 = wasm_i32x4_add(matches_u32x4, wasm_u32x4_extadd_pairwise_u16x8(match_u16x8));
    }

    matches += nk_reduce_add_u32x4_v128_(matches_u32x4);
    for (; i < n; ++i) matches += (a[i] == b[i]);

    *result = (n != 0) ? 1.0f - (nk_f32_t)matches / (nk_f32_t)n : 0.0f;
}

#pragma endregion Integer Sets

#pragma region Binary Sets from Dot

NK_HELPER_INLINE void nk_hamming_u32x4_from_dot_v128_( //
    nk_b128_vec_t const *dots_vec, nk_u32_t query_pop, nk_b128_vec_t const *target_pops_vec,
    nk_b128_vec_t *result_vec) {
    v128_t dots_u32x4 = dots_vec->v128;
    v128_t query_u32x4 = wasm_u32x4_splat(query_pop);
    v128_t target_u32x4 = target_pops_vec->v128;
    result_vec->v128 = wasm_i32x4_sub(wasm_i32x4_add(query_u32x4, target_u32x4), wasm_i32x4_shl(dots_u32x4, 1));
}

NK_HELPER_INLINE void nk_jaccard_f32x4_from_dot_v128_( //
    nk_b128_vec_t const *dots_vec, nk_u32_t query_pop, nk_b128_vec_t const *target_pops_vec,
    nk_b128_vec_t *result_vec) {
    v128_t dot_f32x4 = wasm_f32x4_convert_u32x4(dots_vec->v128);
    v128_t query_f32x4 = wasm_f32x4_splat((nk_f32_t)query_pop);
    v128_t target_f32x4 = wasm_f32x4_convert_u32x4(target_pops_vec->v128);
    v128_t union_f32x4 = wasm_f32x4_sub(wasm_f32x4_add(query_f32x4, target_f32x4), dot_f32x4);

    v128_t zero_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t one_f32x4 = wasm_f32x4_splat(1.0f);
    v128_t zero_mask_u32x4 = wasm_f32x4_eq(union_f32x4, zero_f32x4);
    v128_t safe_union_f32x4 = wasm_v128_bitselect(one_f32x4, union_f32x4, zero_mask_u32x4);

    v128_t ratio_f32x4 = wasm_f32x4_div(dot_f32x4, safe_union_f32x4);
    v128_t jaccard_f32x4 = wasm_f32x4_sub(one_f32x4, ratio_f32x4);
    result_vec->v128 = wasm_v128_bitselect(zero_f32x4, jaccard_f32x4, zero_mask_u32x4);
}

#pragma endregion Binary Sets from Dot

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128
#endif // NK_SET_V128_H
