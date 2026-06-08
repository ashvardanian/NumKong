/**
 *  @brief SIMD-accelerated Set Similarity Measures for Power ISA VSX.
 *  @file include/numkong/set/powervsx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/set.h
 *
 *  @section set_powervsx_instructions Power9 VSX Set Instructions
 *
 *  Key Power9 VSX instructions for binary/bitwise operations:
 *
 *      Intrinsic      Instruction          P9
 *      vec_popcnt     vpopcntb/h/w/d       2cy @ 2p    element-wise popcount
 *      vec_xor        xxlxor               1cy @ 4p
 *      vec_and        xxland               1cy @ 4p
 *      vec_or         xxlor                1cy @ 4p
 *      vec_cmpne      vcmpneb/h/w          2cy @ 2p    byte/half/word not-equal
 *      vec_xl_len     lxvll                6cy @ 1p    partial vector load
 *
 *  Power9 has native doubleword `vpopcntd` instruction, providing efficient SIMD popcount
 *  with minimal data flow complexity. `vec_xl_len` enables branchless tail handling.
 *
 *  @section set_powervsx_stateful Stateful Streaming Logic
 *
 *  To build memory-optimal tiled algorithms, this file defines:
 *
 *  - nk_hamming_u1x128_state_powervsx_t for streaming Hamming distance
 *  - nk_jaccard_u1x128_state_powervsx_t for streaming Jaccard similarity
 *
 *  @code{c}
 *  nk_jaccard_u1x128_state_powervsx_t state_first, state_second, state_third, state_fourth;
 *  nk_jaccard_u1x128_init_powervsx(&state_first);
 *  // ... stream through packed binary vectors ...
 *  nk_jaccard_u1x128_finalize_powervsx(&state_first, &state_second, &state_third, &state_fourth,
 *      query_popcount, &target_popcounts_vec, total_dimensions, &result_vec);
 *  @endcode
 */
#ifndef NK_SET_POWERVSX_H
#define NK_SET_POWERVSX_H

#if NK_TARGET_POWER64_
#if NK_TARGET_POWERVSX

#include "numkong/types.h"
#include "numkong/set/serial.h"   // `nk_u1x8_popcount_`
#include "numkong/dot/powervsx.h" // `nk_hsum_u32x4_powervsx_`, `nk_hsum_u64x2_powervsx_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

NK_PUBLIC void nk_hamming_u1_powervsx(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);
    nk_vu64x2_t differences_u64x2 = vec_splats((nk_u64_t)0);
    nk_size_t i = 0;
    // Process 16 bytes at a time using doubleword popcount (vpopcntd)
    for (; i + 16 <= n_bytes; i += 16) {
        nk_vu8x16_t a_u8x16 = vec_xl(0, (nk_u8_t const *)(a + i));
        nk_vu8x16_t b_u8x16 = vec_xl(0, (nk_u8_t const *)(b + i));
        nk_vu8x16_t xor_u8x16 = vec_xor(a_u8x16, b_u8x16);
        nk_vu64x2_t popcnt_u64x2 = vec_popcnt((nk_vu64x2_t)xor_u8x16);
        differences_u64x2 = vec_add(differences_u64x2, popcnt_u64x2);
    }
    // Branchless tail: vec_xl_len zero-fills beyond remaining_bytes
    nk_size_t remaining_bytes = n_bytes - i;
    nk_vu8x16_t a_u8x16 = vec_xl_len((nk_u8_t *)(a + i), remaining_bytes);
    nk_vu8x16_t b_u8x16 = vec_xl_len((nk_u8_t *)(b + i), remaining_bytes);
    nk_vu8x16_t xor_u8x16 = vec_xor(a_u8x16, b_u8x16);
    nk_vu64x2_t popcnt_u64x2 = vec_popcnt((nk_vu64x2_t)xor_u8x16);
    differences_u64x2 = vec_add(differences_u64x2, popcnt_u64x2);
    *result = (nk_u32_t)nk_hsum_u64x2_powervsx_(differences_u64x2);
}

NK_PUBLIC void nk_jaccard_u1_powervsx(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);
    nk_vu64x2_t intersection_u64x2 = vec_splats((nk_u64_t)0);
    nk_vu64x2_t union_u64x2 = vec_splats((nk_u64_t)0);
    nk_size_t i = 0;
    for (; i + 16 <= n_bytes; i += 16) {
        nk_vu8x16_t a_u8x16 = vec_xl(0, (nk_u8_t const *)(a + i));
        nk_vu8x16_t b_u8x16 = vec_xl(0, (nk_u8_t const *)(b + i));
        nk_vu64x2_t and_popcnt_u64x2 = vec_popcnt((nk_vu64x2_t)vec_and(a_u8x16, b_u8x16));
        nk_vu64x2_t or_popcnt_u64x2 = vec_popcnt((nk_vu64x2_t)vec_or(a_u8x16, b_u8x16));
        intersection_u64x2 = vec_add(intersection_u64x2, and_popcnt_u64x2);
        union_u64x2 = vec_add(union_u64x2, or_popcnt_u64x2);
    }
    // Branchless tail
    nk_size_t remaining_bytes = n_bytes - i;
    nk_vu8x16_t a_u8x16 = vec_xl_len((nk_u8_t *)(a + i), remaining_bytes);
    nk_vu8x16_t b_u8x16 = vec_xl_len((nk_u8_t *)(b + i), remaining_bytes);
    nk_vu64x2_t and_popcnt_u64x2 = vec_popcnt((nk_vu64x2_t)vec_and(a_u8x16, b_u8x16));
    nk_vu64x2_t or_popcnt_u64x2 = vec_popcnt((nk_vu64x2_t)vec_or(a_u8x16, b_u8x16));
    intersection_u64x2 = vec_add(intersection_u64x2, and_popcnt_u64x2);
    union_u64x2 = vec_add(union_u64x2, or_popcnt_u64x2);
    nk_u32_t intersection_count = (nk_u32_t)nk_hsum_u64x2_powervsx_(intersection_u64x2);
    nk_u32_t union_count = (nk_u32_t)nk_hsum_u64x2_powervsx_(union_u64x2);
    *result = (union_count != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)union_count : 0.0f;
}

NK_PUBLIC void nk_hamming_u8_powervsx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_vu32x4_t differences_u32x4 = vec_splats((nk_u32_t)0);
    nk_vu8x16_t ones_u8x16 = vec_splats((nk_u8_t)1);
    nk_size_t i = 0;
    // Process 16 bytes at a time using vec_cmpne
    for (; i + 16 <= n; i += 16) {
        nk_vu8x16_t a_u8x16 = vec_xl(0, (nk_u8_t const *)(a + i));
        nk_vu8x16_t b_u8x16 = vec_xl(0, (nk_u8_t const *)(b + i));
        // vec_cmpne returns 0xFF for not-equal, 0x00 for equal
        // AND with 1 to get 0x01 for not-equal, then sum groups of 4 bytes → u32
        nk_vu8x16_t not_equal_u8x16 = vec_and((nk_vu8x16_t)vec_cmpne(a_u8x16, b_u8x16), ones_u8x16);
        differences_u32x4 = vec_sum4s(not_equal_u8x16, differences_u32x4);
    }
    // Branchless tail
    nk_size_t remaining_bytes = n - i;
    nk_vu8x16_t a_u8x16 = vec_xl_len((nk_u8_t *)(a + i), remaining_bytes);
    nk_vu8x16_t b_u8x16 = vec_xl_len((nk_u8_t *)(b + i), remaining_bytes);
    nk_vu8x16_t not_equal_u8x16 = vec_and((nk_vu8x16_t)vec_cmpne(a_u8x16, b_u8x16), ones_u8x16);
    differences_u32x4 = vec_sum4s(not_equal_u8x16, differences_u32x4);
    *result = nk_hsum_u32x4_powervsx_(differences_u32x4);
}

typedef struct nk_hamming_u1x128_state_powervsx_t {
    nk_vu32x4_t intersection_count_u32x4;
} nk_hamming_u1x128_state_powervsx_t;

NK_INTERNAL void nk_hamming_u1x128_init_powervsx(nk_hamming_u1x128_state_powervsx_t *state) {
    state->intersection_count_u32x4 = vec_splats((nk_u32_t)0);
}

NK_INTERNAL void nk_hamming_u1x128_update_powervsx(nk_hamming_u1x128_state_powervsx_t *state, nk_b128_vec_t a,
                                                   nk_b128_vec_t b, nk_size_t depth_offset,
                                                   nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);

    // Process one 128-bit chunk (native VSX register size).
    // Uses vector accumulation → horizontal sum deferred to finalize.
    //
    // Power9 VSX instruction characteristics:
    // - `vec_xor`:     xxlxor (V, V, V)             1cy, bitwise XOR
    // - `vec_popcnt`:  vpopcntw (V.4S, V.4S)        3cy, word popcount
    // - `vec_add`:     vadduwm (V.4S, V.4S, V.4S)   2cy, u32 add
    // Total: ~6cy per 128-bit chunk (horizontal sum deferred to finalize)

    // Step 1: Compute difference bits (A XOR B)
    nk_vu8x16_t a_u8x16 = *(nk_vu8x16_t *)&a;
    nk_vu8x16_t b_u8x16 = *(nk_vu8x16_t *)&b;
    nk_vu8x16_t xor_u8x16 = vec_xor(a_u8x16, b_u8x16);

    // Step 2: Word popcount → each u32 lane contains set bits for 4 bytes
    nk_vu32x4_t popcnt_u32x4 = vec_popcnt((nk_vu32x4_t)xor_u8x16);

    // Step 3: Vector accumulation (defers horizontal sum to finalize)
    state->intersection_count_u32x4 = vec_add(state->intersection_count_u32x4, popcnt_u32x4);
}

NK_INTERNAL void nk_hamming_u1x128_finalize_powervsx( //
    nk_hamming_u1x128_state_powervsx_t const *state_a, nk_hamming_u1x128_state_powervsx_t const *state_b,
    nk_hamming_u1x128_state_powervsx_t const *state_c, nk_hamming_u1x128_state_powervsx_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);

    nk_vu32x4_t a_u32x4 = state_a->intersection_count_u32x4, b_u32x4 = state_b->intersection_count_u32x4,
                c_u32x4 = state_c->intersection_count_u32x4, d_u32x4 = state_d->intersection_count_u32x4;
    nk_vu32x4_t transpose_ab_low_u32x4 = vec_mergeh(a_u32x4, b_u32x4);
    nk_vu32x4_t transpose_cd_low_u32x4 = vec_mergeh(c_u32x4, d_u32x4);
    nk_vu32x4_t transpose_ab_high_u32x4 = vec_mergel(a_u32x4, b_u32x4);
    nk_vu32x4_t transpose_cd_high_u32x4 = vec_mergel(c_u32x4, d_u32x4);
    nk_vu32x4_t sum_lane0_u32x4 = (nk_vu32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_low_u32x4,
                                                            (nk_vu64x2_t)transpose_cd_low_u32x4, 0);
    nk_vu32x4_t sum_lane1_u32x4 = (nk_vu32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_low_u32x4,
                                                            (nk_vu64x2_t)transpose_cd_low_u32x4, 3);
    nk_vu32x4_t sum_lane2_u32x4 = (nk_vu32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_high_u32x4,
                                                            (nk_vu64x2_t)transpose_cd_high_u32x4, 0);
    nk_vu32x4_t sum_lane3_u32x4 = (nk_vu32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_high_u32x4,
                                                            (nk_vu64x2_t)transpose_cd_high_u32x4, 3);
    result->vu32x4 = vec_add(vec_add(sum_lane0_u32x4, sum_lane1_u32x4), vec_add(sum_lane2_u32x4, sum_lane3_u32x4));
}

typedef struct nk_jaccard_u1x128_state_powervsx_t {
    nk_vu32x4_t intersection_count_u32x4;
} nk_jaccard_u1x128_state_powervsx_t;

NK_INTERNAL void nk_jaccard_u1x128_init_powervsx(nk_jaccard_u1x128_state_powervsx_t *state) {
    state->intersection_count_u32x4 = vec_splats((nk_u32_t)0);
}

NK_INTERNAL void nk_jaccard_u1x128_update_powervsx(nk_jaccard_u1x128_state_powervsx_t *state, nk_b128_vec_t a,
                                                   nk_b128_vec_t b, nk_size_t depth_offset,
                                                   nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);

    // Process one 128-bit chunk (native VSX register size).
    // Uses vector accumulation → horizontal sum deferred to finalize.
    //
    // Power9 VSX instruction characteristics:
    // - `vec_and`:     xxland (V, V, V)              1cy, bitwise AND
    // - `vec_popcnt`:  vpopcntw (V.4S, V.4S)         3cy, word popcount
    // - `vec_add`:     vadduwm (V.4S, V.4S, V.4S)    2cy, u32 add
    // Total: ~6cy per 128-bit chunk (horizontal sum deferred to finalize)

    // Step 1: Compute intersection bits (A AND B)
    nk_vu8x16_t a_u8x16 = *(nk_vu8x16_t *)&a;
    nk_vu8x16_t b_u8x16 = *(nk_vu8x16_t *)&b;
    nk_vu8x16_t intersection_u8x16 = vec_and(a_u8x16, b_u8x16);

    // Step 2: Word popcount → each u32 lane contains set bits for 4 bytes
    nk_vu32x4_t popcnt_u32x4 = vec_popcnt((nk_vu32x4_t)intersection_u8x16);

    // Step 3: Vector accumulation (defers horizontal sum to finalize)
    state->intersection_count_u32x4 = vec_add(state->intersection_count_u32x4, popcnt_u32x4);
}

NK_INTERNAL void nk_jaccard_u1x128_finalize_powervsx( //
    nk_jaccard_u1x128_state_powervsx_t const *state_a, nk_jaccard_u1x128_state_powervsx_t const *state_b,
    nk_jaccard_u1x128_state_powervsx_t const *state_c, nk_jaccard_u1x128_state_powervsx_t const *state_d,
    nk_f32_t query_popcount, nk_b128_vec_t const *target_popcounts_vec, nk_size_t total_dimensions,
    nk_b128_vec_t *result_vec) {
    nk_unused_(total_dimensions);

    // Transpose-based 4-way horizontal sum of u32x4 intersection counts
    nk_vu32x4_t a_u32x4 = state_a->intersection_count_u32x4, b_u32x4 = state_b->intersection_count_u32x4,
                c_u32x4 = state_c->intersection_count_u32x4, d_u32x4 = state_d->intersection_count_u32x4;
    nk_vu32x4_t transpose_ab_low_u32x4 = vec_mergeh(a_u32x4, b_u32x4);
    nk_vu32x4_t transpose_cd_low_u32x4 = vec_mergeh(c_u32x4, d_u32x4);
    nk_vu32x4_t transpose_ab_high_u32x4 = vec_mergel(a_u32x4, b_u32x4);
    nk_vu32x4_t transpose_cd_high_u32x4 = vec_mergel(c_u32x4, d_u32x4);
    nk_vu32x4_t sum_lane0_u32x4 = (nk_vu32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_low_u32x4,
                                                            (nk_vu64x2_t)transpose_cd_low_u32x4, 0);
    nk_vu32x4_t sum_lane1_u32x4 = (nk_vu32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_low_u32x4,
                                                            (nk_vu64x2_t)transpose_cd_low_u32x4, 3);
    nk_vu32x4_t sum_lane2_u32x4 = (nk_vu32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_high_u32x4,
                                                            (nk_vu64x2_t)transpose_cd_high_u32x4, 0);
    nk_vu32x4_t sum_lane3_u32x4 = (nk_vu32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_high_u32x4,
                                                            (nk_vu64x2_t)transpose_cd_high_u32x4, 3);
    nk_vu32x4_t intersection_u32x4 = vec_add(vec_add(sum_lane0_u32x4, sum_lane1_u32x4),
                                             vec_add(sum_lane2_u32x4, sum_lane3_u32x4));
    nk_vf32x4_t intersection_f32x4 = vec_ctf(intersection_u32x4, 0);

    nk_vf32x4_t targets_f32x4 = target_popcounts_vec->vf32x4;
    nk_vf32x4_t query_f32x4 = vec_splats(query_popcount);

    // Compute union using |A union B| = |A| + |B| - |A intersection B|
    nk_vf32x4_t union_f32x4 = vec_sub(vec_add(query_f32x4, targets_f32x4), intersection_f32x4);

    // Handle zero-union edge case (empty vectors → distance = 0.0)
    nk_vf32x4_t one_f32x4 = vec_splats(1.0f);
    nk_vf32x4_t zero_f32x4 = vec_splats(0.0f);
    nk_vu32x4_t zero_union_mask_u32x4 = (nk_vu32x4_t)vec_cmpeq(union_f32x4, zero_f32x4);
    nk_vf32x4_t safe_union_f32x4 = vec_sel(union_f32x4, one_f32x4, zero_union_mask_u32x4);

    // Fast reciprocal with Newton-Raphson refinement
    nk_vf32x4_t union_reciprocal_f32x4 = vec_re(safe_union_f32x4);
    // One Newton-Raphson step: reciprocal = reciprocal × (2 - value * reciprocal)
    nk_vf32x4_t two_f32x4 = vec_splats(2.0f);
    union_reciprocal_f32x4 = vec_mul(union_reciprocal_f32x4,
                                     vec_sub(two_f32x4, vec_mul(safe_union_f32x4, union_reciprocal_f32x4)));

    // Compute Jaccard distance = 1 - intersection / union
    nk_vf32x4_t ratio_f32x4 = vec_mul(intersection_f32x4, union_reciprocal_f32x4);
    nk_vf32x4_t jaccard_f32x4 = vec_sub(one_f32x4, ratio_f32x4);
    result_vec->vf32x4 = vec_sel(jaccard_f32x4, zero_f32x4, zero_union_mask_u32x4);
}

/** @brief Hamming from_dot: computes pop_a + pop_b - 2 × dot for 4 pairs (Power VSX). */
NK_INTERNAL void nk_hamming_u32x4_from_dot_powervsx_(nk_b128_vec_t const *dots_vec, nk_u32_t query_pop,
                                                     nk_b128_vec_t const *target_pops_vec, nk_b128_vec_t *result_vec) {
    nk_vu32x4_t dots_u32x4 = dots_vec->vu32x4;
    nk_vu32x4_t query_u32x4 = vec_splats(query_pop);
    nk_vu32x4_t target_u32x4 = target_pops_vec->vu32x4;
    nk_vu32x4_t two_dots_u32x4 = vec_add(dots_u32x4, dots_u32x4);
    result_vec->vu32x4 = vec_sub(vec_add(query_u32x4, target_u32x4), two_dots_u32x4);
}

/** @brief Jaccard from_dot: computes 1 - dot / (pop_a + pop_b - dot) for 4 pairs (Power VSX). */
NK_INTERNAL void nk_jaccard_f32x4_from_dot_powervsx_(nk_b128_vec_t const *dots_vec, nk_u32_t query_pop,
                                                     nk_b128_vec_t const *target_pops_vec, nk_b128_vec_t *result_vec) {
    nk_vf32x4_t dot_f32x4 = vec_ctf(dots_vec->vu32x4, 0);
    nk_vf32x4_t query_f32x4 = vec_splats((nk_f32_t)query_pop);
    nk_vf32x4_t target_f32x4 = vec_ctf(target_pops_vec->vu32x4, 0);
    nk_vf32x4_t union_f32x4 = vec_sub(vec_add(query_f32x4, target_f32x4), dot_f32x4);

    nk_vf32x4_t one_f32x4 = vec_splats(1.0f);
    nk_vf32x4_t zero_f32x4 = vec_splats(0.0f);
    nk_vu32x4_t zero_union_mask_u32x4 = (nk_vu32x4_t)vec_cmpeq(union_f32x4, zero_f32x4);
    nk_vf32x4_t safe_union_f32x4 = vec_sel(union_f32x4, one_f32x4, zero_union_mask_u32x4);

    // Fast reciprocal with Newton-Raphson
    nk_vf32x4_t union_reciprocal_f32x4 = vec_re(safe_union_f32x4);
    nk_vf32x4_t two_f32x4 = vec_splats(2.0f);
    union_reciprocal_f32x4 = vec_mul(union_reciprocal_f32x4,
                                     vec_sub(two_f32x4, vec_mul(safe_union_f32x4, union_reciprocal_f32x4)));

    nk_vf32x4_t ratio_f32x4 = vec_mul(dot_f32x4, union_reciprocal_f32x4);
    nk_vf32x4_t jaccard_f32x4 = vec_sub(one_f32x4, ratio_f32x4);
    result_vec->vf32x4 = vec_sel(jaccard_f32x4, zero_f32x4, zero_union_mask_u32x4);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_POWERVSX
#endif // NK_TARGET_POWER64_
#endif // NK_SET_POWERVSX_H
