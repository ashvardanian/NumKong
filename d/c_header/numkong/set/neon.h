/**
 *  @brief SIMD-accelerated Set Similarity Measures for NEON.
 *  @file include/numkong/set/neon.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/set.h
 *
 *  @section set_neon_instructions NEON Set Instructions
 *
 *  Key NEON instructions for binary/bitwise operations (Cortex-A76 class):
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *      vcntq_u8                    CNT (V.16B, V.16B)              2cy         2/cy
 *      veorq_u8                    EOR (V.16B, V.16B, V.16B)       1cy         4/cy
 *      vandq_u8                    AND (V.16B, V.16B, V.16B)       1cy         4/cy
 *      vorrq_u8                    ORR (V.16B, V.16B, V.16B)       1cy         4/cy
 *      vpaddlq_u8                  UADDLP (V.8H, V.16B)            2cy         2/cy
 *      vaddvq_u32                  ADDV (S, V.4S)                  3cy         1/cy
 *
 *  According to the available literature, the throughput for those basic integer ops is
 *  identical across most Apple, Qualcomm, and AWS Graviton chips. As long as we avoid widening
 *  operations and horizontal reductions, we won't face any reasonable bottlenecks.
 *
 *  @section set_neon_stateful Stateful Streaming Logic
 *
 *  To build memory-optimal tiled algorithms, this file defines:
 *
 *  - nk_hamming_u1x128_state_neon_t for streaming Hamming distance
 *  - nk_jaccard_u1x128_state_neon_t for streaming Jaccard similarity
 *
 *  @code{c}
 *  nk_jaccard_u1x128_state_neon_t state_first, state_second, state_third, state_fourth;
 *  nk_jaccard_u1x128_init_neon(&state_first);
 *  // ... stream through packed binary vectors ...
 *  nk_jaccard_u1x128_finalize_neon(&state_first, &state_second, &state_third, &state_fourth,
 *      query_popcount, target_popcount_a, target_popcount_b, target_popcount_c, target_popcount_d,
 *      total_dimensions, &results);
 *  @endcode
 */
#ifndef NK_SET_NEON_H
#define NK_SET_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/types.h"      // `nk_u1x8_t`
#include "numkong/set/serial.h" // `nk_u1x8_popcount_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

#pragma region - Binary Sets

NK_PUBLIC void nk_hamming_u1_neon(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);
    nk_u32_t differences = 0;
    nk_size_t i = 0;
    // In each 8-bit word we may have up to 8 differences.
    // So for up-to 31 cycles (31 * 16 = 496 word-dimensions = 3968 bits)
    // we can aggregate the differences into a `uint8x16_t` vector,
    // where each component will be up-to 255.
    while (i + 16 <= n_bytes) {
        uint8x16_t popcount_u8x16 = vdupq_n_u8(0);
        for (nk_size_t cycle = 0; cycle < 31 && i + 16 <= n_bytes; ++cycle, i += 16) {
            uint8x16_t a_u8x16 = vld1q_u8(a + i);
            uint8x16_t b_u8x16 = vld1q_u8(b + i);
            uint8x16_t xor_popcount_u8x16 = vcntq_u8(veorq_u8(a_u8x16, b_u8x16));
            popcount_u8x16 = vaddq_u8(popcount_u8x16, xor_popcount_u8x16);
        }
        differences += (nk_u32_t)vaddlvq_u8(popcount_u8x16);
    }
    // Handle the tail
    for (; i != n_bytes; ++i) differences += nk_u1x8_popcount_(a[i] ^ b[i]);
    *result = differences;
}

NK_PUBLIC void nk_jaccard_u1_neon(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);
    nk_u32_t intersection_count = 0, union_count = 0;
    nk_size_t i = 0;
    // In each 8-bit word we may have up to 8 intersections/unions.
    // So for up-to 31 cycles (31 * 16 = 496 word-dimensions = 3968 bits)
    // we can aggregate the intersections/unions into a `uint8x16_t` vector,
    // where each component will be up-to 255.
    while (i + 16 <= n_bytes) {
        uint8x16_t intersection_popcount_u8x16 = vdupq_n_u8(0);
        uint8x16_t union_popcount_u8x16 = vdupq_n_u8(0);
        for (nk_size_t cycle = 0; cycle < 31 && i + 16 <= n_bytes; ++cycle, i += 16) {
            uint8x16_t a_u8x16 = vld1q_u8(a + i);
            uint8x16_t b_u8x16 = vld1q_u8(b + i);
            intersection_popcount_u8x16 = vaddq_u8(intersection_popcount_u8x16, vcntq_u8(vandq_u8(a_u8x16, b_u8x16)));
            union_popcount_u8x16 = vaddq_u8(union_popcount_u8x16, vcntq_u8(vorrq_u8(a_u8x16, b_u8x16)));
        }
        intersection_count += (nk_u32_t)vaddlvq_u8(intersection_popcount_u8x16);
        union_count += (nk_u32_t)vaddlvq_u8(union_popcount_u8x16);
    }
    // Handle the tail
    for (; i != n_bytes; ++i)
        intersection_count += nk_u1x8_popcount_(a[i] & b[i]), union_count += nk_u1x8_popcount_(a[i] | b[i]);
    *result = (union_count != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)union_count : 0.0f;
}

#pragma endregion - Binary Sets

#pragma region - Integer Sets

NK_PUBLIC void nk_jaccard_u32_neon(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t intersection_count = 0;
    nk_size_t i = 0;
    uint32x4_t intersection_count_u32x4 = vdupq_n_u32(0);
    for (; i + 4 <= n; i += 4) {
        uint32x4_t a_u32x4 = vld1q_u32(a + i);
        uint32x4_t b_u32x4 = vld1q_u32(b + i);
        uint32x4_t equality_mask = vceqq_u32(a_u32x4, b_u32x4);
        intersection_count_u32x4 = vaddq_u32(intersection_count_u32x4, vshrq_n_u32(equality_mask, 31));
    }
    intersection_count += vaddvq_u32(intersection_count_u32x4);
    for (; i != n; ++i) intersection_count += (a[i] == b[i]);
    *result = (n != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)n : 0.0f;
}

NK_PUBLIC void nk_hamming_u8_neon(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_size_t i = 0;
    uint32x4_t diff_count_u32x4 = vdupq_n_u32(0);
    // Process 16 bytes at a time using NEON with widening adds to avoid overflow.
    // Uses pairwise widening chain: 16 u8 → 8 u16 → 4 u32 per iteration.
    for (; i + 16 <= n; i += 16) {
        uint8x16_t a_u8x16 = vld1q_u8(a + i);
        uint8x16_t b_u8x16 = vld1q_u8(b + i);
        // vceqq_u8 returns 0xFF for equal, 0x00 for not-equal
        // Invert to get 0xFF for not-equal, then shift right by 7 to get 1
        uint8x16_t not_equal_u8x16 = vmvnq_u8(vceqq_u8(a_u8x16, b_u8x16));
        uint8x16_t diff_bits_u8x16 = vshrq_n_u8(not_equal_u8x16, 7);
        // Widen: 16 u8 → 8 u16 → 4 u32 using pairwise add and widen
        uint16x8_t diff_u16x8 = vpaddlq_u8(diff_bits_u8x16);
        uint32x4_t diff_u32x4 = vpaddlq_u16(diff_u16x8);
        diff_count_u32x4 = vaddq_u32(diff_count_u32x4, diff_u32x4);
    }
    nk_u32_t differences = vaddvq_u32(diff_count_u32x4);
    // Handle tail elements
    for (; i != n; ++i) differences += (a[i] != b[i]);
    *result = differences;
}

NK_PUBLIC void nk_jaccard_u16_neon(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t matches = 0;
    nk_size_t i = 0;
    uint32x4_t match_count_u32x4 = vdupq_n_u32(0);
    // Process 8 u16 values at a time using NEON
    for (; i + 8 <= n; i += 8) {
        uint16x8_t a_u16x8 = vld1q_u16(a + i);
        uint16x8_t b_u16x8 = vld1q_u16(b + i);
        // vceqq_u16 returns 0xFFFF for equal, 0x0000 for not-equal
        uint16x8_t equality_mask = vceqq_u16(a_u16x8, b_u16x8);
        // Count matches by shifting right by 15 to get 1 for match, 0 for non-match
        // Then widen and accumulate into u32
        uint16x8_t match_bits = vshrq_n_u16(equality_mask, 15);
        // Pairwise add and widen to u32
        uint32x4_t match_u32x4 = vpaddlq_u16(match_bits);
        match_count_u32x4 = vaddq_u32(match_count_u32x4, match_u32x4);
    }
    matches += vaddvq_u32(match_count_u32x4);
    // Handle tail elements
    for (; i != n; ++i) matches += (a[i] == b[i]);
    *result = (n != 0) ? 1.0f - (nk_f32_t)matches / (nk_f32_t)n : 0.0f;
}

#pragma endregion - Integer Sets

#pragma region - Stateful Streaming

typedef struct nk_hamming_u1x128_state_neon_t {
    uint32x4_t intersection_count_u32x4;
} nk_hamming_u1x128_state_neon_t;

NK_INTERNAL void nk_hamming_u1x128_init_neon(nk_hamming_u1x128_state_neon_t *state) {
    state->intersection_count_u32x4 = vdupq_n_u32(0);
}

NK_INTERNAL void nk_hamming_u1x128_update_neon(nk_hamming_u1x128_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                               nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);

    // Process one 128-bit chunk (native ARM NEON register size).
    // Uses vector accumulation - horizontal sum deferred to finalize.
    //
    // ARM NEON instruction characteristics:
    // - `veorq_u8`:    XOR (V.16B, V.16B, V.16B)       1cy
    // - `vcntq_u8`:    CNT (V.16B, V.16B)              1-2cy, byte popcount
    // - `vpaddlq_u8`:  UADDLP (V.8H, V.16B)            1cy, pairwise widen u8 → u16
    // - `vpaddlq_u16`: UADDLP (V.4S, V.8H)             1cy, pairwise widen u16 → u32
    // - `vaddq_u32`:   ADD (V.4S, V.4S, V.4S)          1cy
    // Total: ~5-6cy per 128-bit chunk (horizontal sum deferred to finalize)

    // Step 1: Compute intersection bits (A XOR B)
    uint8x16_t intersection_u8x16 = veorq_u8(a.u8x16, b.u8x16);

    // Step 2: Byte-level popcount - each byte contains count of set bits (0-8)
    uint8x16_t popcount_u8x16 = vcntq_u8(intersection_u8x16);

    // Step 3: Pairwise widening reduction chain
    // u8x16 → u16x8: pairs of adjacent bytes summed into 16-bit
    uint16x8_t popcount_u16x8 = vpaddlq_u8(popcount_u8x16);
    // u16x8 → u32x4: pairs of 16-bit values summed into 32-bit
    uint32x4_t popcount_u32x4 = vpaddlq_u16(popcount_u16x8);

    // Step 4: Vector accumulation (defers horizontal sum to finalize)
    state->intersection_count_u32x4 = vaddq_u32(state->intersection_count_u32x4, popcount_u32x4);
}

NK_INTERNAL void nk_hamming_u1x128_finalize_neon( //
    nk_hamming_u1x128_state_neon_t const *state_a, nk_hamming_u1x128_state_neon_t const *state_b,
    nk_hamming_u1x128_state_neon_t const *state_c, nk_hamming_u1x128_state_neon_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);

    // Horizontal sum using pairwise adds - same pattern as Jaccard.
    // 3× ADDP to reduce 4 state vectors into [sum_a, sum_b, sum_c, sum_d].
    uint32x4_t ab_sum_u32x4 = vpaddq_u32(state_a->intersection_count_u32x4, state_b->intersection_count_u32x4);
    uint32x4_t cd_sum_u32x4 = vpaddq_u32(state_c->intersection_count_u32x4, state_d->intersection_count_u32x4);
    result->u32x4 = vpaddq_u32(ab_sum_u32x4, cd_sum_u32x4);
}

typedef struct nk_jaccard_u1x128_state_neon_t {
    uint32x4_t intersection_count_u32x4;
} nk_jaccard_u1x128_state_neon_t;

NK_INTERNAL void nk_jaccard_u1x128_init_neon(nk_jaccard_u1x128_state_neon_t *state) {
    state->intersection_count_u32x4 = vdupq_n_u32(0);
}

NK_INTERNAL void nk_jaccard_u1x128_update_neon(nk_jaccard_u1x128_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                               nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);

    // Process one 128-bit chunk (native ARM NEON register size).
    // Uses vector accumulation - horizontal sum deferred to finalize.
    //
    // ARM NEON instruction characteristics:
    // - `vandq_u8`:    AND (V.16B, V.16B, V.16B)       1cy
    // - `vcntq_u8`:    CNT (V.16B, V.16B)              1-2cy, byte popcount
    // - `vpaddlq_u8`:  UADDLP (V.8H, V.16B)            1cy, pairwise widen u8 → u16
    // - `vpaddlq_u16`: UADDLP (V.4S, V.8H)             1cy, pairwise widen u16 → u32
    // - `vaddq_u32`:   ADD (V.4S, V.4S, V.4S)          1cy
    // Total: ~5-6cy per 128-bit chunk (horizontal sum deferred to finalize)

    // Step 1: Compute intersection bits (A AND B)
    uint8x16_t intersection_u8x16 = vandq_u8(a.u8x16, b.u8x16);

    // Step 2: Byte-level popcount - each byte contains count of set bits (0-8)
    uint8x16_t popcount_u8x16 = vcntq_u8(intersection_u8x16);

    // Step 3: Pairwise widening reduction chain
    // u8x16 → u16x8: pairs of adjacent bytes summed into 16-bit
    uint16x8_t popcount_u16x8 = vpaddlq_u8(popcount_u8x16);
    // u16x8 → u32x4: pairs of 16-bit values summed into 32-bit
    uint32x4_t popcount_u32x4 = vpaddlq_u16(popcount_u16x8);

    // Step 4: Vector accumulation (defers horizontal sum to finalize)
    state->intersection_count_u32x4 = vaddq_u32(state->intersection_count_u32x4, popcount_u32x4);
}

NK_INTERNAL void nk_jaccard_u1x128_finalize_neon( //
    nk_jaccard_u1x128_state_neon_t const *state_a, nk_jaccard_u1x128_state_neon_t const *state_b,
    nk_jaccard_u1x128_state_neon_t const *state_c, nk_jaccard_u1x128_state_neon_t const *state_d,
    nk_f32_t query_popcount, nk_f32_t target_popcount_a, nk_f32_t target_popcount_b, nk_f32_t target_popcount_c,
    nk_f32_t target_popcount_d, nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);

    // Horizontal sum using pairwise adds instead of `vaddvq_u32` (ADDV).
    // `vpaddq_u32` (ADDP) has better throughput: 2/cy vs 1/cy for ADDV on Cortex-A76.
    // 3 ADDP instructions vs 4 ADDV + union store/load.
    //
    // Step 1: vpaddq_u32(A, B) = [a0+a1, a2+a3, b0+b1, b2+b3]
    uint32x4_t ab_sum_u32x4 = vpaddq_u32(state_a->intersection_count_u32x4, state_b->intersection_count_u32x4);
    uint32x4_t cd_sum_u32x4 = vpaddq_u32(state_c->intersection_count_u32x4, state_d->intersection_count_u32x4);
    // Step 2: Final pairwise reduction gives [sum_a, sum_b, sum_c, sum_d]
    uint32x4_t intersection_u32x4 = vpaddq_u32(ab_sum_u32x4, cd_sum_u32x4);
    float32x4_t intersection_f32x4 = vcvtq_f32_u32(intersection_u32x4);

    // Compute union using |A ∪ B| = |A| + |B| - |A ∩ B|
    // Build target popcounts vector using lane insertion (avoids union store/load round-trip).
    float32x4_t query_f32x4 = vdupq_n_f32(query_popcount);
    float32x4_t targets_f32x4 = vdupq_n_f32(target_popcount_a);
    targets_f32x4 = vsetq_lane_f32(target_popcount_b, targets_f32x4, 1);
    targets_f32x4 = vsetq_lane_f32(target_popcount_c, targets_f32x4, 2);
    targets_f32x4 = vsetq_lane_f32(target_popcount_d, targets_f32x4, 3);
    float32x4_t union_f32x4 = vsubq_f32(vaddq_f32(query_f32x4, targets_f32x4), intersection_f32x4);

    // Handle zero-union edge case (empty vectors → distance = 0.0, matching scipy convention)
    float32x4_t one_f32x4 = vdupq_n_f32(1.0f);
    uint32x4_t zero_union_mask = vceqq_f32(union_f32x4, vdupq_n_f32(0.0f));
    float32x4_t safe_union_f32x4 = vbslq_f32(zero_union_mask, one_f32x4, union_f32x4);

    // Fast reciprocal with Newton-Raphson refinement:
    // - `vrecpeq_f32`: ~12-bit estimate, 1 cycle
    // - `vrecpsq_f32`: Newton-Raphson step computes (2 - a × b), 1 cycle
    // - `vmulq_f32`: multiply, 1 cycle
    // One N-R iteration: ~24-bit accuracy, sufficient for f32 (23 mantissa bits).
    // Total: ~3-4 cycles vs ~10-14 cycles for division.
    float32x4_t union_reciprocal_f32x4 = vrecpeq_f32(safe_union_f32x4);
    union_reciprocal_f32x4 = vmulq_f32(union_reciprocal_f32x4, vrecpsq_f32(safe_union_f32x4, union_reciprocal_f32x4));

    // Compute Jaccard distance = 1 - intersection ÷ union
    float32x4_t ratio_f32x4 = vmulq_f32(intersection_f32x4, union_reciprocal_f32x4);
    float32x4_t jaccard_f32x4 = vsubq_f32(one_f32x4, ratio_f32x4);
    result->f32x4 = vbslq_f32(zero_union_mask, vdupq_n_f32(0.0f), jaccard_f32x4);
}

/** @brief Hamming from_dot: computes pop_a + pop_b - 2*dot for 4 pairs (NEON). */
NK_INTERNAL void nk_hamming_u32x4_from_dot_neon_(nk_b128_vec_t dots, nk_u32_t query_pop, nk_b128_vec_t target_pops,
                                                 nk_b128_vec_t *results) {
    uint32x4_t dots_u32x4 = dots.u32x4;
    uint32x4_t query_u32x4 = vdupq_n_u32(query_pop);
    uint32x4_t target_u32x4 = target_pops.u32x4;
    results->u32x4 = vsubq_u32(vaddq_u32(query_u32x4, target_u32x4), vshlq_n_u32(dots_u32x4, 1));
}

/** @brief Jaccard from_dot: computes 1 - dot / (pop_a + pop_b - dot) for 4 pairs (NEON). */
NK_INTERNAL void nk_jaccard_f32x4_from_dot_neon_(nk_b128_vec_t dots, nk_u32_t query_pop, nk_b128_vec_t target_pops,
                                                 nk_b128_vec_t *results) {
    float32x4_t dot_f32x4 = vcvtq_f32_u32(dots.u32x4);
    float32x4_t query_f32x4 = vdupq_n_f32((nk_f32_t)query_pop);
    float32x4_t target_f32x4 = vcvtq_f32_u32(target_pops.u32x4);
    float32x4_t union_f32x4 = vsubq_f32(vaddq_f32(query_f32x4, target_f32x4), dot_f32x4);

    float32x4_t one_f32x4 = vdupq_n_f32(1.0f);
    uint32x4_t zero_union_mask = vceqq_f32(union_f32x4, vdupq_n_f32(0.0f));
    float32x4_t safe_union_f32x4 = vbslq_f32(zero_union_mask, one_f32x4, union_f32x4);

    float32x4_t union_reciprocal_f32x4 = vrecpeq_f32(safe_union_f32x4);
    union_reciprocal_f32x4 = vmulq_f32(union_reciprocal_f32x4, vrecpsq_f32(safe_union_f32x4, union_reciprocal_f32x4));

    float32x4_t ratio_f32x4 = vmulq_f32(dot_f32x4, union_reciprocal_f32x4);
    float32x4_t jaccard_f32x4 = vsubq_f32(one_f32x4, ratio_f32x4);
    results->f32x4 = vbslq_f32(zero_union_mask, vdupq_n_f32(0.0f), jaccard_f32x4);
}

#pragma endregion - Stateful Streaming

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
#endif // NK_SET_NEON_H
