/**
 *  @brief SWAR-accelerated Set Similarity Measures for SIMD-free CPUs.
 *  @file include/numkong/set/serial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/set.h
 *
 *  @section set_serial_instructions Key SWAR Set Instructions
 *
 *  Serial backend uses lookup-table-based popcount for bit operations.
 *  No SIMD instructions required - works on any architecture.
 *
 *  @section set_serial_stateful Stateful Streaming Logic
 *
 *  To build memory-optimal tiled algorithms, this file defines:
 *
 *  - nk_hamming_u1x128_state_serial_t for streaming Hamming distance
 *  - nk_jaccard_u1x128_state_serial_t for streaming Jaccard similarity
 *
 *  @code{c}
 *  nk_jaccard_u1x128_state_serial_t state_first, state_second, state_third, state_fourth;
 *  nk_jaccard_u1x128_init_serial(&state_first);
 *  // ... stream through packed binary vectors ...
 *  nk_jaccard_u1x128_finalize_serial(&state_first, &state_second, &state_third, &state_fourth,
 *      query_popcount, &target_popcounts_vec, total_dimensions, &result_vec);
 *  @endcode
 */
#ifndef NK_SET_SERIAL_H
#define NK_SET_SERIAL_H
#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Binary Sets

NK_PUBLIC void nk_hamming_u1_serial(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);
    nk_u32_t differences = 0;
    for (nk_size_t i = 0; i != n_bytes; ++i) differences += nk_u1x8_popcount_(a[i] ^ b[i]);
    *result = differences;
}

NK_PUBLIC void nk_jaccard_u1_serial(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);
    nk_u32_t intersection_count = 0, union_count = 0;
    for (nk_size_t i = 0; i != n_bytes; ++i)
        intersection_count += nk_u1x8_popcount_(a[i] & b[i]), union_count += nk_u1x8_popcount_(a[i] | b[i]);
    *result = (union_count != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)union_count : 0.0f;
}

#pragma endregion Binary Sets

#pragma region Integer Sets

NK_PUBLIC void nk_jaccard_u32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t intersection_count = 0;
    for (nk_size_t i = 0; i != n; ++i) intersection_count += (a[i] == b[i]);
    *result = (n != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)n : 0.0f;
}

NK_PUBLIC void nk_hamming_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_u32_t differences = 0;
    for (nk_size_t i = 0; i != n; ++i) differences += (a[i] != b[i]);
    *result = differences;
}

NK_PUBLIC void nk_jaccard_u16_serial(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t matches = 0;
    for (nk_size_t i = 0; i != n; ++i) matches += (a[i] == b[i]);
    *result = (n != 0) ? 1.0f - (nk_f32_t)matches / (nk_f32_t)n : 0.0f;
}

#pragma endregion Integer Sets

#pragma region Stateful Streaming

typedef struct nk_jaccard_u1x128_state_serial_t {
    nk_u64_t intersection_count;
} nk_jaccard_u1x128_state_serial_t;

NK_INTERNAL void nk_jaccard_u1x128_init_serial(nk_jaccard_u1x128_state_serial_t *state) {
    state->intersection_count = 0;
}

NK_INTERNAL void nk_jaccard_u1x128_update_serial(nk_jaccard_u1x128_state_serial_t *state, nk_b128_vec_t a,
                                                 nk_b128_vec_t b, nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    nk_u64_t intersection_low = a.u64s[0] & b.u64s[0];
    nk_u64_t intersection_high = a.u64s[1] & b.u64s[1];
    state->intersection_count += nk_u64_popcount_(intersection_low);
    state->intersection_count += nk_u64_popcount_(intersection_high);
}

NK_INTERNAL void nk_jaccard_u1x128_finalize_serial( //
    nk_jaccard_u1x128_state_serial_t const *state_a, nk_jaccard_u1x128_state_serial_t const *state_b,
    nk_jaccard_u1x128_state_serial_t const *state_c, nk_jaccard_u1x128_state_serial_t const *state_d,
    nk_f32_t query_popcount, nk_b128_vec_t const *target_popcounts_vec, nk_size_t total_dimensions,
    nk_b128_vec_t *result_vec) {
    nk_unused_(total_dimensions);

    nk_f32_t intersection_a = (nk_f32_t)state_a->intersection_count;
    nk_f32_t intersection_b = (nk_f32_t)state_b->intersection_count;
    nk_f32_t intersection_c = (nk_f32_t)state_c->intersection_count;
    nk_f32_t intersection_d = (nk_f32_t)state_d->intersection_count;

    nk_f32_t union_a = query_popcount + target_popcounts_vec->f32s[0] - intersection_a;
    nk_f32_t union_b = query_popcount + target_popcounts_vec->f32s[1] - intersection_b;
    nk_f32_t union_c = query_popcount + target_popcounts_vec->f32s[2] - intersection_c;
    nk_f32_t union_d = query_popcount + target_popcounts_vec->f32s[3] - intersection_d;

    result_vec->f32s[0] = (union_a != 0) ? 1.0f - intersection_a / union_a : 0.0f;
    result_vec->f32s[1] = (union_b != 0) ? 1.0f - intersection_b / union_b : 0.0f;
    result_vec->f32s[2] = (union_c != 0) ? 1.0f - intersection_c / union_c : 0.0f;
    result_vec->f32s[3] = (union_d != 0) ? 1.0f - intersection_d / union_d : 0.0f;
}

typedef struct nk_hamming_u1x128_state_serial_t {
    nk_u64_t intersection_count;
} nk_hamming_u1x128_state_serial_t;

NK_INTERNAL void nk_hamming_u1x128_init_serial(nk_hamming_u1x128_state_serial_t *state) {
    state->intersection_count = 0;
}

NK_INTERNAL void nk_hamming_u1x128_update_serial(nk_hamming_u1x128_state_serial_t *state, nk_b128_vec_t a,
                                                 nk_b128_vec_t b, nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    nk_u64_t intersection_low = a.u64s[0] ^ b.u64s[0];
    nk_u64_t intersection_high = a.u64s[1] ^ b.u64s[1];
    state->intersection_count += nk_u64_popcount_(intersection_low);
    state->intersection_count += nk_u64_popcount_(intersection_high);
}

NK_INTERNAL void nk_hamming_u1x128_finalize_serial( //
    nk_hamming_u1x128_state_serial_t const *state_a, nk_hamming_u1x128_state_serial_t const *state_b,
    nk_hamming_u1x128_state_serial_t const *state_c, nk_hamming_u1x128_state_serial_t const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);

    result->u32s[0] = (nk_u32_t)state_a->intersection_count;
    result->u32s[1] = (nk_u32_t)state_b->intersection_count;
    result->u32s[2] = (nk_u32_t)state_c->intersection_count;
    result->u32s[3] = (nk_u32_t)state_d->intersection_count;
}

/** @brief Hamming from_dot: computes pop_a + pop_b - 2*dot for 4 pairs (serial). */
NK_INTERNAL void nk_hamming_u32x4_from_dot_serial_(nk_b128_vec_t const *dots_vec, nk_u32_t query_pop,
                                                   nk_b128_vec_t const *target_pops_vec, nk_b128_vec_t *result_vec) {
    for (int i = 0; i < 4; ++i) result_vec->u32s[i] = query_pop + target_pops_vec->u32s[i] - 2 * dots_vec->u32s[i];
}

/** @brief Jaccard from_dot: computes 1 - dot / (pop_a + pop_b - dot) for 4 pairs (serial). */
NK_INTERNAL void nk_jaccard_f32x4_from_dot_serial_(nk_b128_vec_t const *dots_vec, nk_u32_t query_pop,
                                                   nk_b128_vec_t const *target_pops_vec, nk_b128_vec_t *result_vec) {
    for (int i = 0; i < 4; ++i) {
        nk_f32_t dot = (nk_f32_t)dots_vec->u32s[i];
        nk_f32_t union_val = (nk_f32_t)query_pop + (nk_f32_t)target_pops_vec->u32s[i] - dot;
        result_vec->f32s[i] = (union_val != 0) ? 1.0f - dot / union_val : 0.0f;
    }
}

#pragma endregion Stateful Streaming

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_SET_SERIAL_H
