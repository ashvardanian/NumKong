/**
 *  @file include/numkong/set/serial.h
 *  @author Ash Vardanian
 *  @date March 23, 2023
 *  @brief SWAR-accelerated set similarity measures for SIMD-free CPUs.
 *
 *  @sa include/numkong/set.h
 *
 *  @section set_serial_instructions Key SWAR Set Instructions
 *
 *  Serial backend uses lookup-table-based popcount for bit operations.
 *  No SIMD instructions required - works on any architecture.
 */
#ifndef NUMKONG_SET_SERIAL_H
#define NUMKONG_SET_SERIAL_H
#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/*  GCC inlines a helper only into callers whose targets include its own, so serial code builds at
 *  the Armv8-A floor. */
#if defined(__GNUC__) && !defined(__clang__) && NUMKONG_ARCH_ARM64_
#pragma GCC push_options
#pragma GCC target("arch=armv8-a")
#endif

#if NUMKONG_TARGET_SERIAL

/*  Keep the serial instantiations below actually scalar, regardless of build type.
 *  See dots/serial.h for rationale. */
#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

#pragma region Binary Sets

NUMKONG_API nk_status_t nk_hamming_u1_serial(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                             nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_dims_(n, nk_u1_k);
    nk_size_t n_bytes = n / NUMKONG_BITS_PER_BYTE;
    nk_u32_t differences = 0;
    for (nk_size_t i = 0; i != n_bytes; ++i) differences += nk_u1x8_popcount_(a[i] ^ b[i]);
    *result = differences;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_jaccard_u1_serial(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                             nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_dims_(n, nk_u1_k);
    nk_size_t n_bytes = n / NUMKONG_BITS_PER_BYTE;
    nk_u32_t intersection_count = 0, union_count = 0;
    for (nk_size_t i = 0; i != n_bytes; ++i)
        intersection_count += nk_u1x8_popcount_(a[i] & b[i]), union_count += nk_u1x8_popcount_(a[i] | b[i]);
    *result = (union_count != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)union_count : 0.0f;
    return nk_success_k;
}

#pragma endregion Binary Sets

#pragma region Integer Sets

NUMKONG_API nk_status_t nk_jaccard_u32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u32_t intersection_count = 0;
    for (nk_size_t i = 0; i != n; ++i) intersection_count += (a[i] == b[i]);
    *result = (n != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)n : 0.0f;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hamming_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                             nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u32_t differences = 0;
    for (nk_size_t i = 0; i != n; ++i) differences += (a[i] != b[i]);
    *result = differences;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_jaccard_u16_serial(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result,
                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u32_t matches = 0;
    for (nk_size_t i = 0; i != n; ++i) matches += (a[i] == b[i]);
    *result = (n != 0) ? 1.0f - (nk_f32_t)matches / (nk_f32_t)n : 0.0f;
    return nk_success_k;
}

#pragma endregion Integer Sets

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NUMKONG_TARGET_SERIAL

#pragma region Distances from Dot Products

/** Hamming from_dot: computes pop_a + pop_b - 2*dot for 4 pairs (serial). */
NUMKONG_INLINE void nk_hamming_u32x4_from_dot_serial_(nk_b128_vec_t const *dots_vec, nk_u32_t query_pop,
                                                      nk_b128_vec_t const *target_pops_vec, nk_b128_vec_t *result_vec) {
    for (int i = 0; i < 4; ++i) result_vec->u32s[i] = query_pop + target_pops_vec->u32s[i] - 2 * dots_vec->u32s[i];
}

/** Jaccard from_dot: computes 1 - dot / (pop_a + pop_b - dot) for 4 pairs (serial). */
NUMKONG_INLINE void nk_jaccard_f32x4_from_dot_serial_(nk_b128_vec_t const *dots_vec, nk_u32_t query_pop,
                                                      nk_b128_vec_t const *target_pops_vec, nk_b128_vec_t *result_vec) {
    for (int i = 0; i < 4; ++i) {
        nk_f32_t dot = (nk_f32_t)dots_vec->u32s[i];
        nk_f32_t union_val = (nk_f32_t)query_pop + (nk_f32_t)target_pops_vec->u32s[i] - dot;
        result_vec->f32s[i] = (union_val != 0) ? 1.0f - dot / union_val : 0.0f;
    }
}

#pragma endregion Distances from Dot Products

#if defined(__GNUC__) && !defined(__clang__) && NUMKONG_ARCH_ARM64_
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_SET_SERIAL_H
