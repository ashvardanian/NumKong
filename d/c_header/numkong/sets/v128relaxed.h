/**
 *  @brief Batched Set Operations for WASM Relaxed SIMD.
 *  @file include/numkong/sets/v128relaxed.h
 *  @author Ash Vardanian
 *  @date March 10, 2026
 *
 *  @sa include/numkong/sets.h
 */
#ifndef NK_SETS_V128RELAXED_H
#define NK_SETS_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/set/v128relaxed.h"
#include "numkong/dots/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

nk_define_cross_normalized_packed_(hamming, u1, v128relaxed, u1x8, u1x8, u32, /*norm_value_type=*/u32, u32,
                                   nk_b128_vec_t, nk_dots_packed_u1_v128relaxed, nk_hamming_u32x4_from_dot_v128relaxed_,
                                   nk_dots_reduce_sum_u1_, nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_,
                                   /*dimensions_per_value=*/8)

nk_define_cross_normalized_packed_(jaccard, u1, v128relaxed, u1x8, u1x8, u32, /*norm_value_type=*/u32, f32,
                                   nk_b128_vec_t, nk_dots_packed_u1_v128relaxed, nk_jaccard_f32x4_from_dot_v128relaxed_,
                                   nk_dots_reduce_sum_u1_, nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_,
                                   /*dimensions_per_value=*/8)

nk_define_cross_normalized_symmetric_(hamming, u1, v128relaxed, u1x8, u32, /*norm_value_type=*/u32, u32, nk_b128_vec_t,
                                      nk_dots_symmetric_u1_v128relaxed, nk_hamming_u32x4_from_dot_v128relaxed_,
                                      nk_dots_reduce_sum_u1_, nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_,
                                      /*dimensions_per_value=*/8)

nk_define_cross_normalized_symmetric_(jaccard, u1, v128relaxed, u1x8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u1_v128relaxed, nk_jaccard_f32x4_from_dot_v128relaxed_,
                                      nk_dots_reduce_sum_u1_, nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_,
                                      /*dimensions_per_value=*/8)

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_SETS_V128RELAXED_H
