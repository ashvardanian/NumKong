/**
 *  @brief Batched Set Operations for NEON.
 *  @file include/numkong/sets/neon.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/sets.h
 */
#ifndef NK_SETS_NEON_H
#define NK_SETS_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/set/neon.h"
#include "numkong/dots/neon.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

nk_define_cross_normalized_packed_(hamming, u1, neon, u1x8, u1x8, u32, /*norm_value_type=*/u32, u32, nk_b128_vec_t,
                                   nk_dots_packed_u1_neon, nk_hamming_u32x4_from_dot_neon_, nk_dots_reduce_sum_u1_,
                                   nk_load_b128_serial_, nk_partial_load_b32x4_serial_, nk_store_b128_serial_,
                                   nk_partial_store_b32x4_serial_, /*dimensions_per_value=*/8)

nk_define_cross_normalized_packed_(jaccard, u1, neon, u1x8, u1x8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u1_neon, nk_jaccard_f32x4_from_dot_neon_, nk_dots_reduce_sum_u1_,
                                   nk_load_b128_serial_, nk_partial_load_b32x4_serial_, nk_store_b128_serial_,
                                   nk_partial_store_b32x4_serial_, /*dimensions_per_value=*/8)

nk_define_cross_normalized_symmetric_(hamming, u1, neon, u1x8, u32, /*norm_value_type=*/u32, u32, nk_b128_vec_t,
                                      nk_dots_symmetric_u1_neon, nk_hamming_u32x4_from_dot_neon_,
                                      nk_dots_reduce_sum_u1_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, /*dimensions_per_value=*/8)

nk_define_cross_normalized_symmetric_(jaccard, u1, neon, u1x8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u1_neon, nk_jaccard_f32x4_from_dot_neon_,
                                      nk_dots_reduce_sum_u1_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, /*dimensions_per_value=*/8)

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
#endif // NK_SETS_NEON_H
