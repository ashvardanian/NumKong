/**
 *  @file include/numkong/sets/haswell.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *  @brief AVX2 batched set operations for Haswell.
 *
 *  @sa include/numkong/sets.h
 */
#ifndef NUMKONG_SETS_HASWELL_H
#define NUMKONG_SETS_HASWELL_H

#if NUMKONG_ARCH_X86_64_
#if NUMKONG_TARGET_HASWELL

#include "numkong/set/haswell.h"
#include "numkong/dots/haswell.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "popcnt")
#endif

nk_define_cross_normalized_packed_(hamming, u1, haswell, u1x8, u1x8, u32, /*norm_value_type=*/u32, u32, nk_b128_vec_t,
                                   nk_dots_packed_u1_haswell, nk_hamming_u32x4_from_dot_haswell_,
                                   nk_dots_reduce_sum_u1_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, /*dimensions_per_value=*/8)

nk_define_cross_normalized_packed_(jaccard, u1, haswell, u1x8, u1x8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u1_haswell, nk_jaccard_f32x4_from_dot_haswell_,
                                   nk_dots_reduce_sum_u1_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, /*dimensions_per_value=*/8)

nk_define_cross_normalized_symmetric_(hamming, u1, haswell, u1x8, u32, /*norm_value_type=*/u32, u32, nk_b128_vec_t,
                                      nk_dots_symmetric_u1_haswell, nk_hamming_u32x4_from_dot_haswell_,
                                      nk_dots_reduce_sum_u1_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_,
                                      /*dimensions_per_value=*/8)

nk_define_cross_normalized_symmetric_(jaccard, u1, haswell, u1x8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u1_haswell, nk_jaccard_f32x4_from_dot_haswell_,
                                      nk_dots_reduce_sum_u1_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_,
                                      /*dimensions_per_value=*/8)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_HASWELL
#endif // NUMKONG_ARCH_X86_64_
#endif // NUMKONG_SETS_HASWELL_H
