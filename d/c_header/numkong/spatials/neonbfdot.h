/**
 *  @brief Batched Spatial Distances for NEON BF16 Dot Product.
 *  @file include/numkong/spatials/neonbfdot.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_NEONBFDOT_H
#define NK_SPATIALS_NEONBFDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONBFDOT

#include "numkong/spatial/neon.h"
#include "numkong/dots/neonbfdot.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#endif

nk_define_cross_normalized_packed_(angular, bf16, neonbfdot, bf16, bf16, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_bf16_neonbfdot, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_bf16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, bf16, neonbfdot, bf16, bf16, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_bf16_neonbfdot,
                                   nk_euclidean_through_f32_from_dot_neon_, nk_dots_reduce_sumsq_bf16_,
                                   nk_load_b128_neon_, nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, bf16, neonbfdot, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_neonbfdot, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_bf16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, bf16, neonbfdot, bf16, f32, /*norm_value_type=*/f32, f32,
                                      nk_b128_vec_t, nk_dots_symmetric_bf16_neonbfdot,
                                      nk_euclidean_through_f32_from_dot_neon_, nk_dots_reduce_sumsq_bf16_,
                                      nk_load_b128_neon_, nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONBFDOT
#endif // NK_TARGET_ARM_
#endif // NK_SPATIALS_NEONBFDOT_H
