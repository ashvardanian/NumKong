/**
 *  @brief Batched Spatial Distances for NEON Signed/Unsigned Dot Product.
 *  @file include/numkong/spatials/neonsdot.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_NEONSDOT_H
#define NK_SPATIALS_NEONSDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONSDOT

#include "numkong/spatial/neon.h"
#include "numkong/dots/neonsdot.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod")
#endif

nk_define_cross_normalized_packed_(angular, i8, neonsdot, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_neonsdot, nk_angular_through_i32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, i8, neonsdot, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_neonsdot, nk_euclidean_through_i32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, i8, neonsdot, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_neonsdot, nk_angular_through_i32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, i8, neonsdot, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_neonsdot, nk_euclidean_through_i32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, u8, neonsdot, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_neonsdot, nk_angular_through_u32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, u8, neonsdot, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_neonsdot, nk_euclidean_through_u32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, u8, neonsdot, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_neonsdot, nk_angular_through_u32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, u8, neonsdot, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_neonsdot, nk_euclidean_through_u32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, i4, neonsdot, i4x2, i4x2, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i4_neonsdot, nk_angular_through_i32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_i4_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_packed_(euclidean, i4, neonsdot, i4x2, i4x2, i32, /*norm_value_type=*/u32, f32,
                                   nk_b128_vec_t, nk_dots_packed_i4_neonsdot, nk_euclidean_through_i32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_i4_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_symmetric_(angular, i4, neonsdot, i4x2, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i4_neonsdot, nk_angular_through_i32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_i4_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_symmetric_(euclidean, i4, neonsdot, i4x2, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i4_neonsdot, nk_euclidean_through_i32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_i4_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)

nk_define_cross_normalized_packed_(angular, u4, neonsdot, u4x2, u4x2, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u4_neonsdot, nk_angular_through_u32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_u4_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_packed_(euclidean, u4, neonsdot, u4x2, u4x2, u32, /*norm_value_type=*/u32, f32,
                                   nk_b128_vec_t, nk_dots_packed_u4_neonsdot, nk_euclidean_through_u32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_u4_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_symmetric_(angular, u4, neonsdot, u4x2, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u4_neonsdot, nk_angular_through_u32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_u4_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_symmetric_(euclidean, u4, neonsdot, u4x2, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u4_neonsdot, nk_euclidean_through_u32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_u4_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONSDOT
#endif // NK_TARGET_ARM_
#endif // NK_SPATIALS_NEONSDOT_H
