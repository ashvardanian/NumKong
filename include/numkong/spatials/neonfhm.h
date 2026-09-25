/**
 *  @file include/numkong/spatials/neonfhm.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *  @brief Batched spatial distances for NEON FHMA.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NUMKONG_SPATIALS_NEONFHM_H
#define NUMKONG_SPATIALS_NEONFHM_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONFHM

#include "numkong/spatial/neon.h"
#include "numkong/dots/neonfhm.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

nk_define_cross_normalized_packed_(angular, f16, neonfhm, f16, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_neonfhm, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f16, neonfhm, f16, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_neonfhm, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f16, neonfhm, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_neonfhm, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f16, neonfhm, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_neonfhm, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, e4m3, neonfhm, e4m3, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e4m3_neonfhm, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e4m3_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e4m3, neonfhm, e4m3, e4m3, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e4m3_neonfhm, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e4m3_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e4m3, neonfhm, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e4m3_neonfhm, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e4m3_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e4m3, neonfhm, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e4m3_neonfhm, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e4m3_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, e5m2, neonfhm, e5m2, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e5m2_neonfhm, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e5m2_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e5m2, neonfhm, e5m2, e5m2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e5m2_neonfhm, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e5m2_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e5m2, neonfhm, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e5m2_neonfhm, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e5m2_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e5m2, neonfhm, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e5m2_neonfhm, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e5m2_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_NEONFHM
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_SPATIALS_NEONFHM_H
