/**
 *  @file include/numkong/spatials/neonfp8.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *  @brief SIMD-accelerated batched spatial distances for NEON FP8DOT4.
 *
 *  @sa include/numkong/spatials.h
 *
 *  Uses FDOT (FEAT_FP8DOT4) for native FP8 dot products, then derives angular and Euclidean
 *  distances via the batched dots/ infrastructure.
 */
#ifndef NUMKONG_SPATIALS_NEONFP8_H
#define NUMKONG_SPATIALS_NEONFP8_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONFP8

#include "numkong/dots/neonfp8.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd+fp8dot4"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd+fp8dot4")
#endif

nk_define_cross_normalized_packed_(angular, e4m3, neonfp8, e4m3, e4m3, f32, f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e4m3_neonfp8, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e4m3_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e4m3, neonfp8, e4m3, e4m3, f32, f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e4m3_neonfp8, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e4m3_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e4m3, neonfp8, e4m3, f32, f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e4m3_neonfp8, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e4m3_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e4m3, neonfp8, e4m3, f32, f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e4m3_neonfp8, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e4m3_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, e5m2, neonfp8, e5m2, e5m2, f32, f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e5m2_neonfp8, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e5m2_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e5m2, neonfp8, e5m2, e5m2, f32, f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e5m2_neonfp8, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e5m2_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e5m2, neonfp8, e5m2, f32, f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e5m2_neonfp8, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e5m2_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e5m2, neonfp8, e5m2, f32, f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e5m2_neonfp8, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e5m2_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, e2m3, neonfp8, e2m3, e2m3, f32, f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e2m3_neonfp8, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e2m3_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e2m3, neonfp8, e2m3, e2m3, f32, f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e2m3_neonfp8, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e2m3_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e2m3, neonfp8, e2m3, f32, f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m3_neonfp8, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e2m3_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e2m3, neonfp8, e2m3, f32, f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m3_neonfp8, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e2m3_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, e2m1, neonfp8, e2m1x2, e2m1x2, f32, f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e2m1_neonfp8, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e2m1_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_packed_(euclidean, e2m1, neonfp8, e2m1x2, e2m1x2, f32, f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e2m1_neonfp8, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e2m1_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_symmetric_(angular, e2m1, neonfp8, e2m1x2, f32, f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m1_neonfp8, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e2m1_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_symmetric_(euclidean, e2m1, neonfp8, e2m1x2, f32, f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m1_neonfp8, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e2m1_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)

nk_define_cross_normalized_packed_(angular, e3m2, neonfp8, e3m2, e3m2, f32, f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e3m2_neonfp8, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e3m2_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e3m2, neonfp8, e3m2, e3m2, f32, f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e3m2_neonfp8, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e3m2_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e3m2, neonfp8, e3m2, f32, f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e3m2_neonfp8, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e3m2_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e3m2, neonfp8, e3m2, f32, f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e3m2_neonfp8, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e3m2_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_NEONFP8
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_SPATIALS_NEONFP8_H
