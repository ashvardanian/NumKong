/**
 *  @brief Batched Spatial Distances for NEON.
 *  @file include/numkong/spatials/neon.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_NEON_H
#define NK_SPATIALS_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/spatial/neon.h"
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

nk_define_cross_normalized_packed_(angular, f32, neon, f32, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f32_neon, nk_angular_through_f64_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f32_, nk_load_b256_neon_, nk_partial_load_b64x4_serial_,
                                   nk_store_b256_neon_, nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f32, neon, f32, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f32_neon, nk_euclidean_through_f64_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f32_, nk_load_b256_neon_, nk_partial_load_b64x4_serial_,
                                   nk_store_b256_neon_, nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f32, neon, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f32_neon, nk_angular_through_f64_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f32_, nk_load_b256_neon_, nk_partial_load_b64x4_serial_,
                                      nk_store_b256_neon_, nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f32, neon, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f32_neon, nk_euclidean_through_f64_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f32_, nk_load_b256_neon_, nk_partial_load_b64x4_serial_,
                                      nk_store_b256_neon_, nk_partial_store_b64x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, bf16, neon, bf16, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_bf16_neon, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_bf16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, bf16, neon, bf16, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_bf16_neon, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_bf16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, bf16, neon, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_neon, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_bf16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, bf16, neon, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_neon, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_bf16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, f16, neon, f16, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_neon, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f16, neon, f16, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_neon, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f16, neon, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_neon, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f16, neon, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_neon, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, f64, neon, f64, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f64_neon, nk_angular_through_f64_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f64_, nk_load_b256_neon_, nk_partial_load_b64x4_serial_,
                                   nk_store_b256_neon_, nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f64, neon, f64, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f64_neon, nk_euclidean_through_f64_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f64_, nk_load_b256_neon_, nk_partial_load_b64x4_serial_,
                                   nk_store_b256_neon_, nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f64, neon, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f64_neon, nk_angular_through_f64_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f64_, nk_load_b256_neon_, nk_partial_load_b64x4_serial_,
                                      nk_store_b256_neon_, nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f64, neon, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f64_neon, nk_euclidean_through_f64_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f64_, nk_load_b256_neon_, nk_partial_load_b64x4_serial_,
                                      nk_store_b256_neon_, nk_partial_store_b64x4_serial_, 1)

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
#endif // NK_SPATIALS_NEON_H
