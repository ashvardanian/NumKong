/**
 *  @brief Batched Spatial Distances for NEON FP16 (Half-Precision).
 *  @file include/numkong/spatials/neonhalf.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_NEONHALF_H
#define NK_SPATIALS_NEONHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF

#include "numkong/spatial/neon.h"
#include "numkong/dots/neonhalf.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

nk_define_cross_normalized_packed_(angular, f16, neonhalf, f16, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_neonhalf, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f16, neonhalf, f16, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_neonhalf, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f16, neonhalf, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_neonhalf, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f16, neonhalf, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_neonhalf, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f16_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONHALF
#endif // NK_TARGET_ARM_
#endif // NK_SPATIALS_NEONHALF_H
