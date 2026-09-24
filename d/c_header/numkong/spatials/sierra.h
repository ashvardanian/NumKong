/**
 *  @brief Batched Spatial Distances for Sierra Forest (AVX-VNNI-INT8).
 *  @file include/numkong/spatials/sierra.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_SIERRA_H
#define NK_SPATIALS_SIERRA_H

#if NK_TARGET_X86_
#if NK_TARGET_SIERRA

#include "numkong/spatial/haswell.h"
#include "numkong/spatial/serial.h"
#include "numkong/dots/sierra.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni,avxvnniint8"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni", "avxvnniint8")
#endif

nk_define_cross_normalized_packed_(angular, i8, sierra, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_sierra, nk_angular_through_i32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(angular, i8, sierra, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_sierra, nk_angular_through_i32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(euclidean, i8, sierra, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_sierra, nk_euclidean_through_i32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(euclidean, i8, sierra, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_sierra, nk_euclidean_through_i32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(angular, u8, sierra, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_sierra, nk_angular_through_u32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(angular, u8, sierra, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_sierra, nk_angular_through_u32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(euclidean, u8, sierra, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_sierra, nk_euclidean_through_u32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(euclidean, u8, sierra, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_sierra, nk_euclidean_through_u32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(angular, e2m3, sierra, e2m3, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e2m3_sierra, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e2m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_packed_(euclidean, e2m3, sierra, e2m3, e2m3, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e2m3_sierra,
                                   nk_euclidean_through_f32_from_dot_haswell_, nk_dots_reduce_sumsq_e2m3_,
                                   nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_, nk_store_b128_haswell_,
                                   nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(angular, e2m3, sierra, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m3_sierra, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e2m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e2m3, sierra, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m3_sierra, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e2m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SIERRA
#endif // NK_TARGET_X86_
#endif // NK_SPATIALS_SIERRA_H
