/**
 *  @file include/numkong/spatials/alder.h
 *  @author Ash Vardanian
 *  @date March 4, 2026
 *  @brief Batched spatial distances for Alder Lake, AVX-VNNI.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NUMKONG_SPATIALS_ALDER_H
#define NUMKONG_SPATIALS_ALDER_H

#if NUMKONG_ARCH_X86_64_
#if NUMKONG_TARGET_ALDER

#include "numkong/spatial/haswell.h"
#include "numkong/dots/alder.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni")
#endif

nk_define_cross_normalized_packed_(angular, i8, alder, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_alder, nk_angular_through_i32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(angular, i8, alder, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_alder, nk_angular_through_i32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(euclidean, i8, alder, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_alder, nk_euclidean_through_i32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(euclidean, i8, alder, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_alder, nk_euclidean_through_i32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(angular, u8, alder, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_alder, nk_angular_through_u32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(angular, u8, alder, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_alder, nk_angular_through_u32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(euclidean, u8, alder, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_alder, nk_euclidean_through_u32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(euclidean, u8, alder, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_alder, nk_euclidean_through_u32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(angular, e2m3, alder, e2m3, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e2m3_alder, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e2m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_packed_(euclidean, e2m3, alder, e2m3, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e2m3_alder, nk_euclidean_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e2m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(angular, e2m3, alder, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m3_alder, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e2m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e2m3, alder, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m3_alder, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e2m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(angular, e2m1, alder, e2m1x2, e2m1x2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e2m1_alder, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e2m1_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 2)
nk_define_cross_normalized_packed_(euclidean, e2m1, alder, e2m1x2, e2m1x2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e2m1_alder, nk_euclidean_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e2m1_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 2)
nk_define_cross_normalized_symmetric_(angular, e2m1, alder, e2m1x2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m1_alder, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e2m1_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 2)
nk_define_cross_normalized_symmetric_(euclidean, e2m1, alder, e2m1x2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m1_alder, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e2m1_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 2)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_ALDER
#endif // NUMKONG_ARCH_X86_64_
#endif // NUMKONG_SPATIALS_ALDER_H
