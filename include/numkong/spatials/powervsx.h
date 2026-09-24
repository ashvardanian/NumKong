/**
 *  @file include/numkong/spatials/powervsx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *  @brief Batched spatial distances for Power VSX.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_POWERVSX_H
#define NK_SPATIALS_POWERVSX_H

#if NK_TARGET_POWER64_
#if NK_TARGET_POWERVSX

#include "numkong/spatial/powervsx.h"
#include "numkong/dots/powervsx.h"

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

nk_define_cross_normalized_packed_(angular, f32, powervsx, f32, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f32_powervsx, nk_angular_through_f64_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_f32_, nk_load_b256_powervsx_, nk_partial_load_b64x4_powervsx_,
                                   nk_store_b256_powervsx_, nk_partial_store_b64x4_powervsx_, 1)
nk_define_cross_normalized_packed_(euclidean, f32, powervsx, f32, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f32_powervsx, nk_euclidean_through_f64_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_f32_, nk_load_b256_powervsx_, nk_partial_load_b64x4_powervsx_,
                                   nk_store_b256_powervsx_, nk_partial_store_b64x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(angular, f32, powervsx, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f32_powervsx, nk_angular_through_f64_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_f32_, nk_load_b256_powervsx_,
                                      nk_partial_load_b64x4_powervsx_, nk_store_b256_powervsx_,
                                      nk_partial_store_b64x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f32, powervsx, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f32_powervsx, nk_euclidean_through_f64_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_f32_, nk_load_b256_powervsx_,
                                      nk_partial_load_b64x4_powervsx_, nk_store_b256_powervsx_,
                                      nk_partial_store_b64x4_powervsx_, 1)

nk_define_cross_normalized_packed_(angular, bf16, powervsx, bf16, bf16, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_bf16_powervsx,
                                   nk_angular_through_f32_from_dot_powervsx_, nk_dots_reduce_sumsq_bf16_,
                                   nk_load_b128_powervsx_, nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                   nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_packed_(euclidean, bf16, powervsx, bf16, bf16, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_bf16_powervsx,
                                   nk_euclidean_through_f32_from_dot_powervsx_, nk_dots_reduce_sumsq_bf16_,
                                   nk_load_b128_powervsx_, nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                   nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(angular, bf16, powervsx, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_powervsx, nk_angular_through_f32_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_bf16_, nk_load_b128_powervsx_,
                                      nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                      nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(euclidean, bf16, powervsx, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_powervsx, nk_euclidean_through_f32_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_bf16_, nk_load_b128_powervsx_,
                                      nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                      nk_partial_store_b32x4_powervsx_, 1)

nk_define_cross_normalized_packed_(angular, f16, powervsx, f16, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_powervsx, nk_angular_through_f32_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_f16_, nk_load_b128_powervsx_, nk_partial_load_b32x4_powervsx_,
                                   nk_store_b128_powervsx_, nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_packed_(euclidean, f16, powervsx, f16, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_powervsx, nk_euclidean_through_f32_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_f16_, nk_load_b128_powervsx_, nk_partial_load_b32x4_powervsx_,
                                   nk_store_b128_powervsx_, nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(angular, f16, powervsx, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_powervsx, nk_angular_through_f32_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_f16_, nk_load_b128_powervsx_,
                                      nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                      nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f16, powervsx, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_powervsx, nk_euclidean_through_f32_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_f16_, nk_load_b128_powervsx_,
                                      nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                      nk_partial_store_b32x4_powervsx_, 1)

nk_define_cross_normalized_packed_(angular, i8, powervsx, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_powervsx, nk_angular_through_i32_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_powervsx_, nk_partial_load_b32x4_powervsx_,
                                   nk_store_b128_powervsx_, nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_packed_(euclidean, i8, powervsx, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_powervsx, nk_euclidean_through_i32_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_powervsx_, nk_partial_load_b32x4_powervsx_,
                                   nk_store_b128_powervsx_, nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(angular, i8, powervsx, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_powervsx, nk_angular_through_i32_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_powervsx_, nk_partial_load_b32x4_powervsx_,
                                      nk_store_b128_powervsx_, nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(euclidean, i8, powervsx, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_powervsx, nk_euclidean_through_i32_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_powervsx_, nk_partial_load_b32x4_powervsx_,
                                      nk_store_b128_powervsx_, nk_partial_store_b32x4_powervsx_, 1)

nk_define_cross_normalized_packed_(angular, u8, powervsx, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_powervsx, nk_angular_through_u32_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_powervsx_, nk_partial_load_b32x4_powervsx_,
                                   nk_store_b128_powervsx_, nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_packed_(euclidean, u8, powervsx, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_powervsx, nk_euclidean_through_u32_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_powervsx_, nk_partial_load_b32x4_powervsx_,
                                   nk_store_b128_powervsx_, nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(angular, u8, powervsx, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_powervsx, nk_angular_through_u32_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_powervsx_, nk_partial_load_b32x4_powervsx_,
                                      nk_store_b128_powervsx_, nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(euclidean, u8, powervsx, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_powervsx, nk_euclidean_through_u32_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_powervsx_, nk_partial_load_b32x4_powervsx_,
                                      nk_store_b128_powervsx_, nk_partial_store_b32x4_powervsx_, 1)

nk_define_cross_normalized_packed_(angular, f64, powervsx, f64, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f64_powervsx, nk_angular_through_f64_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_f64_, nk_load_b256_powervsx_, nk_partial_load_b64x4_powervsx_,
                                   nk_store_b256_powervsx_, nk_partial_store_b64x4_powervsx_, 1)
nk_define_cross_normalized_packed_(euclidean, f64, powervsx, f64, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f64_powervsx, nk_euclidean_through_f64_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_f64_, nk_load_b256_powervsx_, nk_partial_load_b64x4_powervsx_,
                                   nk_store_b256_powervsx_, nk_partial_store_b64x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(angular, f64, powervsx, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f64_powervsx, nk_angular_through_f64_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_f64_, nk_load_b256_powervsx_,
                                      nk_partial_load_b64x4_powervsx_, nk_store_b256_powervsx_,
                                      nk_partial_store_b64x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f64, powervsx, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f64_powervsx, nk_euclidean_through_f64_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_f64_, nk_load_b256_powervsx_,
                                      nk_partial_load_b64x4_powervsx_, nk_store_b256_powervsx_,
                                      nk_partial_store_b64x4_powervsx_, 1)

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_POWERVSX
#endif // NK_TARGET_POWER64_
#endif // NK_SPATIALS_POWERVSX_H
