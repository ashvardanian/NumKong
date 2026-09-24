/**
 *  @file include/numkong/spatials/v128.h
 *  @author Ash Vardanian
 *  @date September 12, 2026
 *  @brief Batched spatial distances for WASM.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_V128_H
#define NK_SPATIALS_V128_H

#if NK_TARGET_V128

#include "numkong/spatial/v128.h"
#include "numkong/dots/v128.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

nk_define_cross_normalized_packed_(angular, bf16, v128, bf16, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_bf16_v128, nk_angular_through_f32_from_dot_v128_,
                                   nk_dots_reduce_sumsq_bf16_, nk_load_b128_v128_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_v128_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, bf16, v128, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_v128, nk_angular_through_f32_from_dot_v128_,
                                      nk_dots_reduce_sumsq_bf16_, nk_load_b128_v128_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, bf16, v128, bf16, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_bf16_v128, nk_euclidean_through_f32_from_dot_v128_,
                                   nk_dots_reduce_sumsq_bf16_, nk_load_b128_v128_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_v128_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, bf16, v128, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_v128, nk_euclidean_through_f32_from_dot_v128_,
                                      nk_dots_reduce_sumsq_bf16_, nk_load_b128_v128_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, i8, v128, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_v128, nk_angular_through_i32_from_dot_v128_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_v128_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_v128_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, i8, v128, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_v128, nk_angular_through_i32_from_dot_v128_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_v128_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, i8, v128, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_v128, nk_euclidean_through_i32_from_dot_v128_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_v128_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_v128_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, i8, v128, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_v128, nk_euclidean_through_i32_from_dot_v128_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_v128_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, u8, v128, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_v128, nk_angular_through_u32_from_dot_v128_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_v128_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_v128_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, u8, v128, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_v128, nk_angular_through_u32_from_dot_v128_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_v128_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, u8, v128, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_v128, nk_euclidean_through_u32_from_dot_v128_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_v128_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_v128_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, u8, v128, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_v128, nk_euclidean_through_u32_from_dot_v128_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_v128_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128_, nk_partial_store_b32x4_serial_, 1)

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128
#endif // NK_SPATIALS_V128_H
