/**
 *  @brief Batched Spatial Distances for WASM Relaxed SIMD.
 *  @file include/numkong/spatials/v128relaxed.h
 *  @author Ash Vardanian
 *  @date March 5, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_V128RELAXED_H
#define NK_SPATIALS_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/spatial/v128relaxed.h"
#include "numkong/dots/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

nk_define_cross_normalized_packed_(angular, i8, v128relaxed, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_v128relaxed, nk_angular_through_i32_from_dot_v128relaxed_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, i8, v128relaxed, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_v128relaxed, nk_angular_through_i32_from_dot_v128relaxed_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_v128relaxed_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                      nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, i8, v128relaxed, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_v128relaxed, nk_euclidean_through_i32_from_dot_v128relaxed_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, i8, v128relaxed, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_v128relaxed, nk_euclidean_through_i32_from_dot_v128relaxed_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_v128relaxed_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                      nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, u8, v128relaxed, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_v128relaxed, nk_angular_through_u32_from_dot_v128relaxed_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, u8, v128relaxed, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_v128relaxed, nk_angular_through_u32_from_dot_v128relaxed_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_v128relaxed_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                      nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, u8, v128relaxed, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_v128relaxed, nk_euclidean_through_u32_from_dot_v128relaxed_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, u8, v128relaxed, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_v128relaxed, nk_euclidean_through_u32_from_dot_v128relaxed_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_v128relaxed_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                      nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, e2m3, v128relaxed, e2m3, e2m3, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e2m3_v128relaxed,
                                   nk_angular_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_e2m3_,
                                   nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e2m3, v128relaxed, e2m3, f32, /*norm_value_type=*/f32, f32,
                                      nk_b128_vec_t, nk_dots_symmetric_e2m3_v128relaxed,
                                      nk_angular_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_e2m3_,
                                      nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e2m3, v128relaxed, e2m3, e2m3, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e2m3_v128relaxed,
                                   nk_euclidean_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_e2m3_,
                                   nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e2m3, v128relaxed, e2m3, f32, /*norm_value_type=*/f32, f32,
                                      nk_b128_vec_t, nk_dots_symmetric_e2m3_v128relaxed,
                                      nk_euclidean_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_e2m3_,
                                      nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, e3m2, v128relaxed, e3m2, e3m2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e3m2_v128relaxed,
                                   nk_angular_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_e3m2_,
                                   nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e3m2, v128relaxed, e3m2, f32, /*norm_value_type=*/f32, f32,
                                      nk_b128_vec_t, nk_dots_symmetric_e3m2_v128relaxed,
                                      nk_angular_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_e3m2_,
                                      nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e3m2, v128relaxed, e3m2, e3m2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e3m2_v128relaxed,
                                   nk_euclidean_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_e3m2_,
                                   nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e3m2, v128relaxed, e3m2, f32, /*norm_value_type=*/f32, f32,
                                      nk_b128_vec_t, nk_dots_symmetric_e3m2_v128relaxed,
                                      nk_euclidean_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_e3m2_,
                                      nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, e4m3, v128relaxed, e4m3, e4m3, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e4m3_v128relaxed,
                                   nk_angular_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_e4m3_,
                                   nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e4m3, v128relaxed, e4m3, f32, /*norm_value_type=*/f32, f32,
                                      nk_b128_vec_t, nk_dots_symmetric_e4m3_v128relaxed,
                                      nk_angular_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_e4m3_,
                                      nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e4m3, v128relaxed, e4m3, e4m3, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e4m3_v128relaxed,
                                   nk_euclidean_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_e4m3_,
                                   nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e4m3, v128relaxed, e4m3, f32, /*norm_value_type=*/f32, f32,
                                      nk_b128_vec_t, nk_dots_symmetric_e4m3_v128relaxed,
                                      nk_euclidean_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_e4m3_,
                                      nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, e5m2, v128relaxed, e5m2, e5m2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e5m2_v128relaxed,
                                   nk_angular_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_e5m2_,
                                   nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e5m2, v128relaxed, e5m2, f32, /*norm_value_type=*/f32, f32,
                                      nk_b128_vec_t, nk_dots_symmetric_e5m2_v128relaxed,
                                      nk_angular_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_e5m2_,
                                      nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e5m2, v128relaxed, e5m2, e5m2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e5m2_v128relaxed,
                                   nk_euclidean_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_e5m2_,
                                   nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e5m2, v128relaxed, e5m2, f32, /*norm_value_type=*/f32, f32,
                                      nk_b128_vec_t, nk_dots_symmetric_e5m2_v128relaxed,
                                      nk_euclidean_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_e5m2_,
                                      nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, bf16, v128relaxed, bf16, f32, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_bf16_v128relaxed,
                                   nk_angular_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_bf16_,
                                   nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, bf16, v128relaxed, bf16, f32, /*norm_value_type=*/f32, f32,
                                      nk_b128_vec_t, nk_dots_symmetric_bf16_v128relaxed,
                                      nk_angular_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_bf16_,
                                      nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, bf16, v128relaxed, bf16, f32, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_bf16_v128relaxed,
                                   nk_euclidean_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_bf16_,
                                   nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, bf16, v128relaxed, bf16, f32, /*norm_value_type=*/f32, f32,
                                      nk_b128_vec_t, nk_dots_symmetric_bf16_v128relaxed,
                                      nk_euclidean_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_bf16_,
                                      nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, f16, v128relaxed, f16, f32, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_f16_v128relaxed,
                                   nk_angular_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_f16_,
                                   nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f16, v128relaxed, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_v128relaxed, nk_angular_through_f32_from_dot_v128relaxed_,
                                      nk_dots_reduce_sumsq_f16_, nk_load_b128_v128relaxed_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                      nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f16, v128relaxed, f16, f32, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_f16_v128relaxed,
                                   nk_euclidean_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_f16_,
                                   nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_, nk_store_b128_v128relaxed_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f16, v128relaxed, f16, f32, /*norm_value_type=*/f32, f32,
                                      nk_b128_vec_t, nk_dots_symmetric_f16_v128relaxed,
                                      nk_euclidean_through_f32_from_dot_v128relaxed_, nk_dots_reduce_sumsq_f16_,
                                      nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, f32, v128relaxed, f32, f32, f64, /*norm_value_type=*/f64, f64,
                                   nk_b256_vec_t, nk_dots_packed_f32_v128relaxed,
                                   nk_angular_through_f64_from_dot_serial_, nk_dots_reduce_sumsq_f32_,
                                   nk_load_b256_v128relaxed_, nk_partial_load_b64x4_serial_, nk_store_b256_v128relaxed_,
                                   nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f32, v128relaxed, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f32_v128relaxed, nk_angular_through_f64_from_dot_serial_,
                                      nk_dots_reduce_sumsq_f32_, nk_load_b256_v128relaxed_,
                                      nk_partial_load_b64x4_serial_, nk_store_b256_v128relaxed_,
                                      nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f32, v128relaxed, f32, f32, f64, /*norm_value_type=*/f64, f64,
                                   nk_b256_vec_t, nk_dots_packed_f32_v128relaxed,
                                   nk_euclidean_through_f64_from_dot_serial_, nk_dots_reduce_sumsq_f32_,
                                   nk_load_b256_v128relaxed_, nk_partial_load_b64x4_serial_, nk_store_b256_v128relaxed_,
                                   nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f32, v128relaxed, f32, f64, /*norm_value_type=*/f64, f64,
                                      nk_b256_vec_t, nk_dots_symmetric_f32_v128relaxed,
                                      nk_euclidean_through_f64_from_dot_serial_, nk_dots_reduce_sumsq_f32_,
                                      nk_load_b256_v128relaxed_, nk_partial_load_b64x4_serial_,
                                      nk_store_b256_v128relaxed_, nk_partial_store_b64x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, f64, v128relaxed, f64, f64, f64, /*norm_value_type=*/f64, f64,
                                   nk_b256_vec_t, nk_dots_packed_f64_v128relaxed,
                                   nk_angular_through_f64_from_dot_serial_, nk_dots_reduce_sumsq_f64_,
                                   nk_load_b256_v128relaxed_, nk_partial_load_b64x4_serial_, nk_store_b256_v128relaxed_,
                                   nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f64, v128relaxed, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f64_v128relaxed, nk_angular_through_f64_from_dot_serial_,
                                      nk_dots_reduce_sumsq_f64_, nk_load_b256_v128relaxed_,
                                      nk_partial_load_b64x4_serial_, nk_store_b256_v128relaxed_,
                                      nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f64, v128relaxed, f64, f64, f64, /*norm_value_type=*/f64, f64,
                                   nk_b256_vec_t, nk_dots_packed_f64_v128relaxed,
                                   nk_euclidean_through_f64_from_dot_serial_, nk_dots_reduce_sumsq_f64_,
                                   nk_load_b256_v128relaxed_, nk_partial_load_b64x4_serial_, nk_store_b256_v128relaxed_,
                                   nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f64, v128relaxed, f64, f64, /*norm_value_type=*/f64, f64,
                                      nk_b256_vec_t, nk_dots_symmetric_f64_v128relaxed,
                                      nk_euclidean_through_f64_from_dot_serial_, nk_dots_reduce_sumsq_f64_,
                                      nk_load_b256_v128relaxed_, nk_partial_load_b64x4_serial_,
                                      nk_store_b256_v128relaxed_, nk_partial_store_b64x4_serial_, 1)

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_SPATIALS_V128RELAXED_H
