/**
 *  @file include/numkong/spatials/loongsonasx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *  @brief Batched spatial distances for LoongArch LASX, 256-bit.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NUMKONG_SPATIALS_LOONGSONASX_H
#define NUMKONG_SPATIALS_LOONGSONASX_H

#if NUMKONG_ARCH_LOONGARCH64_
#if NUMKONG_TARGET_LOONGSONASX

#include "numkong/spatial/loongsonasx.h"
#include "numkong/spatial/serial.h"
#include "numkong/dots/loongsonasx.h"

#if defined(__cplusplus)
extern "C" {
#endif

nk_define_cross_normalized_packed_(angular, f32, loongsonasx, f32, f32, f64, /*norm_value_type=*/f64, f64,
                                   nk_b256_vec_t, nk_dots_packed_f32_loongsonasx,
                                   nk_angular_through_f64_from_dot_loongsonasx_, nk_dots_reduce_sumsq_f32_,
                                   nk_load_b256_loongsonasx_, nk_partial_load_b64x4_serial_, nk_store_b256_loongsonasx_,
                                   nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f32, loongsonasx, f32, f32, f64, /*norm_value_type=*/f64, f64,
                                   nk_b256_vec_t, nk_dots_packed_f32_loongsonasx,
                                   nk_euclidean_through_f64_from_dot_loongsonasx_, nk_dots_reduce_sumsq_f32_,
                                   nk_load_b256_loongsonasx_, nk_partial_load_b64x4_serial_, nk_store_b256_loongsonasx_,
                                   nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f32, loongsonasx, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f32_loongsonasx, nk_angular_through_f64_from_dot_loongsonasx_,
                                      nk_dots_reduce_sumsq_f32_, nk_load_b256_loongsonasx_,
                                      nk_partial_load_b64x4_serial_, nk_store_b256_loongsonasx_,
                                      nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f32, loongsonasx, f32, f64, /*norm_value_type=*/f64, f64,
                                      nk_b256_vec_t, nk_dots_symmetric_f32_loongsonasx,
                                      nk_euclidean_through_f64_from_dot_loongsonasx_, nk_dots_reduce_sumsq_f32_,
                                      nk_load_b256_loongsonasx_, nk_partial_load_b64x4_serial_,
                                      nk_store_b256_loongsonasx_, nk_partial_store_b64x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, f64, loongsonasx, f64, f64, f64, /*norm_value_type=*/f64, f64,
                                   nk_b256_vec_t, nk_dots_packed_f64_loongsonasx,
                                   nk_angular_through_f64_from_dot_loongsonasx_, nk_dots_reduce_sumsq_f64_,
                                   nk_load_b256_loongsonasx_, nk_partial_load_b64x4_serial_, nk_store_b256_loongsonasx_,
                                   nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f64, loongsonasx, f64, f64, f64, /*norm_value_type=*/f64, f64,
                                   nk_b256_vec_t, nk_dots_packed_f64_loongsonasx,
                                   nk_euclidean_through_f64_from_dot_loongsonasx_, nk_dots_reduce_sumsq_f64_,
                                   nk_load_b256_loongsonasx_, nk_partial_load_b64x4_serial_, nk_store_b256_loongsonasx_,
                                   nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f64, loongsonasx, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f64_loongsonasx, nk_angular_through_f64_from_dot_loongsonasx_,
                                      nk_dots_reduce_sumsq_f64_, nk_load_b256_loongsonasx_,
                                      nk_partial_load_b64x4_serial_, nk_store_b256_loongsonasx_,
                                      nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f64, loongsonasx, f64, f64, /*norm_value_type=*/f64, f64,
                                      nk_b256_vec_t, nk_dots_symmetric_f64_loongsonasx,
                                      nk_euclidean_through_f64_from_dot_loongsonasx_, nk_dots_reduce_sumsq_f64_,
                                      nk_load_b256_loongsonasx_, nk_partial_load_b64x4_serial_,
                                      nk_store_b256_loongsonasx_, nk_partial_store_b64x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, i8, loongsonasx, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_loongsonasx, nk_angular_through_i32_from_dot_loongsonasx_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_loongsonasx_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_loongsonasx_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, i8, loongsonasx, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_loongsonasx, nk_angular_through_i32_from_dot_loongsonasx_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_loongsonasx_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_loongsonasx_,
                                      nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(euclidean, i8, loongsonasx, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_loongsonasx, nk_euclidean_through_i32_from_dot_loongsonasx_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_loongsonasx_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_loongsonasx_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, i8, loongsonasx, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_loongsonasx, nk_euclidean_through_i32_from_dot_loongsonasx_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_loongsonasx_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_loongsonasx_,
                                      nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, u8, loongsonasx, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_loongsonasx, nk_angular_through_u32_from_dot_loongsonasx_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_loongsonasx_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_loongsonasx_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, u8, loongsonasx, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_loongsonasx, nk_angular_through_u32_from_dot_loongsonasx_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_loongsonasx_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_loongsonasx_,
                                      nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(euclidean, u8, loongsonasx, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_loongsonasx, nk_euclidean_through_u32_from_dot_loongsonasx_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_loongsonasx_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_loongsonasx_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, u8, loongsonasx, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_loongsonasx, nk_euclidean_through_u32_from_dot_loongsonasx_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_loongsonasx_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_loongsonasx_,
                                      nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, bf16, loongsonasx, bf16, bf16, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_bf16_loongsonasx,
                                   nk_angular_through_f32_from_dot_loongsonasx_, nk_dots_reduce_sumsq_bf16_,
                                   nk_load_b128_loongsonasx_, nk_partial_load_b32x4_serial_, nk_store_b128_loongsonasx_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, bf16, loongsonasx, bf16, bf16, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_bf16_loongsonasx,
                                   nk_euclidean_through_f32_from_dot_loongsonasx_, nk_dots_reduce_sumsq_bf16_,
                                   nk_load_b128_loongsonasx_, nk_partial_load_b32x4_serial_, nk_store_b128_loongsonasx_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, bf16, loongsonasx, bf16, f32, /*norm_value_type=*/f32, f32,
                                      nk_b128_vec_t, nk_dots_symmetric_bf16_loongsonasx,
                                      nk_angular_through_f32_from_dot_loongsonasx_, nk_dots_reduce_sumsq_bf16_,
                                      nk_load_b128_loongsonasx_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_loongsonasx_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, bf16, loongsonasx, bf16, f32, /*norm_value_type=*/f32, f32,
                                      nk_b128_vec_t, nk_dots_symmetric_bf16_loongsonasx,
                                      nk_euclidean_through_f32_from_dot_loongsonasx_, nk_dots_reduce_sumsq_bf16_,
                                      nk_load_b128_loongsonasx_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_loongsonasx_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, f16, loongsonasx, f16, f32, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_f16_loongsonasx,
                                   nk_angular_through_f32_from_dot_loongsonasx_, nk_dots_reduce_sumsq_f16_,
                                   nk_load_b128_loongsonasx_, nk_partial_load_b32x4_serial_, nk_store_b128_loongsonasx_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f16, loongsonasx, f16, f32, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_f16_loongsonasx,
                                   nk_euclidean_through_f32_from_dot_loongsonasx_, nk_dots_reduce_sumsq_f16_,
                                   nk_load_b128_loongsonasx_, nk_partial_load_b32x4_serial_, nk_store_b128_loongsonasx_,
                                   nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f16, loongsonasx, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_loongsonasx, nk_angular_through_f32_from_dot_loongsonasx_,
                                      nk_dots_reduce_sumsq_f16_, nk_load_b128_loongsonasx_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_loongsonasx_,
                                      nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f16, loongsonasx, f16, f32, /*norm_value_type=*/f32, f32,
                                      nk_b128_vec_t, nk_dots_symmetric_f16_loongsonasx,
                                      nk_euclidean_through_f32_from_dot_loongsonasx_, nk_dots_reduce_sumsq_f16_,
                                      nk_load_b128_loongsonasx_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_loongsonasx_, nk_partial_store_b32x4_serial_, 1)

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_LOONGSONASX
#endif // NUMKONG_ARCH_LOONGARCH64_
#endif // NUMKONG_SPATIALS_LOONGSONASX_H
