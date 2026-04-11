/**
 *  @brief Batched Spatial Distances for Serial (non-SIMD) Backends.
 *  @file include/numkong/spatials/serial.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_SERIAL_H
#define NK_SPATIALS_SERIAL_H

#include "numkong/dots/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* Optimize serial fallbacks for size — see dots/serial.h for rationale. */
#if defined(NDEBUG)
#if defined(_MSC_VER)
#pragma optimize("s", on)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((minsize)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("Os")
#endif
#endif

nk_define_cross_normalized_packed_(angular, f64, serial, f64, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f64_serial, nk_angular_through_f64_from_dot_serial_,
                                   nk_dots_reduce_sumsq_f64_, nk_load_b256_serial_, nk_partial_load_b64x4_serial_,
                                   nk_store_b256_serial_, nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f64, serial, f64, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f64_serial, nk_euclidean_through_f64_from_dot_serial_,
                                   nk_dots_reduce_sumsq_f64_, nk_load_b256_serial_, nk_partial_load_b64x4_serial_,
                                   nk_store_b256_serial_, nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f64, serial, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f64_serial, nk_angular_through_f64_from_dot_serial_,
                                      nk_dots_reduce_sumsq_f64_, nk_load_b256_serial_, nk_partial_load_b64x4_serial_,
                                      nk_store_b256_serial_, nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f64, serial, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f64_serial, nk_euclidean_through_f64_from_dot_serial_,
                                      nk_dots_reduce_sumsq_f64_, nk_load_b256_serial_, nk_partial_load_b64x4_serial_,
                                      nk_store_b256_serial_, nk_partial_store_b64x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, f32, serial, f32, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f32_serial, nk_angular_through_f64_from_dot_serial_,
                                   nk_dots_reduce_sumsq_f32_, nk_load_b256_serial_, nk_partial_load_b64x4_serial_,
                                   nk_store_b256_serial_, nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f32, serial, f32, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f32_serial, nk_euclidean_through_f64_from_dot_serial_,
                                   nk_dots_reduce_sumsq_f32_, nk_load_b256_serial_, nk_partial_load_b64x4_serial_,
                                   nk_store_b256_serial_, nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f32, serial, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f32_serial, nk_angular_through_f64_from_dot_serial_,
                                      nk_dots_reduce_sumsq_f32_, nk_load_b256_serial_, nk_partial_load_b64x4_serial_,
                                      nk_store_b256_serial_, nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f32, serial, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f32_serial, nk_euclidean_through_f64_from_dot_serial_,
                                      nk_dots_reduce_sumsq_f32_, nk_load_b256_serial_, nk_partial_load_b64x4_serial_,
                                      nk_store_b256_serial_, nk_partial_store_b64x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, f16, serial, f16, f32, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_serial, nk_angular_through_f32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_f16_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f16, serial, f16, f32, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_serial, nk_euclidean_through_f32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_f16_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f16, serial, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_serial, nk_angular_through_f32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_f16_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f16, serial, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_serial, nk_euclidean_through_f32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_f16_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, bf16, serial, bf16, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_bf16_serial, nk_angular_through_f32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_bf16_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, bf16, serial, bf16, bf16, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_bf16_serial, nk_euclidean_through_f32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_bf16_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, bf16, serial, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_serial, nk_angular_through_f32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_bf16_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, bf16, serial, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_serial, nk_euclidean_through_f32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_bf16_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, e4m3, serial, e4m3, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e4m3_serial, nk_angular_through_f32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_e4m3_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e4m3, serial, e4m3, e4m3, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e4m3_serial, nk_euclidean_through_f32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_e4m3_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e4m3, serial, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e4m3_serial, nk_angular_through_f32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_e4m3_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e4m3, serial, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e4m3_serial, nk_euclidean_through_f32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_e4m3_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, e5m2, serial, e5m2, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e5m2_serial, nk_angular_through_f32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_e5m2_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e5m2, serial, e5m2, e5m2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e5m2_serial, nk_euclidean_through_f32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_e5m2_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e5m2, serial, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e5m2_serial, nk_angular_through_f32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_e5m2_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e5m2, serial, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e5m2_serial, nk_euclidean_through_f32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_e5m2_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, e2m3, serial, e2m3, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e2m3_serial, nk_angular_through_f32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_e2m3_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e2m3, serial, e2m3, e2m3, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e2m3_serial, nk_euclidean_through_f32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_e2m3_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e2m3, serial, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m3_serial, nk_angular_through_f32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_e2m3_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e2m3, serial, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m3_serial, nk_euclidean_through_f32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_e2m3_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, e3m2, serial, e3m2, e3m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e3m2_serial, nk_angular_through_f32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_e3m2_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e3m2, serial, e3m2, e3m2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e3m2_serial, nk_euclidean_through_f32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_e3m2_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e3m2, serial, e3m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e3m2_serial, nk_angular_through_f32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_e3m2_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e3m2, serial, e3m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e3m2_serial, nk_euclidean_through_f32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_e3m2_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, i8, serial, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_serial, nk_angular_through_i32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, i8, serial, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_serial, nk_euclidean_through_i32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, i8, serial, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_serial, nk_angular_through_i32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, i8, serial, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_serial, nk_euclidean_through_i32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, u8, serial, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_serial, nk_angular_through_u32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, u8, serial, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_serial, nk_euclidean_through_u32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, u8, serial, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_serial, nk_angular_through_u32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, u8, serial, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_serial, nk_euclidean_through_u32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, i4, serial, i4x2, i4x2, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i4_serial, nk_angular_through_i32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_i4_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_packed_(euclidean, i4, serial, i4x2, i4x2, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i4_serial, nk_euclidean_through_i32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_i4_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_symmetric_(angular, i4, serial, i4x2, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i4_serial, nk_angular_through_i32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_i4_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_symmetric_(euclidean, i4, serial, i4x2, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i4_serial, nk_euclidean_through_i32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_i4_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 2)

nk_define_cross_normalized_packed_(angular, u4, serial, u4x2, u4x2, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u4_serial, nk_angular_through_u32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_u4_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_packed_(euclidean, u4, serial, u4x2, u4x2, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u4_serial, nk_euclidean_through_u32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_u4_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_symmetric_(angular, u4, serial, u4x2, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u4_serial, nk_angular_through_u32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_u4_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_symmetric_(euclidean, u4, serial, u4x2, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u4_serial, nk_euclidean_through_u32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_u4_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 2)

#if defined(NDEBUG)
#if defined(_MSC_VER)
#pragma optimize("", on)
#elif defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_SPATIALS_SERIAL_H
