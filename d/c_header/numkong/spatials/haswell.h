/**
 *  @brief Batched Spatial Distances for Haswell (AVX2).
 *  @file include/numkong/spatials/haswell.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_HASWELL_H
#define NK_SPATIALS_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL

#include "numkong/spatial/haswell.h"
#include "numkong/spatial/serial.h"
#include "numkong/dots/haswell.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "popcnt")
#endif

nk_define_cross_normalized_packed_(angular, f32, haswell, f32, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f32_haswell, nk_angular_through_f64_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_f32_, nk_load_b256_haswell_, nk_partial_load_b64x4_haswell_,
                                   nk_store_b256_haswell_, nk_partial_store_b64x4_haswell_, 1)
nk_define_cross_normalized_packed_(euclidean, f32, haswell, f32, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f32_haswell, nk_euclidean_through_f64_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_f32_, nk_load_b256_haswell_, nk_partial_load_b64x4_haswell_,
                                   nk_store_b256_haswell_, nk_partial_store_b64x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(angular, f32, haswell, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f32_haswell, nk_angular_through_f64_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_f32_, nk_load_b256_haswell_, nk_partial_load_b64x4_haswell_,
                                      nk_store_b256_haswell_, nk_partial_store_b64x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f32, haswell, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f32_haswell, nk_euclidean_through_f64_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_f32_, nk_load_b256_haswell_, nk_partial_load_b64x4_haswell_,
                                      nk_store_b256_haswell_, nk_partial_store_b64x4_haswell_, 1)

nk_define_cross_normalized_packed_(angular, f64, haswell, f64, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f64_haswell, nk_angular_through_f64_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_f64_, nk_load_b256_haswell_, nk_partial_load_b64x4_haswell_,
                                   nk_store_b256_haswell_, nk_partial_store_b64x4_haswell_, 1)
nk_define_cross_normalized_packed_(euclidean, f64, haswell, f64, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f64_haswell, nk_euclidean_through_f64_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_f64_, nk_load_b256_haswell_, nk_partial_load_b64x4_haswell_,
                                   nk_store_b256_haswell_, nk_partial_store_b64x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(angular, f64, haswell, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f64_haswell, nk_angular_through_f64_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_f64_, nk_load_b256_haswell_, nk_partial_load_b64x4_haswell_,
                                      nk_store_b256_haswell_, nk_partial_store_b64x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f64, haswell, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f64_haswell, nk_euclidean_through_f64_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_f64_, nk_load_b256_haswell_, nk_partial_load_b64x4_haswell_,
                                      nk_store_b256_haswell_, nk_partial_store_b64x4_haswell_, 1)

nk_define_cross_normalized_packed_(angular, f16, haswell, f16, f32, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_haswell, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_f16_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_packed_(euclidean, f16, haswell, f16, f32, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_haswell, nk_euclidean_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_f16_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(angular, f16, haswell, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_haswell, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_f16_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f16, haswell, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_haswell, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_f16_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(angular, bf16, haswell, bf16, f32, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_bf16_haswell, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_bf16_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_packed_(euclidean, bf16, haswell, bf16, f32, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_bf16_haswell,
                                   nk_euclidean_through_f32_from_dot_haswell_, nk_dots_reduce_sumsq_bf16_,
                                   nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_, nk_store_b128_haswell_,
                                   nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(angular, bf16, haswell, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_haswell, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_bf16_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(euclidean, bf16, haswell, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_haswell, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_bf16_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(angular, e4m3, haswell, e4m3, f32, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e4m3_haswell, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e4m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_packed_(euclidean, e4m3, haswell, e4m3, f32, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e4m3_haswell,
                                   nk_euclidean_through_f32_from_dot_haswell_, nk_dots_reduce_sumsq_e4m3_,
                                   nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_, nk_store_b128_haswell_,
                                   nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(angular, e4m3, haswell, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e4m3_haswell, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e4m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e4m3, haswell, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e4m3_haswell, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e4m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(angular, e5m2, haswell, e5m2, f32, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e5m2_haswell, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e5m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_packed_(euclidean, e5m2, haswell, e5m2, f32, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e5m2_haswell,
                                   nk_euclidean_through_f32_from_dot_haswell_, nk_dots_reduce_sumsq_e5m2_,
                                   nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_, nk_store_b128_haswell_,
                                   nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(angular, e5m2, haswell, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e5m2_haswell, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e5m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e5m2, haswell, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e5m2_haswell, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e5m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(angular, e2m3, haswell, e2m3, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e2m3_haswell, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e2m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_packed_(euclidean, e2m3, haswell, e2m3, e2m3, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e2m3_haswell,
                                   nk_euclidean_through_f32_from_dot_haswell_, nk_dots_reduce_sumsq_e2m3_,
                                   nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_, nk_store_b128_haswell_,
                                   nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(angular, e2m3, haswell, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m3_haswell, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e2m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e2m3, haswell, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m3_haswell, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e2m3_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(angular, e3m2, haswell, e3m2, e3m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e3m2_haswell, nk_angular_through_f32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_e3m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_packed_(euclidean, e3m2, haswell, e3m2, e3m2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e3m2_haswell,
                                   nk_euclidean_through_f32_from_dot_haswell_, nk_dots_reduce_sumsq_e3m2_,
                                   nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_, nk_store_b128_haswell_,
                                   nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(angular, e3m2, haswell, e3m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e3m2_haswell, nk_angular_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e3m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e3m2, haswell, e3m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e3m2_haswell, nk_euclidean_through_f32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_e3m2_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(angular, i8, haswell, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_haswell, nk_angular_through_i32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(angular, i8, haswell, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_haswell, nk_angular_through_i32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(euclidean, i8, haswell, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_haswell, nk_euclidean_through_i32_from_dot_haswell_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(euclidean, i8, haswell, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_haswell, nk_euclidean_through_i32_from_dot_haswell_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(angular, u8, haswell, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_haswell, nk_angular_through_u32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(angular, u8, haswell, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_haswell, nk_angular_through_u32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

nk_define_cross_normalized_packed_(euclidean, u8, haswell, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_haswell, nk_euclidean_through_u32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_u8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                   nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)
nk_define_cross_normalized_symmetric_(euclidean, u8, haswell, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_haswell, nk_euclidean_through_u32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_u8_, nk_load_b128_haswell_, nk_partial_load_b32x4_haswell_,
                                      nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, 1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_
#endif // NK_SPATIALS_HASWELL_H
