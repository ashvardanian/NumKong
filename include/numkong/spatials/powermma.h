/**
 *  @file include/numkong/spatials/powermma.h
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief Batched spatial distances for the Power ISA 3.1 Matrix-Multiply Assist, finalized in VSX.
 *
 *  @sa c/numkong/spatials.h
 */
#ifndef NUMKONG_SPATIALS_POWERMMA_H
#define NUMKONG_SPATIALS_POWERMMA_H

#if NUMKONG_ARCH_PPC64_
#if NUMKONG_ARCH_PPC64_POWERMMA_

#include "numkong/cast/powervsx.h"    // `nk_load_b128_powervsx_`, `nk_partial_store_b32x4_powervsx_`
#include "numkong/spatial/powervsx.h" // `nk_angular_through_f32_from_dot_powervsx_`
#include "numkong/dots/powermma.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if NUMKONG_TARGET_POWERMMA // Kernels only: an empty pragma region would fail under -Werror
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("cpu=pwr10"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("cpu=power10")
#endif

nk_define_cross_normalized_packed_(angular, f32, powermma, f32, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f32_powermma, nk_angular_through_f64_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_f32_serial_, nk_load_b256_powervsx_,
                                   nk_partial_load_b64x4_powervsx_, nk_store_b256_powervsx_,
                                   nk_partial_store_b64x4_powervsx_, 1)
nk_define_cross_normalized_packed_(euclidean, f32, powermma, f32, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f32_powermma, nk_euclidean_through_f64_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_f32_serial_, nk_load_b256_powervsx_,
                                   nk_partial_load_b64x4_powervsx_, nk_store_b256_powervsx_,
                                   nk_partial_store_b64x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(angular, f32, powermma, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f32_powermma, nk_angular_through_f64_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_f32_serial_, nk_load_b256_powervsx_,
                                      nk_partial_load_b64x4_powervsx_, nk_store_b256_powervsx_,
                                      nk_partial_store_b64x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f32, powermma, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f32_powermma, nk_euclidean_through_f64_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_f32_serial_, nk_load_b256_powervsx_,
                                      nk_partial_load_b64x4_powervsx_, nk_store_b256_powervsx_,
                                      nk_partial_store_b64x4_powervsx_, 1)

nk_define_cross_normalized_packed_(angular, bf16, powermma, bf16, bf16, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_bf16_powermma,
                                   nk_angular_through_f32_from_dot_powervsx_, nk_dots_reduce_sumsq_bf16_serial_,
                                   nk_load_b128_powervsx_, nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                   nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_packed_(euclidean, bf16, powermma, bf16, bf16, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_bf16_powermma,
                                   nk_euclidean_through_f32_from_dot_powervsx_, nk_dots_reduce_sumsq_bf16_serial_,
                                   nk_load_b128_powervsx_, nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                   nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(angular, bf16, powermma, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_powermma, nk_angular_through_f32_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_bf16_serial_, nk_load_b128_powervsx_,
                                      nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                      nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(euclidean, bf16, powermma, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_powermma, nk_euclidean_through_f32_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_bf16_serial_, nk_load_b128_powervsx_,
                                      nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                      nk_partial_store_b32x4_powervsx_, 1)

nk_define_cross_normalized_packed_(angular, f16, powermma, f16, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_powermma, nk_angular_through_f32_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_f16_serial_, nk_load_b128_powervsx_,
                                   nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                   nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_packed_(euclidean, f16, powermma, f16, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_powermma, nk_euclidean_through_f32_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_f16_serial_, nk_load_b128_powervsx_,
                                   nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                   nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(angular, f16, powermma, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_powermma, nk_angular_through_f32_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_f16_serial_, nk_load_b128_powervsx_,
                                      nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                      nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f16, powermma, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_powermma, nk_euclidean_through_f32_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_f16_serial_, nk_load_b128_powervsx_,
                                      nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                      nk_partial_store_b32x4_powervsx_, 1)

nk_define_cross_normalized_packed_(angular, i8, powermma, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_powermma, nk_angular_through_i32_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_i8_serial_, nk_load_b128_powervsx_,
                                   nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                   nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_packed_(euclidean, i8, powermma, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_powermma, nk_euclidean_through_i32_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_i8_serial_, nk_load_b128_powervsx_,
                                   nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                   nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(angular, i8, powermma, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_powermma, nk_angular_through_i32_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_i8_serial_, nk_load_b128_powervsx_,
                                      nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                      nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(euclidean, i8, powermma, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_powermma, nk_euclidean_through_i32_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_i8_serial_, nk_load_b128_powervsx_,
                                      nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                      nk_partial_store_b32x4_powervsx_, 1)

nk_define_cross_normalized_packed_(angular, u8, powermma, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_powermma, nk_angular_through_u32_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_u8_serial_, nk_load_b128_powervsx_,
                                   nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                   nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_packed_(euclidean, u8, powermma, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_powermma, nk_euclidean_through_u32_from_dot_powervsx_,
                                   nk_dots_reduce_sumsq_u8_serial_, nk_load_b128_powervsx_,
                                   nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                   nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(angular, u8, powermma, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_powermma, nk_angular_through_u32_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_u8_serial_, nk_load_b128_powervsx_,
                                      nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                      nk_partial_store_b32x4_powervsx_, 1)
nk_define_cross_normalized_symmetric_(euclidean, u8, powermma, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_powermma, nk_euclidean_through_u32_from_dot_powervsx_,
                                      nk_dots_reduce_sumsq_u8_serial_, nk_load_b128_powervsx_,
                                      nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_,
                                      nk_partial_store_b32x4_powervsx_, 1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NUMKONG_TARGET_POWERMMA

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_PPC64_POWERMMA_
#endif // NUMKONG_ARCH_PPC64_
#endif // NUMKONG_SPATIALS_POWERMMA_H
