/**
 *  @file include/numkong/spatials/neon.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *  @brief Batched spatial distances for NEON.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NUMKONG_SPATIALS_NEON_H
#define NUMKONG_SPATIALS_NEON_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_NEON_

#include "numkong/spatial/neon.h"
#include "numkong/dots/neon.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

#if NUMKONG_TARGET_NEON
nk_define_cross_normalized_packed_(angular, f32, neon, f32, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f32_neon, nk_angular_through_f64_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f32_serial_, nk_load_b256_neon_, nk_partial_load_b64x4_serial_,
                                   nk_store_b256_neon_, nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f32, neon, f32, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f32_neon, nk_euclidean_through_f64_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f32_serial_, nk_load_b256_neon_, nk_partial_load_b64x4_serial_,
                                   nk_store_b256_neon_, nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f32, neon, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f32_neon, nk_angular_through_f64_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f32_serial_, nk_load_b256_neon_,
                                      nk_partial_load_b64x4_serial_, nk_store_b256_neon_,
                                      nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f32, neon, f32, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f32_neon, nk_euclidean_through_f64_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f32_serial_, nk_load_b256_neon_,
                                      nk_partial_load_b64x4_serial_, nk_store_b256_neon_,
                                      nk_partial_store_b64x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, bf16, neon, bf16, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_bf16_neon, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_bf16_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, bf16, neon, bf16, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_bf16_neon, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_bf16_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, bf16, neon, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_neon, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_bf16_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, bf16, neon, bf16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_bf16_neon, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_bf16_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, f16, neon, f16, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_neon, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f16_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f16, neon, f16, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_neon, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f16_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f16, neon, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_neon, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f16_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f16, neon, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_neon, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f16_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, f64, neon, f64, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f64_neon, nk_angular_through_f64_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f64_serial_, nk_load_b256_neon_, nk_partial_load_b64x4_serial_,
                                   nk_store_b256_neon_, nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f64, neon, f64, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                   nk_dots_packed_f64_neon, nk_euclidean_through_f64_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f64_serial_, nk_load_b256_neon_, nk_partial_load_b64x4_serial_,
                                   nk_store_b256_neon_, nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f64, neon, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f64_neon, nk_angular_through_f64_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f64_serial_, nk_load_b256_neon_,
                                      nk_partial_load_b64x4_serial_, nk_store_b256_neon_,
                                      nk_partial_store_b64x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f64, neon, f64, f64, /*norm_value_type=*/f64, f64, nk_b256_vec_t,
                                      nk_dots_symmetric_f64_neon, nk_euclidean_through_f64_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f64_serial_, nk_load_b256_neon_,
                                      nk_partial_load_b64x4_serial_, nk_store_b256_neon_,
                                      nk_partial_store_b64x4_serial_, 1)

/*  Block-scaled distances finish in the scaled kernels' epilogue, from relative dots and norms */
nk_define_cross_scaled_packed_(angulars, nvfp4, neon, u16, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_angular_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_nvfp4_neon_,
                               nk_angular_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/2, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(angulars, nvfp4, neon, u16, nk_angular_from_wide_f32_serial_, /*normalized=*/1,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)
nk_define_cross_scaled_packed_(euclideans, nvfp4, neon, u16, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_euclidean_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_nvfp4_neon_,
                               nk_euclidean_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/2, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(euclideans, nvfp4, neon, u16, nk_euclidean_from_wide_f32_serial_, /*normalized=*/1,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)

nk_define_cross_scaled_packed_(angulars, mxfp4, neon, u16, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_angular_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp4_neon_,
                               nk_angular_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/2, /*scale_bytes=*/9, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(angulars, mxfp4, neon, u16, nk_angular_from_wide_f32_serial_, /*normalized=*/1,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, /*scale_bytes=*/9,
                                  /*register_rows=*/2, /*register_columns=*/4)
nk_define_cross_scaled_packed_(euclideans, mxfp4, neon, u16, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_euclidean_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp4_neon_,
                               nk_euclidean_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/2, /*scale_bytes=*/9, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(euclideans, mxfp4, neon, u16, nk_euclidean_from_wide_f32_serial_, /*normalized=*/1,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, /*scale_bytes=*/9,
                                  /*register_rows=*/2, /*register_columns=*/4)

nk_define_cross_scaled_packed_(angulars, mxfp8e4m3, neon, f16, nk_dot_scaled_f16x32_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_f16x32_neon_,
                               nk_dot_scaled_f16x32_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_angular_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp8e4m3_neon_,
                               nk_angular_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(angulars, mxfp8e4m3, neon, f16, nk_angular_from_wide_f32_serial_, /*normalized=*/1,
                                  /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)
nk_define_cross_scaled_packed_(euclideans, mxfp8e4m3, neon, f16, nk_dot_scaled_f16x32_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_f16x32_neon_,
                               nk_dot_scaled_f16x32_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_euclidean_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp8e4m3_neon_,
                               nk_euclidean_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(euclideans, mxfp8e4m3, neon, f16, nk_euclidean_from_wide_f32_serial_,
                                  /*normalized=*/1,
                                  /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)

nk_define_cross_scaled_packed_(angulars, mxfp8e5m2, neon, f16, nk_dot_scaled_f16x32_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_f16x32_neon_,
                               nk_dot_scaled_f16x32_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_angular_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp8e5m2_neon_,
                               nk_angular_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(angulars, mxfp8e5m2, neon, f16, nk_angular_from_wide_f32_serial_, /*normalized=*/1,
                                  /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)
nk_define_cross_scaled_packed_(euclideans, mxfp8e5m2, neon, f16, nk_dot_scaled_f16x32_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_f16x32_neon_,
                               nk_dot_scaled_f16x32_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_euclidean_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp8e5m2_neon_,
                               nk_euclidean_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(euclideans, mxfp8e5m2, neon, f16, nk_euclidean_from_wide_f32_serial_,
                                  /*normalized=*/1,
                                  /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)

nk_define_cross_scaled_packed_(angulars, mxfp6e2m3, neon, i8, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_angular_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp6e2m3_neon_,
                               nk_angular_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/9, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(angulars, mxfp6e2m3, neon, i8, nk_angular_from_wide_f32_serial_, /*normalized=*/1,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1, /*scale_bytes=*/9,
                                  /*register_rows=*/2, /*register_columns=*/4)
nk_define_cross_scaled_packed_(euclideans, mxfp6e2m3, neon, i8, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_euclidean_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp6e2m3_neon_,
                               nk_euclidean_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/9, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(euclideans, mxfp6e2m3, neon, i8, nk_euclidean_from_wide_f32_serial_, /*normalized=*/1,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1, /*scale_bytes=*/9,
                                  /*register_rows=*/2, /*register_columns=*/4)

nk_define_cross_scaled_packed_(angulars, mxfp6e3m2, neon, i16, nk_dot_scaled_i16x32_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i16x32_neon_,
                               nk_dot_scaled_i16x32_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_angular_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp6e3m2_neon_,
                               nk_angular_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(angulars, mxfp6e3m2, neon, i16, nk_angular_from_wide_f32_serial_, /*normalized=*/1,
                                  /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)
nk_define_cross_scaled_packed_(euclideans, mxfp6e3m2, neon, i16, nk_dot_scaled_i16x32_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i16x32_neon_,
                               nk_dot_scaled_i16x32_update_neon, nk_dot_scaled_f32_finalize_neon,
                               nk_euclidean_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp6e3m2_neon_,
                               nk_euclidean_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(euclideans, mxfp6e3m2, neon, i16, nk_euclidean_from_wide_f32_serial_,
                                  /*normalized=*/1,
                                  /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)
#endif // NUMKONG_TARGET_NEON

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ARM64_NEON_
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_SPATIALS_NEON_H
