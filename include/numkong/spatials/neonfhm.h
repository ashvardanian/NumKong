/**
 *  @file include/numkong/spatials/neonfhm.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *  @brief Batched spatial distances for NEON FHMA.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NUMKONG_SPATIALS_NEONFHM_H
#define NUMKONG_SPATIALS_NEONFHM_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONFHM

#include "numkong/spatial/neon.h"
#include "numkong/dots/neonfhm.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

nk_define_cross_normalized_packed_(angular, f16, neonfhm, f16, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_neonfhm, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f16_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, f16, neonfhm, f16, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_f16_neonfhm, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_f16_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, f16, neonfhm, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_neonfhm, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f16_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, f16, neonfhm, f16, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_f16_neonfhm, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_f16_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, e4m3, neonfhm, e4m3, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e4m3_neonfhm, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e4m3_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e4m3, neonfhm, e4m3, e4m3, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e4m3_neonfhm, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e4m3_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e4m3, neonfhm, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e4m3_neonfhm, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e4m3_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e4m3, neonfhm, e4m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e4m3_neonfhm, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e4m3_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, e5m2, neonfhm, e5m2, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                   nk_dots_packed_e5m2_neonfhm, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e5m2_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e5m2, neonfhm, e5m2, e5m2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e5m2_neonfhm, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e5m2_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e5m2, neonfhm, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e5m2_neonfhm, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e5m2_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e5m2, neonfhm, e5m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e5m2_neonfhm, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e5m2_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)

/*  Block-scaled distances finish in the scaled kernels' epilogue, from relative dots and norms */
nk_define_cross_scaled_packed_(angulars, mxfp8e4m3, neonfhm, f16, nk_dot_scaled_f16x32_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_f16x32_neon_,
                               nk_dot_scaled_f16x32_update_neonfhm, nk_dot_scaled_f32_finalize_neon,
                               nk_angular_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp8e4m3_neon_,
                               nk_angular_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(angulars, mxfp8e4m3, neonfhm, f16, nk_angular_from_wide_f32_serial_, /*normalized=*/1,
                                  /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)
nk_define_cross_scaled_packed_(euclideans, mxfp8e4m3, neonfhm, f16, nk_dot_scaled_f16x32_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_f16x32_neon_,
                               nk_dot_scaled_f16x32_update_neonfhm, nk_dot_scaled_f32_finalize_neon,
                               nk_euclidean_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp8e4m3_neon_,
                               nk_euclidean_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(euclideans, mxfp8e4m3, neonfhm, f16, nk_euclidean_from_wide_f32_serial_,
                                  /*normalized=*/1,
                                  /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)

nk_define_cross_scaled_packed_(angulars, mxfp8e5m2, neonfhm, f16, nk_dot_scaled_f16x32_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_f16x32_neon_,
                               nk_dot_scaled_f16x32_update_neonfhm, nk_dot_scaled_f32_finalize_neon,
                               nk_angular_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp8e5m2_neon_,
                               nk_angular_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(angulars, mxfp8e5m2, neonfhm, f16, nk_angular_from_wide_f32_serial_, /*normalized=*/1,
                                  /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)
nk_define_cross_scaled_packed_(euclideans, mxfp8e5m2, neonfhm, f16, nk_dot_scaled_f16x32_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_f16x32_neon_,
                               nk_dot_scaled_f16x32_update_neonfhm, nk_dot_scaled_f32_finalize_neon,
                               nk_euclidean_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp8e5m2_neon_,
                               nk_euclidean_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(euclideans, mxfp8e5m2, neonfhm, f16, nk_euclidean_from_wide_f32_serial_,
                                  /*normalized=*/1,
                                  /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_NEONFHM
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_SPATIALS_NEONFHM_H
