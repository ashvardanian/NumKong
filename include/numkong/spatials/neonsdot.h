/**
 *  @file include/numkong/spatials/neonsdot.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *  @brief Batched spatial distances for NEON signed/unsigned dot product.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NUMKONG_SPATIALS_NEONSDOT_H
#define NUMKONG_SPATIALS_NEONSDOT_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONSDOT

#include "numkong/spatial/neon.h"
#include "numkong/dots/neonsdot.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod")
#endif

nk_define_cross_normalized_packed_(angular, i8, neonsdot, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_neonsdot, nk_angular_through_i32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_i8_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, i8, neonsdot, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_neonsdot, nk_euclidean_through_i32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_i8_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, i8, neonsdot, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_neonsdot, nk_angular_through_i32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_i8_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, i8, neonsdot, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_neonsdot, nk_euclidean_through_i32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_i8_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, u8, neonsdot, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_neonsdot, nk_angular_through_u32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_u8_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, u8, neonsdot, u8, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u8_neonsdot, nk_euclidean_through_u32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_u8_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, u8, neonsdot, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_neonsdot, nk_angular_through_u32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_u8_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, u8, neonsdot, u8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u8_neonsdot, nk_euclidean_through_u32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_u8_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, i4, neonsdot, i4x2, i4x2, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i4_neonsdot, nk_angular_through_i32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_i4_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_packed_(euclidean, i4, neonsdot, i4x2, i4x2, i32, /*norm_value_type=*/u32, f32,
                                   nk_b128_vec_t, nk_dots_packed_i4_neonsdot, nk_euclidean_through_i32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_i4_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_symmetric_(angular, i4, neonsdot, i4x2, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i4_neonsdot, nk_angular_through_i32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_i4_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_symmetric_(euclidean, i4, neonsdot, i4x2, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i4_neonsdot, nk_euclidean_through_i32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_i4_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 2)

nk_define_cross_normalized_packed_(angular, u4, neonsdot, u4x2, u4x2, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u4_neonsdot, nk_angular_through_u32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_u4_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_packed_(euclidean, u4, neonsdot, u4x2, u4x2, u32, /*norm_value_type=*/u32, f32,
                                   nk_b128_vec_t, nk_dots_packed_u4_neonsdot, nk_euclidean_through_u32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_u4_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_symmetric_(angular, u4, neonsdot, u4x2, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u4_neonsdot, nk_angular_through_u32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_u4_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_symmetric_(euclidean, u4, neonsdot, u4x2, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u4_neonsdot, nk_euclidean_through_u32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_u4_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 2)

nk_define_cross_normalized_packed_(angular, e2m3, neonsdot, e2m3, e2m3, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e2m3_neonsdot, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e2m3_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e2m3, neonsdot, e2m3, e2m3, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e2m3_neonsdot, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e2m3_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e2m3, neonsdot, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m3_neonsdot, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e2m3_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e2m3, neonsdot, e2m3, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m3_neonsdot, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e2m3_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)

nk_define_cross_normalized_packed_(angular, e2m1, neonsdot, e2m1x2, e2m1x2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e2m1_neonsdot, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e2m1_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_packed_(euclidean, e2m1, neonsdot, e2m1x2, e2m1x2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e2m1_neonsdot, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e2m1_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_symmetric_(angular, e2m1, neonsdot, e2m1x2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e2m1_neonsdot, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e2m1_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 2)
nk_define_cross_normalized_symmetric_(euclidean, e2m1, neonsdot, e2m1x2, f32, /*norm_value_type=*/f32, f32,
                                      nk_b128_vec_t, nk_dots_symmetric_e2m1_neonsdot,
                                      nk_euclidean_through_f32_from_dot_neon_, nk_dots_reduce_sumsq_e2m1_serial_,
                                      nk_load_b128_neon_, nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 2)

nk_define_cross_normalized_packed_(angular, e3m2, neonsdot, e3m2, e3m2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e3m2_neonsdot, nk_angular_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e3m2_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_packed_(euclidean, e3m2, neonsdot, e3m2, e3m2, f32, /*norm_value_type=*/f32, f32,
                                   nk_b128_vec_t, nk_dots_packed_e3m2_neonsdot, nk_euclidean_through_f32_from_dot_neon_,
                                   nk_dots_reduce_sumsq_e3m2_serial_, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, e3m2, neonsdot, e3m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e3m2_neonsdot, nk_angular_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e3m2_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(euclidean, e3m2, neonsdot, e3m2, f32, /*norm_value_type=*/f32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_e3m2_neonsdot, nk_euclidean_through_f32_from_dot_neon_,
                                      nk_dots_reduce_sumsq_e3m2_serial_, nk_load_b128_neon_,
                                      nk_partial_load_b32x4_serial_, nk_store_b128_neon_,
                                      nk_partial_store_b32x4_serial_, 1)

/*  Block-scaled distances finish in the scaled kernels' epilogue, from relative dots and norms */
nk_define_cross_scaled_packed_(angulars, nvfp4, neonsdot, u16, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neonsdot, nk_dot_scaled_f32_finalize_neon,
                               nk_angular_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_nvfp4_neon_,
                               nk_angular_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/2, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(angulars, nvfp4, neonsdot, u16, nk_angular_from_wide_f32_serial_, /*normalized=*/1,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)
nk_define_cross_scaled_packed_(euclideans, nvfp4, neonsdot, u16, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neonsdot, nk_dot_scaled_f32_finalize_neon,
                               nk_euclidean_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_nvfp4_neon_,
                               nk_euclidean_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/2, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(euclideans, nvfp4, neonsdot, u16, nk_euclidean_from_wide_f32_serial_,
                                  /*normalized=*/1,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, /*scale_bytes=*/5,
                                  /*register_rows=*/2, /*register_columns=*/4)

nk_define_cross_scaled_packed_(angulars, mxfp4, neonsdot, u16, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neonsdot, nk_dot_scaled_f32_finalize_neon,
                               nk_angular_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp4_neon_,
                               nk_angular_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/2, /*scale_bytes=*/9, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(angulars, mxfp4, neonsdot, u16, nk_angular_from_wide_f32_serial_, /*normalized=*/1,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, /*scale_bytes=*/9,
                                  /*register_rows=*/2, /*register_columns=*/4)
nk_define_cross_scaled_packed_(euclideans, mxfp4, neonsdot, u16, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neonsdot, nk_dot_scaled_f32_finalize_neon,
                               nk_euclidean_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp4_neon_,
                               nk_euclidean_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/2, /*scale_bytes=*/9, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(euclideans, mxfp4, neonsdot, u16, nk_euclidean_from_wide_f32_serial_,
                                  /*normalized=*/1,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, /*scale_bytes=*/9,
                                  /*register_rows=*/2, /*register_columns=*/4)

nk_define_cross_scaled_packed_(angulars, mxfp6e2m3, neonsdot, i8, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neonsdot, nk_dot_scaled_f32_finalize_neon,
                               nk_angular_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp6e2m3_neon_,
                               nk_angular_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/9, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(angulars, mxfp6e2m3, neonsdot, i8, nk_angular_from_wide_f32_serial_, /*normalized=*/1,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1, /*scale_bytes=*/9,
                                  /*register_rows=*/2, /*register_columns=*/4)
nk_define_cross_scaled_packed_(euclideans, mxfp6e2m3, neonsdot, i8, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neonsdot, nk_dot_scaled_f32_finalize_neon,
                               nk_euclidean_f32x4_from_relative_neon_, nk_cross_scaled_exact_wide_mxfp6e2m3_neon_,
                               nk_euclidean_from_wide_f32_serial_, /*normalized=*/1, /*depth_simd_dimensions=*/64,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/9, /*register_rows=*/2,
                               /*register_columns=*/4)
nk_define_cross_scaled_symmetric_(euclideans, mxfp6e2m3, neonsdot, i8, nk_euclidean_from_wide_f32_serial_,
                                  /*normalized=*/1,
                                  /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1, /*scale_bytes=*/9,
                                  /*register_rows=*/2, /*register_columns=*/4)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_NEONSDOT
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_SPATIALS_NEONSDOT_H
