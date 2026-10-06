/**
 *  @brief SIMD-accelerated Batched Dot Products for NEON SDOT.
 *  @file include/numkong/dots/neonsdot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dots.h
 */
#ifndef NK_DOTS_NEONSDOT_H
#define NK_DOTS_NEONSDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONSDOT

#include "numkong/dot/neonsdot.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod")
#endif

/* I8 GEMM: depth_simd_dimensions=16 (16 i8s = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, i8, neonsdot, i8, i8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, i8, neonsdot, i8, i8, nk_assign_from_to_, /*norm_value_type=*/u32, nk_dots_reduce_sumsq_i8_,
                      /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, i8, neonsdot, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_neonsdot_t, nk_b128_vec_t,
                           nk_dot_i8x16_init_neonsdot, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                           nk_dot_i8x16_update_neonsdot, nk_dot_i8x16_finalize_neonsdot, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, i8, neonsdot, i8, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_neonsdot_t, nk_b128_vec_t,
                        nk_dot_i8x16_init_neonsdot, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_neon_, nk_partial_load_b8x16_serial_, nk_dot_i8x16_update_neonsdot,
                        nk_dot_i8x16_finalize_neonsdot, nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=16 (16 u8s = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, u8, neonsdot, u8, u8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, u8, neonsdot, u8, u8, nk_assign_from_to_, /*norm_value_type=*/u32, nk_dots_reduce_sumsq_u8_,
                      /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, u8, neonsdot, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_neonsdot_t, nk_b128_vec_t,
                           nk_dot_u8x16_init_neonsdot, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                           nk_dot_u8x16_update_neonsdot, nk_dot_u8x16_finalize_neonsdot, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, u8, neonsdot, u8, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_neonsdot_t, nk_b128_vec_t,
                        nk_dot_u8x16_init_neonsdot, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_neon_, nk_partial_load_b8x16_serial_, nk_dot_u8x16_update_neonsdot,
                        nk_dot_u8x16_finalize_neonsdot, nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* I4 GEMM: depth_simd_dimensions=32 (32 nibbles = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, i4, neonsdot, i4x2, i4x2, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/2)
nk_define_cross_pack_(dots, i4, neonsdot, i4x2, i4x2, nk_assign_from_to_, /*norm_value_type=*/u32,
                      nk_dots_reduce_sumsq_i4_, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_symmetric_(dots, i4, neonsdot, i4x2, i32, nk_b128_vec_t, nk_dot_i4x32_state_neonsdot_t, nk_b128_vec_t,
                           nk_dot_i4x32_init_neonsdot, nk_load_b128_neon_, nk_partial_load_b4x32_serial_,
                           nk_dot_i4x32_update_neonsdot, nk_dot_i4x32_finalize_neonsdot, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, i4, neonsdot, i4x2, i4x2, i32, nk_b128_vec_t, nk_dot_i4x32_state_neonsdot_t,
                        nk_b128_vec_t, nk_dot_i4x32_init_neonsdot, nk_load_b128_neon_, nk_partial_load_b4x32_serial_,
                        nk_load_b128_neon_, nk_partial_load_b4x32_serial_, nk_dot_i4x32_update_neonsdot,
                        nk_dot_i4x32_finalize_neonsdot, nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)

/* U4 GEMM: depth_simd_dimensions=32 (32 nibbles = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, u4, neonsdot, u4x2, u4x2, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/2)
nk_define_cross_pack_(dots, u4, neonsdot, u4x2, u4x2, nk_assign_from_to_, /*norm_value_type=*/u32,
                      nk_dots_reduce_sumsq_u4_, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_symmetric_(dots, u4, neonsdot, u4x2, u32, nk_b128_vec_t, nk_dot_u4x32_state_neonsdot_t, nk_b128_vec_t,
                           nk_dot_u4x32_init_neonsdot, nk_load_b128_neon_, nk_partial_load_b4x32_serial_,
                           nk_dot_u4x32_update_neonsdot, nk_dot_u4x32_finalize_neonsdot, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, u4, neonsdot, u4x2, u4x2, u32, nk_b128_vec_t, nk_dot_u4x32_state_neonsdot_t,
                        nk_b128_vec_t, nk_dot_u4x32_init_neonsdot, nk_load_b128_neon_, nk_partial_load_b4x32_serial_,
                        nk_load_b128_neon_, nk_partial_load_b4x32_serial_, nk_dot_u4x32_update_neonsdot,
                        nk_dot_u4x32_finalize_neonsdot, nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONSDOT
#endif // NK_TARGET_ARM_
#endif // NK_DOTS_NEONSDOT_H
