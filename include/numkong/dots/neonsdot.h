/**
 *  @file include/numkong/dots/neonsdot.h
 *  @author Ash Vardanian
 *  @date September 14, 2024
 *  @brief SIMD-accelerated Batched Dot Products for NEON SDOT.
 *
 *  @sa include/numkong/dots.h
 */
#ifndef NUMKONG_DOTS_NEONSDOT_H
#define NUMKONG_DOTS_NEONSDOT_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONSDOT

#include "numkong/dots/serial.h"
#include "numkong/dots/neon.h" // `nk_pack_nvfp4_scales_neon_`, `nk_cross_scaled_e2m1_lanes_values_neon_`
#include "numkong/cast/neon.h" // `nk_load_b128_neon_`, `nk_store_b128_neon_`
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
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, i8, neonsdot)
nk_define_cross_pack_(dots, i8, neonsdot, i8, i8, nk_b128_vec_t, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_neon_, nk_partial_store_b8x16_serial_, /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sumsq_i8_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_, /*scale_bytes=*/1)
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
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, u8, neonsdot)
nk_define_cross_pack_(dots, u8, neonsdot, u8, u8, nk_b128_vec_t, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_neon_, nk_partial_store_b8x16_serial_, /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sumsq_u8_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_, /*scale_bytes=*/1)
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
                           /*dimensions_per_value=*/2, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, i4, neonsdot)
nk_define_cross_pack_(dots, i4, neonsdot, i4x2, i4x2, nk_b128_vec_t, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_neon_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/u32, nk_dots_reduce_sumsq_i4_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
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
                           /*dimensions_per_value=*/2, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, u4, neonsdot)
nk_define_cross_pack_(dots, u4, neonsdot, u4x2, u4x2, nk_b128_vec_t, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_neon_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/u32, nk_dots_reduce_sumsq_u4_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
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

/* E2M3: depth_simd_dimensions=16 (16 e2m3 values = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, e2m3, neonsdot, e2m3, e2m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, e2m3, neonsdot)
nk_define_cross_pack_(dots, e2m3, neonsdot, e2m3, e2m3, nk_b128_vec_t, nk_load_b128_neon_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_neon_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m3_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, e2m3, neonsdot, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x16_state_neonsdot_t,
                           nk_b128_vec_t, nk_dot_e2m3x16_init_neonsdot, nk_load_b128_neon_,
                           nk_partial_load_b8x16_serial_, nk_dot_e2m3x16_update_neonsdot,
                           nk_dot_e2m3x16_finalize_neonsdot, nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e2m3, neonsdot, e2m3, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x16_state_neonsdot_t,
                        nk_b128_vec_t, nk_dot_e2m3x16_init_neonsdot, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_neon_, nk_partial_load_b8x16_serial_, nk_dot_e2m3x16_update_neonsdot,
                        nk_dot_e2m3x16_finalize_neonsdot, nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E2M1: depth_simd_dimensions=32 (16 bytes = 32 nibbles = NEON register width) */
nk_define_cross_pack_size_(dots, e2m1, neonsdot, e2m1x2, e2m1x2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/2, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, e2m1, neonsdot)
nk_define_cross_pack_(dots, e2m1, neonsdot, e2m1x2, e2m1x2, nk_b128_vec_t, nk_load_b128_neon_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_neon_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m1_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, e2m1, neonsdot, e2m1x2, f32, nk_b128_vec_t, nk_dot_e2m1x32_state_neonsdot_t,
                           nk_b128_vec_t, nk_dot_e2m1x32_init_neonsdot, nk_load_b128_neon_,
                           nk_partial_load_b4x32_serial_, nk_dot_e2m1x32_update_neonsdot,
                           nk_dot_e2m1x32_finalize_neonsdot, nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, e2m1, neonsdot, e2m1x2, e2m1x2, f32, nk_b128_vec_t, nk_dot_e2m1x32_state_neonsdot_t,
                        nk_b128_vec_t, nk_dot_e2m1x32_init_neonsdot, nk_load_b128_neon_, nk_partial_load_b4x32_serial_,
                        nk_load_b128_neon_, nk_partial_load_b4x32_serial_, nk_dot_e2m1x32_update_neonsdot,
                        nk_dot_e2m1x32_finalize_neonsdot, nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)

/* E3M2: depth_simd_dimensions=16 (16 e3m2 values = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, e3m2, neonsdot, e3m2, e3m2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, e3m2, neonsdot)
nk_define_cross_pack_(dots, e3m2, neonsdot, e3m2, e3m2, nk_b128_vec_t, nk_load_b128_neon_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_neon_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e3m2_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, e3m2, neonsdot, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x16_state_neonsdot_t,
                           nk_b128_vec_t, nk_dot_e3m2x16_init_neonsdot, nk_load_b128_neon_,
                           nk_partial_load_b8x16_serial_, nk_dot_e3m2x16_update_neonsdot,
                           nk_dot_e3m2x16_finalize_neonsdot, nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e3m2, neonsdot, e3m2, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x16_state_neonsdot_t,
                        nk_b128_vec_t, nk_dot_e3m2x16_init_neonsdot, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_neon_, nk_partial_load_b8x16_serial_, nk_dot_e3m2x16_update_neonsdot,
                        nk_dot_e3m2x16_finalize_neonsdot, nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/*  NVFP4, MXFP4 and MXFP6 E2M3 share the base NEON packs and lane layout, with 4 SDOT per step */
nk_define_cross_pack_size_(dots, nvfp4, neonsdot, e2m1x2, u16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/2, /*scale_bytes=*/5)
nk_define_cross_packed_shape_(dots, nvfp4, neonsdot)
nk_define_cross_pack_(dots, nvfp4, neonsdot, e2m1x2, u16, nk_b512_vec_t, nk_e2m1x64_to_i8x64_neon_,
                      nk_partial_e2m1x64_to_i8x64_neon_, nk_store_b512_neon_, nk_partial_store_i8x64_neon_,
                      /*simd_width=*/32, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m1_,
                      /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, nk_pack_nvfp4_scales_neon_,
                      /*scale_bytes=*/5)
nk_define_cross_scaled_packed_(dots, nvfp4, neonsdot, u16, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neonsdot, nk_dot_scaled_f32_finalize_neon,
                               nk_f32x4_scale_neon_, nk_cross_scaled_e2m1_lanes_values_neon_,
                               /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, /*scale_bytes=*/5,
                               /*register_rows=*/2, /*register_columns=*/4)
    nk_define_cross_scaled_symmetric_(dots, nvfp4, neonsdot, u16, /*depth_simd_dimensions=*/64,
                                      /*dimensions_per_value=*/2, /*scale_bytes=*/5, /*register_rows=*/2,
                                      /*register_columns=*/4)

        nk_define_cross_pack_size_(dots, mxfp4, neonsdot, e2m1x2, u16, /*norm_value_type=*/f32,
                                   /*depth_simd_dimensions=*/64,
                                   /*dimensions_per_value=*/2, /*scale_bytes=*/9)
nk_define_cross_packed_shape_(dots, mxfp4, neonsdot)
nk_define_cross_pack_(dots, mxfp4, neonsdot, e2m1x2, u16, nk_b512_vec_t, nk_e2m1x64_to_i8x64_neon_,
                      nk_partial_e2m1x64_to_i8x64_neon_, nk_store_b512_neon_, nk_partial_store_i8x64_neon_,
                      /*simd_width=*/32, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m1_,
                      /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, nk_pack_mxfp4_scales_neon_,
                      /*scale_bytes=*/9)
nk_define_cross_scaled_packed_(dots, mxfp4, neonsdot, u16, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neonsdot, nk_dot_scaled_f32_finalize_neon,
                               nk_f32x4_scale_neon_, nk_cross_scaled_e2m1_lanes_values_neon_,
                               /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, /*scale_bytes=*/9,
                               /*register_rows=*/2, /*register_columns=*/4)
    nk_define_cross_scaled_symmetric_(dots, mxfp4, neonsdot, u16, /*depth_simd_dimensions=*/64,
                                      /*dimensions_per_value=*/2, /*scale_bytes=*/9, /*register_rows=*/2,
                                      /*register_columns=*/4)

        nk_define_cross_pack_size_(dots, mxfp6e2m3, neonsdot, e2m3, i8, /*norm_value_type=*/f32,
                                   /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1, /*scale_bytes=*/9)
nk_define_cross_packed_shape_(dots, mxfp6e2m3, neonsdot)
nk_define_cross_pack_(dots, mxfp6e2m3, neonsdot, e2m3, i8, nk_b512_vec_t, nk_e2m3x64_to_i8x64_neon_,
                      nk_partial_e2m3x64_to_i8x64_neon_, nk_store_b512_neon_, nk_partial_store_i8x64_neon_,
                      /*simd_width=*/64, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m3_,
                      /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1, nk_pack_mxfp6e2m3_scales_neon_,
                      /*scale_bytes=*/9)
nk_define_cross_scaled_packed_(dots, mxfp6e2m3, neonsdot, i8, nk_dot_scaled_i8x64_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_i8x64_neon_,
                               nk_dot_scaled_i8x64_update_neonsdot, nk_dot_scaled_f32_finalize_neon,
                               nk_f32x4_scale_neon_, nk_cross_scaled_e2m3_lanes_values_neon_,
                               /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1, /*scale_bytes=*/9,
                               /*register_rows=*/2, /*register_columns=*/4)
    nk_define_cross_scaled_symmetric_(dots, mxfp6e2m3, neonsdot, i8, /*depth_simd_dimensions=*/64,
                                      /*dimensions_per_value=*/1, /*scale_bytes=*/9, /*register_rows=*/2,
                                      /*register_columns=*/4)

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
#endif // NUMKONG_DOTS_NEONSDOT_H
