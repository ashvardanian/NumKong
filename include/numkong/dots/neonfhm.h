/**
 *  @file include/numkong/dots/neonfhm.h
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *  @brief SIMD-accelerated Batched Dot Products for NEON FHM.
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses FMLAL, FEAT_FHM, for widening fp16 → f32 multiply-accumulate, which is 20-48% faster than
 *  the convert-then-FMA approach used in neonhalf.h.
 */
#ifndef NUMKONG_DOTS_NEONFHM_H
#define NUMKONG_DOTS_NEONFHM_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONFHM

#include "numkong/dots/serial.h"
#include "numkong/dots/neon.h" // `nk_pack_mxfp8_scales_neon_`, `nk_cross_scaled_f16_values_neon_`
#include "numkong/dot/neonfhm.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

/* F16 GEMM using FMLAL: depth_simd_dimensions=8 (8 f16s = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, f16, neonfhm, f16, f16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, f16, neonfhm)
nk_define_cross_pack_(dots, f16, neonfhm, f16, f16, nk_b128_vec_t, nk_load_b128_neon_, nk_partial_load_b16x8_serial_,
                      nk_store_b128_neon_, nk_partial_store_b16x8_serial_,
                      /*simd_width=*/8, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_f16_,
                      /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, f16, neonfhm, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_neonfhm_t, nk_b128_vec_t,
                           nk_dot_f16x8_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b16x8_serial_,
                           nk_dot_f16x8_update_neonfhm, nk_dot_f16x8_finalize_neonfhm, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f16, neonfhm, f16, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_neonfhm_t, nk_b128_vec_t,
                        nk_dot_f16x8_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b16x8_serial_,
                        nk_load_b128_neon_, nk_partial_load_b16x8_serial_, nk_dot_f16x8_update_neonfhm,
                        nk_dot_f16x8_finalize_neonfhm, nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* E4M3 GEMM via FMLAL: depth_simd_dimensions=16 (16 e4m3s = 16 bytes) */
nk_define_cross_pack_size_(dots, e4m3, neonfhm, e4m3, e4m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, e4m3, neonfhm)
nk_define_cross_pack_(dots, e4m3, neonfhm, e4m3, e4m3, nk_b128_vec_t, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_neon_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e4m3_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, e4m3, neonfhm, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x16_state_neonfhm_t, nk_b128_vec_t,
                           nk_dot_e4m3x16_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                           nk_dot_e4m3x16_update_neonfhm, nk_dot_e4m3x16_finalize_neonfhm, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e4m3, neonfhm, e4m3, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x16_state_neonfhm_t,
                        nk_b128_vec_t, nk_dot_e4m3x16_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_neon_, nk_partial_load_b8x16_serial_, nk_dot_e4m3x16_update_neonfhm,
                        nk_dot_e4m3x16_finalize_neonfhm, nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E5M2 GEMM via FMLAL: depth_simd_dimensions=16 (16 e5m2s = 16 bytes) */
nk_define_cross_pack_size_(dots, e5m2, neonfhm, e5m2, e5m2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, e5m2, neonfhm)
nk_define_cross_pack_(dots, e5m2, neonfhm, e5m2, e5m2, nk_b128_vec_t, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_neon_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e5m2_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, e5m2, neonfhm, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x16_state_neonfhm_t, nk_b128_vec_t,
                           nk_dot_e5m2x16_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                           nk_dot_e5m2x16_update_neonfhm, nk_dot_e5m2x16_finalize_neonfhm, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e5m2, neonfhm, e5m2, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x16_state_neonfhm_t,
                        nk_b128_vec_t, nk_dot_e5m2x16_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_neon_, nk_partial_load_b8x16_serial_, nk_dot_e5m2x16_update_neonfhm,
                        nk_dot_e5m2x16_finalize_neonfhm, nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/*  MXFP8 shares the base NEON F16 packs, accumulating each block with FMLAL */
nk_define_cross_pack_size_(dots, mxfp8e4m3, neonfhm, e4m3, f16, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, /*scale_bytes=*/5)
nk_define_cross_packed_shape_(dots, mxfp8e4m3, neonfhm)
nk_define_cross_pack_(dots, mxfp8e4m3, neonfhm, e4m3, f16, nk_b256_vec_t, nk_load_e4m3x16_to_f16x16_neon_,
                      nk_partial_load_e4m3x16_to_f16x16_neon_, nk_store_b256_neon_, nk_partial_store_b16x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e4m3_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, nk_pack_mxfp8_scales_neon_,
                      /*scale_bytes=*/5)
nk_define_cross_scaled_packed_(dots, mxfp8e4m3, neonfhm, f16, nk_dot_scaled_f16x32_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_f16x32_neon_,
                               nk_dot_scaled_f16x32_update_neonfhm, nk_dot_scaled_f32_finalize_neon,
                               nk_f32x4_scale_neon_, nk_cross_scaled_f16_values_neon_, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
    nk_define_cross_scaled_symmetric_(dots, mxfp8e4m3, neonfhm, f16, /*depth_simd_dimensions=*/32,
                                      /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                                      /*register_columns=*/4)

        nk_define_cross_pack_size_(dots, mxfp8e5m2, neonfhm, e5m2, f16, /*norm_value_type=*/f32,
                                   /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, /*scale_bytes=*/5)
nk_define_cross_packed_shape_(dots, mxfp8e5m2, neonfhm)
nk_define_cross_pack_(dots, mxfp8e5m2, neonfhm, e5m2, f16, nk_b256_vec_t, nk_load_e5m2x16_to_f16x16_neon_,
                      nk_partial_load_e5m2x16_to_f16x16_neon_, nk_store_b256_neon_, nk_partial_store_b16x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e5m2_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, nk_pack_mxfp8_scales_neon_,
                      /*scale_bytes=*/5)
nk_define_cross_scaled_packed_(dots, mxfp8e5m2, neonfhm, f16, nk_dot_scaled_f16x32_operand_neon_t,
                               nk_dot_scaled_f32_state_neon_t, nk_dot_scaled_f32_init_neon, nk_load_scaled_f16x32_neon_,
                               nk_dot_scaled_f16x32_update_neonfhm, nk_dot_scaled_f32_finalize_neon,
                               nk_f32x4_scale_neon_, nk_cross_scaled_f16_values_neon_, /*depth_simd_dimensions=*/32,
                               /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                               /*register_columns=*/4)
    nk_define_cross_scaled_symmetric_(dots, mxfp8e5m2, neonfhm, f16, /*depth_simd_dimensions=*/32,
                                      /*dimensions_per_value=*/1, /*scale_bytes=*/5, /*register_rows=*/2,
                                      /*register_columns=*/4)

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
#endif // NUMKONG_DOTS_NEONFHM_H
