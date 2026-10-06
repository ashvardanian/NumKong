/**
 *  @brief SIMD-accelerated Batched Dot Products for NEON FHM.
 *  @file include/numkong/dots/neonfhm.h
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses FMLAL (FEAT_FHM) for widening fp16->f32 multiply-accumulate, which is 20-48% faster
 *  than the convert-then-FMA approach used in neonhalf.h.
 */
#ifndef NK_DOTS_NEONFHM_H
#define NK_DOTS_NEONFHM_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONFHM

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
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f16, neonfhm, f16, f16, nk_assign_from_to_, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_f16_, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)
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
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e4m3, neonfhm, e4m3, e4m3, nk_assign_from_to_, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_e4m3_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
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
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e5m2, neonfhm, e5m2, e5m2, nk_assign_from_to_, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_e5m2_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
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

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONFHM
#endif // NK_TARGET_ARM_
#endif // NK_DOTS_NEONFHM_H
