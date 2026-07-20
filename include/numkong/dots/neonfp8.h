/**
 *  @brief SIMD-accelerated Batched Dot Products for NEON FP8DOT4.
 *  @file include/numkong/dots/neonfp8.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses FDOT (FEAT_FP8DOT4) for native FP8 4-way dot products accumulating into FP32.
 *  Each FDOT processes 16 FP8 elements (128-bit register) into 4 FP32 accumulators.
 */
#ifndef NK_DOTS_NEONFP8_H
#define NK_DOTS_NEONFP8_H

#if NK_TARGET_ARM64_
#if NK_TARGET_NEONFP8

#include "numkong/dot/neonfp8.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd+fp8dot4"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd+fp8dot4")
#endif

nk_define_cross_pack_size_(dots, e4m3, neonfp8, e4m3, e4m3, f32, 16, 1)
nk_define_cross_packed_shape_(dots, e4m3, neonfp8)
    nk_define_cross_pack_(dots, e4m3, neonfp8, e4m3, e4m3, nk_b128_vec_t, nk_load_b128_neon_,
                          nk_partial_load_b8x16_serial_, nk_store_b128_neon_, nk_partial_store_b8x16_serial_,
                          /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e4m3_,
                          /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e4m3, neonfp8, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x16_state_neonfp8_t, nk_b128_vec_t,
                           nk_dot_e4m3x16_init_neonfp8, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                           nk_dot_e4m3x16_update_neonfp8, nk_dot_e4m3x16_finalize_neonfp8, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_, 16, 1)
nk_define_cross_packed_(dots, e4m3, neonfp8, e4m3, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x16_state_neonfp8_t,
                        nk_b128_vec_t, nk_dot_e4m3x16_init_neonfp8, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_neon_, nk_partial_load_b8x16_serial_, nk_dot_e4m3x16_update_neonfp8,
                        nk_dot_e4m3x16_finalize_neonfp8, nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 16, 1)

nk_define_cross_pack_size_(dots, e5m2, neonfp8, e5m2, e5m2, f32, 16, 1)
nk_define_cross_packed_shape_(dots, e5m2, neonfp8)
    nk_define_cross_pack_(dots, e5m2, neonfp8, e5m2, e5m2, nk_b128_vec_t, nk_load_b128_neon_,
                          nk_partial_load_b8x16_serial_, nk_store_b128_neon_, nk_partial_store_b8x16_serial_,
                          /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e5m2_,
                          /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e5m2, neonfp8, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x16_state_neonfp8_t, nk_b128_vec_t,
                           nk_dot_e5m2x16_init_neonfp8, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                           nk_dot_e5m2x16_update_neonfp8, nk_dot_e5m2x16_finalize_neonfp8, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_, 16, 1)
nk_define_cross_packed_(dots, e5m2, neonfp8, e5m2, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x16_state_neonfp8_t,
                        nk_b128_vec_t, nk_dot_e5m2x16_init_neonfp8, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_neon_, nk_partial_load_b8x16_serial_, nk_dot_e5m2x16_update_neonfp8,
                        nk_dot_e5m2x16_finalize_neonfp8, nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 16, 1)

nk_define_cross_pack_size_(dots, e2m3, neonfp8, e2m3, e2m3, f32, 16, 1)
nk_define_cross_packed_shape_(dots, e2m3, neonfp8)
    nk_define_cross_pack_(dots, e2m3, neonfp8, e2m3, e2m3, nk_b128_vec_t, nk_load_b128_neon_,
                          nk_partial_load_b8x16_serial_, nk_store_b128_neon_, nk_partial_store_b8x16_serial_,
                          /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m3_,
                          /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e2m3, neonfp8, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x16_state_neonfp8_t, nk_b128_vec_t,
                           nk_dot_e2m3x16_init_neonfp8, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                           nk_dot_e2m3x16_update_neonfp8, nk_dot_e2m3x16_finalize_neonfp8, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_, 16, 1)
nk_define_cross_packed_(dots, e2m3, neonfp8, e2m3, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x16_state_neonfp8_t,
                        nk_b128_vec_t, nk_dot_e2m3x16_init_neonfp8, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_neon_, nk_partial_load_b8x16_serial_, nk_dot_e2m3x16_update_neonfp8,
                        nk_dot_e2m3x16_finalize_neonfp8, nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 16, 1)

nk_define_cross_pack_size_(dots, e3m2, neonfp8, e3m2, e3m2, f32, 16, 1)
nk_define_cross_packed_shape_(dots, e3m2, neonfp8)
    nk_define_cross_pack_(dots, e3m2, neonfp8, e3m2, e3m2, nk_b128_vec_t, nk_load_b128_neon_,
                          nk_partial_load_b8x16_serial_, nk_store_b128_neon_, nk_partial_store_b8x16_serial_,
                          /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e3m2_,
                          /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e3m2, neonfp8, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x16_state_neonfp8_t, nk_b128_vec_t,
                           nk_dot_e3m2x16_init_neonfp8, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                           nk_dot_e3m2x16_update_neonfp8, nk_dot_e3m2x16_finalize_neonfp8, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_, 16, 1)
nk_define_cross_packed_(dots, e3m2, neonfp8, e3m2, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x16_state_neonfp8_t,
                        nk_b128_vec_t, nk_dot_e3m2x16_init_neonfp8, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_neon_, nk_partial_load_b8x16_serial_, nk_dot_e3m2x16_update_neonfp8,
                        nk_dot_e3m2x16_finalize_neonfp8, nk_store_b128_neon_, nk_partial_store_b32x4_serial_, 16, 1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONFP8
#endif // NK_TARGET_ARM64_
#endif // NK_DOTS_NEONFP8_H
