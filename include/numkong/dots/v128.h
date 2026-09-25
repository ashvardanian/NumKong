/**
 *  @file include/numkong/dots/v128.h
 *  @author Ash Vardanian
 *  @date September 12, 2026
 *  @brief SIMD-accelerated Batched Dot Products for WASM.
 *
 *  @sa include/numkong/dots.h
 *
 *  Widens i8 and u8 to 16-bit lanes and multiplies adjacent pairs with `i32x4.dot_i16x8_s`, so
 *  every integer GEMM is exact without a correction pass; bf16 shifts even and odd halves straight
 *  into f32 lanes.
 */
#ifndef NUMKONG_DOTS_V128_H
#define NUMKONG_DOTS_V128_H

#if NUMKONG_TARGET_V128

#include "numkong/dot/v128.h"
#include "numkong/dots/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

/* BF16 GEMM: depth_simd_dimensions=8, raw bf16 storage, shift+mask even/odd → f32 inline */
nk_define_cross_pack_size_(dots, bf16, v128, bf16, bf16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, bf16, v128)
nk_define_cross_pack_(dots, bf16, v128, bf16, bf16, nk_b128_vec_t, nk_load_b128_v128_, nk_partial_load_b16x8_serial_,
                      nk_store_b128_v128_, nk_partial_store_b16x8_serial_,
                      /*simd_width=*/8, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_bf16_,
                      /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, bf16, v128, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_v128_t, nk_b128_vec_t,
                           nk_dot_bf16x8_init_v128, nk_load_b128_v128_, nk_partial_load_b16x8_serial_,
                           nk_dot_bf16x8_update_v128, nk_dot_bf16x8_finalize_v128, nk_store_b128_v128_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, bf16, v128, bf16, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_v128_t, nk_b128_vec_t,
                        nk_dot_bf16x8_init_v128, nk_load_b128_v128_, nk_partial_load_b16x8_serial_, nk_load_b128_v128_,
                        nk_partial_load_b16x8_serial_, nk_dot_bf16x8_update_v128, nk_dot_bf16x8_finalize_v128,
                        nk_store_b128_v128_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* I8 GEMM: depth_simd_dimensions=16, widened pairwise dots, exact in i32 */
nk_define_cross_pack_size_(dots, i8, v128, i8, i8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, i8, v128)
nk_define_cross_pack_(dots, i8, v128, i8, i8, nk_b128_vec_t, nk_load_b128_v128_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_v128_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/u32, nk_dots_reduce_sumsq_i8_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, i8, v128, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_v128_t, nk_b128_vec_t,
                           nk_dot_i8x16_init_v128, nk_load_b128_v128_, nk_partial_load_b8x16_serial_,
                           nk_dot_i8x16_update_v128, nk_dot_i8x16_finalize_v128, nk_store_b128_v128_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, i8, v128, i8, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_v128_t, nk_b128_vec_t,
                        nk_dot_i8x16_init_v128, nk_load_b128_v128_, nk_partial_load_b8x16_serial_, nk_load_b128_v128_,
                        nk_partial_load_b8x16_serial_, nk_dot_i8x16_update_v128, nk_dot_i8x16_finalize_v128,
                        nk_store_b128_v128_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=16, u16 lanes fit the signed pairwise dot, exact in u32 */
nk_define_cross_pack_size_(dots, u8, v128, u8, u8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, u8, v128)
nk_define_cross_pack_(dots, u8, v128, u8, u8, nk_b128_vec_t, nk_load_b128_v128_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_v128_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/u32, nk_dots_reduce_sumsq_u8_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, u8, v128, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_v128_t, nk_b128_vec_t,
                           nk_dot_u8x16_init_v128, nk_load_b128_v128_, nk_partial_load_b8x16_serial_,
                           nk_dot_u8x16_update_v128, nk_dot_u8x16_finalize_v128, nk_store_b128_v128_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, u8, v128, u8, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_v128_t, nk_b128_vec_t,
                        nk_dot_u8x16_init_v128, nk_load_b128_v128_, nk_partial_load_b8x16_serial_, nk_load_b128_v128_,
                        nk_partial_load_b8x16_serial_, nk_dot_u8x16_update_v128, nk_dot_u8x16_finalize_v128,
                        nk_store_b128_v128_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* U1 GEMM: depth_simd_dimensions=128, one v128 register of bits per step */
nk_define_cross_pack_size_(dots, u1, v128, u1x8, u1x8, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)
nk_define_cross_packed_shape_(dots, u1, v128)
nk_define_cross_pack_(dots, u1, v128, u1x8, u1x8, nk_b128_vec_t, nk_load_b128_v128_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_v128_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/u32, nk_dots_reduce_sum_u1_,
                      /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)
nk_define_cross_symmetric_(dots, u1, v128, u1x8, u32, nk_b128_vec_t, nk_dot_u1x128_state_v128_t, nk_b128_vec_t,
                           nk_dot_u1x128_init_v128, nk_load_b128_v128_, nk_partial_load_b1x128_serial_,
                           nk_dot_u1x128_update_v128, nk_dot_u1x128_finalize_v128, nk_store_b128_v128_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)
nk_define_cross_packed_(dots, u1, v128, u1x8, u1x8, u32, nk_b128_vec_t, nk_dot_u1x128_state_v128_t, nk_b128_vec_t,
                        nk_dot_u1x128_init_v128, nk_load_b128_v128_, nk_partial_load_b1x128_serial_, nk_load_b128_v128_,
                        nk_partial_load_b1x128_serial_, nk_dot_u1x128_update_v128, nk_dot_u1x128_finalize_v128,
                        nk_store_b128_v128_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_V128
#endif // NUMKONG_DOTS_V128_H
