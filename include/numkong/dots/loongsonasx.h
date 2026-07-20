/**
 *  @brief SIMD-accelerated Batched Dot Products for LoongArch ASX.
 *  @file include/numkong/dots/loongsonasx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/dots.h
 *
 *  GEMM kernels use tiled dot products with 4-way parallel accumulation to hide FMA latency.
 *  LASX is 256-bit, matching AVX2 in register width.
 *  Type-specific tile sizes: f32/f64 use depth_simd_dimensions=4, i8/u8 use depth_simd_dimensions=16.
 */
#ifndef NK_DOTS_LOONGSONASX_H
#define NK_DOTS_LOONGSONASX_H

#if NK_TARGET_LOONGARCH64_
#if NK_TARGET_LOONGSONASX

#include "numkong/dot/loongsonasx.h"
#include "numkong/cast/loongsonasx.h"
#include "numkong/cast.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* F32 GEMM: depth_simd_dimensions=8 (8 f32s = 256-bit input → f64 accumulation via low/high widening) */
nk_define_cross_pack_size_(dots, f32, loongsonasx, f32, f32, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, f32, loongsonasx)
    nk_define_cross_pack_(dots, f32, loongsonasx, f32, f32, nk_b256_vec_t, nk_load_b256_loongsonasx_,
                          nk_partial_load_b32x8_serial_, nk_store_b256_loongsonasx_, nk_partial_store_b32x8_serial_,
                          /*simd_width=*/8, /*norm_value_type=*/f64, nk_dots_reduce_sumsq_f32_,
                          /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f32, loongsonasx, f32, f64, nk_b256_vec_t, nk_dot_f32x8_state_loongsonasx_t,
                           nk_b256_vec_t, nk_dot_f32x8_init_loongsonasx, nk_load_b256_loongsonasx_,
                           nk_partial_load_b32x8_serial_, nk_dot_f32x8_update_loongsonasx,
                           nk_dot_f32x8_finalize_loongsonasx, nk_store_b256_loongsonasx_,
                           nk_partial_store_b64x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f32, loongsonasx, f32, f32, f64, nk_b256_vec_t, nk_dot_f32x8_state_loongsonasx_t,
                        nk_b256_vec_t, nk_dot_f32x8_init_loongsonasx, nk_load_b256_loongsonasx_,
                        nk_partial_load_b32x8_serial_, nk_load_b256_loongsonasx_, nk_partial_load_b32x8_serial_,
                        nk_dot_f32x8_update_loongsonasx, nk_dot_f32x8_finalize_loongsonasx, nk_store_b256_loongsonasx_,
                        nk_partial_store_b64x4_serial_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* F64 GEMM: depth_simd_dimensions=4 (4 f64s = 256-bit = full LASX register) */
nk_define_cross_pack_size_(dots, f64, loongsonasx, f64, f64, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/4,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, f64, loongsonasx)
    nk_define_cross_pack_(dots, f64, loongsonasx, f64, f64, nk_b256_vec_t, nk_load_b256_loongsonasx_,
                          nk_partial_load_b64x4_serial_, nk_store_b256_loongsonasx_, nk_partial_store_b64x4_serial_,
                          /*simd_width=*/4, /*norm_value_type=*/f64, nk_dots_reduce_sumsq_f64_,
                          /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f64, loongsonasx, f64, f64, nk_b256_vec_t, nk_dot_f64x4_state_loongsonasx_t,
                           nk_b256_vec_t, nk_dot_f64x4_init_loongsonasx, nk_load_b256_loongsonasx_,
                           nk_partial_load_b64x4_serial_, nk_dot_f64x4_update_loongsonasx,
                           nk_dot_f64x4_finalize_loongsonasx, nk_store_b256_loongsonasx_,
                           nk_partial_store_b64x4_serial_,
                           /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f64, loongsonasx, f64, f64, f64, nk_b256_vec_t, nk_dot_f64x4_state_loongsonasx_t,
                        nk_b256_vec_t, nk_dot_f64x4_init_loongsonasx, nk_load_b256_loongsonasx_,
                        nk_partial_load_b64x4_serial_, nk_load_b256_loongsonasx_, nk_partial_load_b64x4_serial_,
                        nk_dot_f64x4_update_loongsonasx, nk_dot_f64x4_finalize_loongsonasx, nk_store_b256_loongsonasx_,
                        nk_partial_store_b64x4_serial_,
                        /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)

/* I8 GEMM: depth_simd_dimensions=32 (32 i8s = 256-bit input → i32 accumulation) */
nk_define_cross_pack_size_(dots, i8, loongsonasx, i8, i8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, i8, loongsonasx)
    nk_define_cross_pack_(dots, i8, loongsonasx, i8, i8, nk_b256_vec_t, nk_load_b256_loongsonasx_,
                          nk_partial_load_b8x32_serial_, nk_store_b256_loongsonasx_, nk_partial_store_b8x32_serial_,
                          /*simd_width=*/32, /*norm_value_type=*/u32, nk_dots_reduce_sumsq_i8_,
                          /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, i8, loongsonasx, i8, i32, nk_b256_vec_t, nk_dot_i8x32_state_loongsonasx_t,
                           nk_b128_vec_t, nk_dot_i8x32_init_loongsonasx, nk_load_b256_loongsonasx_,
                           nk_partial_load_b8x32_serial_, nk_dot_i8x32_update_loongsonasx,
                           nk_dot_i8x32_finalize_loongsonasx, nk_store_b128_loongsonasx_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, i8, loongsonasx, i8, i8, i32, nk_b256_vec_t, nk_dot_i8x32_state_loongsonasx_t,
                        nk_b128_vec_t, nk_dot_i8x32_init_loongsonasx, nk_load_b256_loongsonasx_,
                        nk_partial_load_b8x32_serial_, nk_load_b256_loongsonasx_, nk_partial_load_b8x32_serial_,
                        nk_dot_i8x32_update_loongsonasx, nk_dot_i8x32_finalize_loongsonasx, nk_store_b128_loongsonasx_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=32 (32 u8s = 256-bit input → u32 accumulation) */
nk_define_cross_pack_size_(dots, u8, loongsonasx, u8, u8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, u8, loongsonasx)
    nk_define_cross_pack_(dots, u8, loongsonasx, u8, u8, nk_b256_vec_t, nk_load_b256_loongsonasx_,
                          nk_partial_load_b8x32_serial_, nk_store_b256_loongsonasx_, nk_partial_store_b8x32_serial_,
                          /*simd_width=*/32, /*norm_value_type=*/u32, nk_dots_reduce_sumsq_u8_,
                          /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, u8, loongsonasx, u8, u32, nk_b256_vec_t, nk_dot_u8x32_state_loongsonasx_t,
                           nk_b128_vec_t, nk_dot_u8x32_init_loongsonasx, nk_load_b256_loongsonasx_,
                           nk_partial_load_b8x32_serial_, nk_dot_u8x32_update_loongsonasx,
                           nk_dot_u8x32_finalize_loongsonasx, nk_store_b128_loongsonasx_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, u8, loongsonasx, u8, u8, u32, nk_b256_vec_t, nk_dot_u8x32_state_loongsonasx_t,
                        nk_b128_vec_t, nk_dot_u8x32_init_loongsonasx, nk_load_b256_loongsonasx_,
                        nk_partial_load_b8x32_serial_, nk_load_b256_loongsonasx_, nk_partial_load_b8x32_serial_,
                        nk_dot_u8x32_update_loongsonasx, nk_dot_u8x32_finalize_loongsonasx, nk_store_b128_loongsonasx_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/* U1 GEMM: depth_simd_dimensions=256 (256 bits = 32 bytes per tile → u32 popcount via XVPCNT.W) */
nk_define_cross_pack_size_(dots, u1, loongsonasx, u1x8, u1x8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/256,
                           /*dimensions_per_value=*/8)
nk_define_cross_packed_shape_(dots, u1, loongsonasx)
    nk_define_cross_pack_(dots, u1, loongsonasx, u1x8, u1x8, nk_b256_vec_t, nk_load_b256_loongsonasx_,
                          nk_partial_load_b8x32_serial_, nk_store_b256_loongsonasx_, nk_partial_store_b8x32_serial_,
                          /*simd_width=*/32, /*norm_value_type=*/u32, nk_dots_reduce_sum_u1_,
                          /*depth_simd_dimensions=*/256, /*dimensions_per_value=*/8)
nk_define_cross_symmetric_(dots, u1, loongsonasx, u1x8, u32, nk_b256_vec_t, nk_dot_u1x256_state_loongsonasx_t,
                           nk_b128_vec_t, nk_dot_u1x256_init_loongsonasx, nk_load_b256_loongsonasx_,
                           nk_partial_load_b1x256_serial_, nk_dot_u1x256_update_loongsonasx,
                           nk_dot_u1x256_finalize_loongsonasx, nk_store_b128_loongsonasx_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/256, /*dimensions_per_value=*/8)
nk_define_cross_packed_(dots, u1, loongsonasx, u1x8, u1x8, u32, nk_b256_vec_t, nk_dot_u1x256_state_loongsonasx_t,
                        nk_b128_vec_t, nk_dot_u1x256_init_loongsonasx, nk_load_b256_loongsonasx_,
                        nk_partial_load_b1x256_serial_, nk_load_b256_loongsonasx_, nk_partial_load_b1x256_serial_,
                        nk_dot_u1x256_update_loongsonasx, nk_dot_u1x256_finalize_loongsonasx,
                        nk_store_b128_loongsonasx_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/256, /*dimensions_per_value=*/8)

/* BF16 GEMM: depth_simd_dimensions=16, packed stores raw bf16 (conversion is just an interleave).
 *  Both symmetric and packed use the bf16x16 update which converts inline. */
nk_define_cross_pack_size_(dots, bf16, loongsonasx, bf16, bf16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, bf16, loongsonasx)
    nk_define_cross_pack_(dots, bf16, loongsonasx, bf16, bf16, nk_b256_vec_t, nk_load_b256_loongsonasx_,
                          nk_partial_load_b16x16_serial_, nk_store_b256_loongsonasx_, nk_partial_store_b16x16_serial_,
                          /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_bf16_,
                          /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, bf16, loongsonasx, bf16, f32, nk_b256_vec_t, nk_dot_bf16x16_state_loongsonasx_t,
                           nk_b128_vec_t, nk_dot_bf16x16_init_loongsonasx, nk_load_b256_loongsonasx_,
                           nk_partial_load_b16x16_serial_, nk_dot_bf16x16_update_loongsonasx,
                           nk_dot_bf16x16_finalize_loongsonasx, nk_store_b128_loongsonasx_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, bf16, loongsonasx, bf16, bf16, f32, nk_b256_vec_t, nk_dot_bf16x16_state_loongsonasx_t,
                        nk_b128_vec_t, nk_dot_bf16x16_init_loongsonasx, nk_load_b256_loongsonasx_,
                        nk_partial_load_b16x16_serial_, nk_load_b256_loongsonasx_, nk_partial_load_b16x16_serial_,
                        nk_dot_bf16x16_update_loongsonasx, nk_dot_bf16x16_finalize_loongsonasx,
                        nk_store_b128_loongsonasx_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* F16 GEMM: symmetric uses 256-bit raw f16 tiles (depth_simd_dimensions=16) with hardware xvfcvtl/xvfcvth,
 *           packed pre-converts to f32 during packing (depth_simd_dimensions=8) since conversion is expensive. */
nk_define_cross_pack_size_(dots, f16, loongsonasx, f16, f32, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, f16, loongsonasx)
    nk_define_cross_pack_(dots, f16, loongsonasx, f16, f32, nk_b256_vec_t, nk_load_f16x8_to_f32x8_loongsonasx_,
                          nk_partial_load_f16x8_to_f32x8_loongsonasx_, nk_store_b256_loongsonasx_,
                          nk_partial_store_b32x8_serial_, /*simd_width=*/8, /*norm_value_type=*/f32,
                          nk_dots_reduce_sumsq_f16_, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f16, loongsonasx, f16, f32, nk_b256_vec_t, nk_dot_f16x16_state_loongsonasx_t,
                           nk_b128_vec_t, nk_dot_f16x16_init_loongsonasx, nk_load_b256_loongsonasx_,
                           nk_partial_load_b16x16_serial_, nk_dot_f16x16_update_loongsonasx,
                           nk_dot_f16x16_finalize_loongsonasx, nk_store_b128_loongsonasx_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f16, loongsonasx, f16, f32, f32, nk_b256_vec_t, nk_dot_through_f32_state_loongsonasx_t_,
                        nk_b128_vec_t, nk_dot_through_f32_init_loongsonasx_, nk_load_f16x8_to_f32x8_loongsonasx_,
                        nk_partial_load_f16x8_to_f32x8_loongsonasx_, nk_load_b256_loongsonasx_,
                        nk_partial_load_b32x8_serial_, nk_dot_through_f32_update_loongsonasx_,
                        nk_dot_through_f32_finalize_loongsonasx_, nk_store_b128_loongsonasx_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_LOONGSONASX
#endif // NK_TARGET_LOONGARCH64_
#endif // NK_DOTS_LOONGSONASX_H
