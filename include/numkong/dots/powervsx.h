/**
 *  @brief SIMD-accelerated Batched Dot Products for Power ISA VSX.
 *  @file include/numkong/dots/powervsx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/dots.h
 *
 *  @section powervsx_dots_instructions Key Power9 VSX GEMM Instructions
 *
 *      Intrinsic      Instruction            P9
 *      vec_madd       xvmaddXsp/dp           6cy @ 2p    fused multiply-add
 *      vec_mul        xvmulXsp/dp            6cy @ 2p
 *      vec_add        xvaddXsp/dp            6cy @ 2p
 *      vec_doublee    xvcvspdp               6cy @ 1p    f32 even lanes → f64
 *      vec_xl         lxv                    5cy @ 1p    aligned vector load
 *      vec_xl_len     lxvll                  6cy @ 1p    partial vector load
 *
 *  GEMM kernels use tiled dot products with 4-way parallel accumulation to hide FMA latency.
 *  Type-specific tile sizes: f32 uses depth_simd_dimensions=4, f64 uses depth_simd_dimensions=2,
 *  bf16/f16 use depth_simd_dimensions=8, u1 uses depth_simd_dimensions=128.
 */
#ifndef NK_DOTS_POWERVSX_H
#define NK_DOTS_POWERVSX_H

#if NK_TARGET_POWER64_
#if NK_TARGET_POWERVSX

#include "numkong/cast/powervsx.h" // `nk_load_b128_powervsx_`, `nk_partial_load_b32x4_powervsx_`
#include "numkong/dot/powervsx.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

/* F32 GEMM: depth_simd_dimensions=4 (4 f32s = 16 bytes = VSX register width, f64 accumulation) */
nk_define_cross_pack_size_(dots, f32, powervsx, f32, f32, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/2,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f32, powervsx, f32, f32, nk_b128_vec_t, nk_load_b128_powervsx_,
                      nk_partial_load_b32x4_powervsx_, nk_store_b128_powervsx_, nk_partial_store_b32x4_powervsx_,
                      /*simd_width=*/4, /*norm_value_type=*/f64, nk_dots_reduce_sumsq_f32_,
                      /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f32, powervsx, f32, f64, nk_b64_vec_t, nk_dot_f32x2_state_powervsx_t, nk_b256_vec_t,
                           nk_dot_f32x2_init_powervsx, nk_load_b64_powervsx_, nk_partial_load_b32x2_powervsx_,
                           nk_dot_f32x2_update_powervsx, nk_dot_f32x2_finalize_powervsx, nk_store_b256_powervsx_,
                           nk_partial_store_b64x4_powervsx_,
                           /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f32, powervsx, f32, f32, f64, nk_b64_vec_t, nk_dot_f32x2_state_powervsx_t, nk_b256_vec_t,
                        nk_dot_f32x2_init_powervsx, nk_load_b64_powervsx_, nk_partial_load_b32x2_powervsx_,
                        nk_load_b64_powervsx_, nk_partial_load_b32x2_powervsx_, nk_dot_f32x2_update_powervsx,
                        nk_dot_f32x2_finalize_powervsx, nk_store_b256_powervsx_, nk_partial_store_b64x4_powervsx_,
                        /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)

/* U1 GEMM: depth_simd_dimensions=128 (128 bits = 16 bytes = VSX register width) */
nk_define_cross_pack_size_(dots, u1, powervsx, u1x8, u1x8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/128,
                           /*dimensions_per_value=*/8)
nk_define_cross_pack_(dots, u1, powervsx, u1x8, u1x8, nk_b128_vec_t, nk_load_b128_powervsx_,
                      nk_partial_load_b8x16_powervsx_, nk_store_b128_powervsx_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/u32, nk_dots_reduce_sum_u1_,
                      /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)
nk_define_cross_symmetric_(dots, u1, powervsx, u1x8, u32, nk_b128_vec_t, nk_dot_u1x128_state_powervsx_t, nk_b128_vec_t,
                           nk_dot_u1x128_init_powervsx, nk_load_b128_powervsx_, nk_partial_load_b1x128_powervsx_,
                           nk_dot_u1x128_update_powervsx, nk_dot_u1x128_finalize_powervsx, nk_store_b128_powervsx_,
                           nk_partial_store_b32x4_powervsx_,
                           /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)
nk_define_cross_packed_(dots, u1, powervsx, u1x8, u1x8, u32, nk_b128_vec_t, nk_dot_u1x128_state_powervsx_t,
                        nk_b128_vec_t, nk_dot_u1x128_init_powervsx, nk_load_b128_powervsx_,
                        nk_partial_load_b1x128_powervsx_, nk_load_b128_powervsx_, nk_partial_load_b1x128_powervsx_,
                        nk_dot_u1x128_update_powervsx, nk_dot_u1x128_finalize_powervsx, nk_store_b128_powervsx_,
                        nk_partial_store_b32x4_powervsx_,
                        /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)

/* BF16 GEMM: depth_simd_dimensions=8 (8 bf16s = 16 bytes = VSX register width) */
nk_define_cross_pack_size_(dots, bf16, powervsx, bf16, bf16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, bf16, powervsx, bf16, bf16, nk_b128_vec_t, nk_load_b128_powervsx_,
                      nk_partial_load_b16x8_powervsx_, nk_store_b128_powervsx_, nk_partial_store_b16x8_serial_,
                      /*simd_width=*/8, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_bf16_,
                      /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, bf16, powervsx, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_powervsx_t,
                           nk_b128_vec_t, nk_dot_bf16x8_init_powervsx, nk_load_b128_powervsx_,
                           nk_partial_load_b16x8_powervsx_, nk_dot_bf16x8_update_powervsx,
                           nk_dot_bf16x8_finalize_powervsx, nk_store_b128_powervsx_, nk_partial_store_b32x4_powervsx_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, bf16, powervsx, bf16, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_powervsx_t,
                        nk_b128_vec_t, nk_dot_bf16x8_init_powervsx, nk_load_b128_powervsx_,
                        nk_partial_load_b16x8_powervsx_, nk_load_b128_powervsx_, nk_partial_load_b16x8_powervsx_,
                        nk_dot_bf16x8_update_powervsx, nk_dot_bf16x8_finalize_powervsx, nk_store_b128_powervsx_,
                        nk_partial_store_b32x4_powervsx_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* F16 GEMM: depth_simd_dimensions=8 (8 f16s = 16 bytes = VSX register width) */
nk_define_cross_pack_size_(dots, f16, powervsx, f16, f16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f16, powervsx, f16, f16, nk_b128_vec_t, nk_load_b128_powervsx_,
                      nk_partial_load_b16x8_powervsx_, nk_store_b128_powervsx_, nk_partial_store_b16x8_serial_,
                      /*simd_width=*/8, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_f16_,
                      /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f16, powervsx, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_powervsx_t, nk_b128_vec_t,
                           nk_dot_f16x8_init_powervsx, nk_load_b128_powervsx_, nk_partial_load_b16x8_powervsx_,
                           nk_dot_f16x8_update_powervsx, nk_dot_f16x8_finalize_powervsx, nk_store_b128_powervsx_,
                           nk_partial_store_b32x4_powervsx_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f16, powervsx, f16, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_powervsx_t, nk_b128_vec_t,
                        nk_dot_f16x8_init_powervsx, nk_load_b128_powervsx_, nk_partial_load_b16x8_powervsx_,
                        nk_load_b128_powervsx_, nk_partial_load_b16x8_powervsx_, nk_dot_f16x8_update_powervsx,
                        nk_dot_f16x8_finalize_powervsx, nk_store_b128_powervsx_, nk_partial_store_b32x4_powervsx_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* I8 GEMM: depth_simd_dimensions=16, compensated (algebraic bias, column sums precomputed) */
nk_define_cross_compensated_pack_size_(dots, i8, powervsx, i8, i8,
                                       /*sum_value_type=*/i32, /*norm_value_type=*/u32,
                                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_compensated_pack_(dots, i8, powervsx, i8, i8, nk_b128_vec_t, nk_load_b128_powervsx_,
                                  nk_partial_load_b8x16_powervsx_, nk_store_b128_powervsx_,
                                  nk_partial_store_b8x16_serial_, /*simd_width=*/16, /*sum_value_type=*/i32,
                                  /*norm_value_type=*/u32, nk_dots_reduce_moments_i8_, /*depth_simd_dimensions=*/16,
                                  /*dimensions_per_value=*/1)
nk_define_cross_compensated_symmetric_(dots, i8, powervsx, i8, i32,
                                       /*sum_value_type=*/i32, /*norm_value_type=*/u32, nk_b128_vec_t,
                                       nk_dot_i8x16_state_powervsx_t, nk_b128_vec_t, nk_dot_i8x16_init_powervsx,
                                       nk_load_b128_powervsx_, nk_partial_load_b8x16_powervsx_,
                                       nk_dot_i8x16_update_powervsx, nk_dot_i8x16_finalize_powervsx,
                                       nk_store_b128_powervsx_, nk_partial_store_b32x4_powervsx_,
                                       nk_load_b128_powervsx_, nk_partial_load_b32x4_powervsx_,
                                       nk_sum_i8x16_state_powervsx_t, nk_sum_i8x16_init_powervsx,
                                       nk_sum_i8x16_update_powervsx, nk_sum_i8x16_finalize_powervsx,
                                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_compensated_packed_(dots, i8, powervsx, i8, i8, i32,
                                    /*sum_value_type=*/i32, /*norm_value_type=*/u32, nk_b128_vec_t,
                                    nk_dot_i8x16_state_powervsx_t, nk_b128_vec_t, nk_dot_i8x16_init_powervsx,
                                    nk_load_b128_powervsx_, nk_partial_load_b8x16_powervsx_, nk_load_b128_powervsx_,
                                    nk_partial_load_b8x16_powervsx_, nk_dot_i8x16_update_powervsx,
                                    nk_dot_i8x16_finalize_powervsx, nk_store_b128_powervsx_,
                                    nk_partial_store_b32x4_powervsx_, nk_load_b128_powervsx_,
                                    nk_partial_load_b32x4_powervsx_, nk_dots_reduce_sum_i8_stub_,
                                    /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=16 (16 u8s = 16 bytes = VSX register width) */
nk_define_cross_pack_size_(dots, u8, powervsx, u8, u8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, u8, powervsx, u8, u8, nk_b128_vec_t, nk_load_b128_powervsx_,
                      nk_partial_load_b8x16_powervsx_, nk_store_b128_powervsx_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/u32, nk_dots_reduce_sumsq_u8_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, u8, powervsx, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_powervsx_t, nk_b128_vec_t,
                           nk_dot_u8x16_init_powervsx, nk_load_b128_powervsx_, nk_partial_load_b8x16_powervsx_,
                           nk_dot_u8x16_update_powervsx, nk_dot_u8x16_finalize_powervsx, nk_store_b128_powervsx_,
                           nk_partial_store_b32x4_powervsx_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, u8, powervsx, u8, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_powervsx_t, nk_b128_vec_t,
                        nk_dot_u8x16_init_powervsx, nk_load_b128_powervsx_, nk_partial_load_b8x16_powervsx_,
                        nk_load_b128_powervsx_, nk_partial_load_b8x16_powervsx_, nk_dot_u8x16_update_powervsx,
                        nk_dot_u8x16_finalize_powervsx, nk_store_b128_powervsx_, nk_partial_store_b32x4_powervsx_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* F64 GEMM: depth_simd_dimensions=2 (2 f64s = 16 bytes = VSX register width) */
nk_define_cross_pack_size_(dots, f64, powervsx, f64, f64, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/2,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f64, powervsx, f64, f64, nk_b128_vec_t, nk_load_b128_powervsx_,
                      nk_partial_load_b64x2_powervsx_, nk_store_b128_powervsx_, nk_partial_store_b64x2_serial_,
                      /*simd_width=*/2, /*norm_value_type=*/f64, nk_dots_reduce_sumsq_f64_,
                      /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f64, powervsx, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_powervsx_t, nk_b256_vec_t,
                           nk_dot_f64x2_init_powervsx, nk_load_b128_powervsx_, nk_partial_load_b64x2_powervsx_,
                           nk_dot_f64x2_update_powervsx, nk_dot_f64x2_finalize_powervsx, nk_store_b256_powervsx_,
                           nk_partial_store_b64x4_powervsx_,
                           /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f64, powervsx, f64, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_powervsx_t, nk_b256_vec_t,
                        nk_dot_f64x2_init_powervsx, nk_load_b128_powervsx_, nk_partial_load_b64x2_powervsx_,
                        nk_load_b128_powervsx_, nk_partial_load_b64x2_powervsx_, nk_dot_f64x2_update_powervsx,
                        nk_dot_f64x2_finalize_powervsx, nk_store_b256_powervsx_, nk_partial_store_b64x4_powervsx_,
                        /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_POWERVSX
#endif // NK_TARGET_POWER64_
#endif // NK_DOTS_POWERVSX_H
