/**
 *  @brief SIMD-accelerated Batched Dot Products for WASM Relaxed SIMD.
 *  @file include/numkong/dots/v128relaxed.h
 *  @author Ash Vardanian
 *  @date March 5, 2026
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses relaxed SIMD dot products for integer GEMM. I8 uses 2×relaxed_dot with
 *  bit-split (b_lo + (-128)·b_hi). U8 uses 2×relaxed_dot with signed reinterpretation
 *  and b_sums compensation. E2M3 uses standard single-register state (no correction).
 */
#ifndef NK_DOTS_V128RELAXED_H
#define NK_DOTS_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/dot/v128relaxed.h"
#include "numkong/dots/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/* I8 GEMM: depth_simd_dimensions=16 — standard (correction depends on both A and B) */
nk_define_cross_pack_size_(dots, i8, v128relaxed, i8, i8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, i8, v128relaxed, i8, i8, nk_assign_from_to_, /*norm_value_type=*/u32,
                      nk_dots_reduce_sumsq_i8_, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, i8, v128relaxed, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_v128relaxed_t,
                           nk_b128_vec_t, nk_dot_i8x16_init_v128relaxed, nk_load_b128_v128relaxed_,
                           nk_partial_load_b8x16_serial_, nk_dot_i8x16_update_v128relaxed,
                           nk_dot_i8x16_finalize_v128relaxed, nk_store_b128_v128relaxed_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, i8, v128relaxed, i8, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_v128relaxed_t,
                        nk_b128_vec_t, nk_dot_i8x16_init_v128relaxed, nk_load_b128_v128relaxed_,
                        nk_partial_load_b8x16_serial_, nk_load_b128_v128relaxed_, nk_partial_load_b8x16_serial_,
                        nk_dot_i8x16_update_v128relaxed, nk_dot_i8x16_finalize_v128relaxed, nk_store_b128_v128relaxed_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=16 — compensated (2×relaxed_dot with bit-split + b_sums correction) */
nk_define_cross_compensated_pack_size_(dots, u8, v128relaxed, u8, u8,
                                       /*sum_value_type=*/u32, /*norm_value_type=*/u32,
                                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_compensated_pack_(dots, u8, v128relaxed, u8, u8, nk_assign_from_to_,
                                  /*sum_value_type=*/u32, /*norm_value_type=*/u32, nk_dots_reduce_moments_u8_,
                                  /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_compensated_symmetric_(dots, u8, v128relaxed, u8, u32,
                                       /*sum_value_type=*/u32, /*norm_value_type=*/u32, nk_b128_vec_t,
                                       nk_dot_u8x16_state_v128relaxed_t, nk_b128_vec_t, nk_dot_u8x16_init_v128relaxed,
                                       nk_load_b128_v128relaxed_, nk_partial_load_b8x16_serial_,
                                       nk_dot_u8x16_update_v128relaxed, nk_dot_u8x16_finalize_v128relaxed,
                                       nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_,
                                       nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                       nk_sum_u8x16_state_v128relaxed_t, nk_sum_u8x16_init_v128relaxed,
                                       nk_sum_u8x16_update_v128relaxed, nk_sum_u8x16_finalize_v128relaxed,
                                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_compensated_packed_(dots, u8, v128relaxed, u8, u8, u32,
                                    /*sum_value_type=*/u32, /*norm_value_type=*/u32, nk_b128_vec_t,
                                    nk_dot_u8x16_state_v128relaxed_t, nk_b128_vec_t, nk_dot_u8x16_init_v128relaxed,
                                    nk_load_b128_v128relaxed_, nk_partial_load_b8x16_serial_, nk_load_b128_v128relaxed_,
                                    nk_partial_load_b8x16_serial_, nk_dot_u8x16_update_v128relaxed,
                                    nk_dot_u8x16_finalize_v128relaxed, nk_store_b128_v128relaxed_,
                                    nk_partial_store_b32x4_serial_, nk_load_b128_v128relaxed_,
                                    nk_partial_load_b32x4_serial_, nk_dots_reduce_sum_u8_stub_,
                                    /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E2M3 GEMM: depth_simd_dimensions=16 — standard (magnitudes fit u7, no correction) */
nk_define_cross_pack_size_(dots, e2m3, v128relaxed, e2m3, e2m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e2m3, v128relaxed, e2m3, e2m3, nk_assign_from_to_, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_e2m3_, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e2m3, v128relaxed, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x16_state_v128relaxed_t,
                           nk_b128_vec_t, nk_dot_e2m3x16_init_v128relaxed, nk_load_b128_v128relaxed_,
                           nk_partial_load_b8x16_serial_, nk_dot_e2m3x16_update_v128relaxed,
                           nk_dot_e2m3x16_finalize_v128relaxed, nk_store_b128_v128relaxed_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e2m3, v128relaxed, e2m3, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x16_state_v128relaxed_t,
                        nk_b128_vec_t, nk_dot_e2m3x16_init_v128relaxed, nk_load_b128_v128relaxed_,
                        nk_partial_load_b8x16_serial_, nk_load_b128_v128relaxed_, nk_partial_load_b8x16_serial_,
                        nk_dot_e2m3x16_update_v128relaxed, nk_dot_e2m3x16_finalize_v128relaxed,
                        nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E3M2 GEMM: depth_simd_dimensions=16 — ×4 scaled relaxed_dot (magnitudes ≤ 112, fits u7) */
nk_define_cross_pack_size_(dots, e3m2, v128relaxed, e3m2, e3m2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e3m2, v128relaxed, e3m2, e3m2, nk_assign_from_to_, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_e3m2_, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e3m2, v128relaxed, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x16_state_v128relaxed_t,
                           nk_b128_vec_t, nk_dot_e3m2x16_init_v128relaxed, nk_load_b128_v128relaxed_,
                           nk_partial_load_b8x16_serial_, nk_dot_e3m2x16_update_v128relaxed,
                           nk_dot_e3m2x16_finalize_v128relaxed, nk_store_b128_v128relaxed_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e3m2, v128relaxed, e3m2, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x16_state_v128relaxed_t,
                        nk_b128_vec_t, nk_dot_e3m2x16_init_v128relaxed, nk_load_b128_v128relaxed_,
                        nk_partial_load_b8x16_serial_, nk_load_b128_v128relaxed_, nk_partial_load_b8x16_serial_,
                        nk_dot_e3m2x16_update_v128relaxed, nk_dot_e3m2x16_finalize_v128relaxed,
                        nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* BF16 GEMM: depth_simd_dimensions=4 — upcast to f32x4 via nk_bf16x4_to_f32x4_v128relaxed_ */
nk_define_cross_pack_size_(dots, bf16, v128relaxed, bf16, f32, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/4,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, bf16, v128relaxed, bf16, f32, nk_bf16_to_f32_serial, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_bf16_, /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, bf16, v128relaxed, bf16, f32, nk_b128_vec_t, nk_dot_through_f32x4_state_v128relaxed_t_,
                           nk_b128_vec_t, nk_dot_through_f32x4_init_v128relaxed_, nk_load_bf16x4_to_f32x4_v128relaxed_,
                           nk_partial_load_bf16x4_to_f32x4_v128relaxed_, nk_dot_through_f32x4_update_v128relaxed_,
                           nk_dot_through_f32x4_finalize_v128relaxed_, nk_store_b128_v128relaxed_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, bf16, v128relaxed, bf16, f32, f32, nk_b128_vec_t,
                        nk_dot_through_f32x4_state_v128relaxed_t_, nk_b128_vec_t,
                        nk_dot_through_f32x4_init_v128relaxed_, nk_load_bf16x4_to_f32x4_v128relaxed_,
                        nk_partial_load_bf16x4_to_f32x4_v128relaxed_, nk_load_b128_v128relaxed_,
                        nk_partial_load_b32x4_serial_, nk_dot_through_f32x4_update_v128relaxed_,
                        nk_dot_through_f32x4_finalize_v128relaxed_, nk_store_b128_v128relaxed_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)

/* F16 GEMM: depth_simd_dimensions=4 — upcast to f32x4 via nk_f16x4_to_f32x4_v128relaxed_ */
nk_define_cross_pack_size_(dots, f16, v128relaxed, f16, f32, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/4,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f16, v128relaxed, f16, f32, nk_f16_to_f32_serial, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_f16_, /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f16, v128relaxed, f16, f32, nk_b128_vec_t, nk_dot_through_f32x4_state_v128relaxed_t_,
                           nk_b128_vec_t, nk_dot_through_f32x4_init_v128relaxed_, nk_load_f16x4_to_f32x4_v128relaxed_,
                           nk_partial_load_f16x4_to_f32x4_v128relaxed_, nk_dot_through_f32x4_update_v128relaxed_,
                           nk_dot_through_f32x4_finalize_v128relaxed_, nk_store_b128_v128relaxed_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f16, v128relaxed, f16, f32, f32, nk_b128_vec_t, nk_dot_through_f32x4_state_v128relaxed_t_,
                        nk_b128_vec_t, nk_dot_through_f32x4_init_v128relaxed_, nk_load_f16x4_to_f32x4_v128relaxed_,
                        nk_partial_load_f16x4_to_f32x4_v128relaxed_, nk_load_b128_v128relaxed_,
                        nk_partial_load_b32x4_serial_, nk_dot_through_f32x4_update_v128relaxed_,
                        nk_dot_through_f32x4_finalize_v128relaxed_, nk_store_b128_v128relaxed_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)

nk_define_cross_pack_size_(dots, f32, v128relaxed, f32, f32, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/2,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f32, v128relaxed, f32, f32, nk_assign_from_to_, /*norm_value_type=*/f64,
                      nk_dots_reduce_sumsq_f32_, /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f32, v128relaxed, f32, f64, nk_b64_vec_t, nk_dot_f32x2_state_v128relaxed_t,
                           nk_b256_vec_t, nk_dot_f32x2_init_v128relaxed, nk_load_b64_serial_,
                           nk_partial_load_b32x2_serial_, nk_dot_f32x2_update_v128relaxed,
                           nk_dot_f32x2_finalize_v128relaxed, nk_store_b256_v128relaxed_,
                           nk_partial_store_b64x4_serial_,
                           /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f32, v128relaxed, f32, f32, f64, nk_b64_vec_t, nk_dot_f32x2_state_v128relaxed_t,
                        nk_b256_vec_t, nk_dot_f32x2_init_v128relaxed, nk_load_b64_serial_,
                        nk_partial_load_b32x2_serial_, nk_load_b64_serial_, nk_partial_load_b32x2_serial_,
                        nk_dot_f32x2_update_v128relaxed, nk_dot_f32x2_finalize_v128relaxed, nk_store_b256_v128relaxed_,
                        nk_partial_store_b64x4_serial_,
                        /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)

nk_define_cross_pack_size_(dots, f64, v128relaxed, f64, f64, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/2,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f64, v128relaxed, f64, f64, nk_assign_from_to_, /*norm_value_type=*/f64,
                      nk_dots_reduce_sumsq_f64_, /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f64, v128relaxed, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_v128relaxed_t,
                           nk_b256_vec_t, nk_dot_f64x2_init_v128relaxed, nk_load_b128_v128relaxed_,
                           nk_partial_load_b64x2_serial_, nk_dot_f64x2_update_v128relaxed,
                           nk_dot_f64x2_finalize_v128relaxed, nk_store_b256_v128relaxed_,
                           nk_partial_store_b64x4_serial_,
                           /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f64, v128relaxed, f64, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_v128relaxed_t,
                        nk_b256_vec_t, nk_dot_f64x2_init_v128relaxed, nk_load_b128_v128relaxed_,
                        nk_partial_load_b64x2_serial_, nk_load_b128_v128relaxed_, nk_partial_load_b64x2_serial_,
                        nk_dot_f64x2_update_v128relaxed, nk_dot_f64x2_finalize_v128relaxed, nk_store_b256_v128relaxed_,
                        nk_partial_store_b64x4_serial_,
                        /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)

/* E4M3 GEMM: depth_simd_dimensions=4 — upcast to f32x4 via nk_e4m3x4_to_f32x4_v128relaxed_ */
nk_define_cross_pack_size_(dots, e4m3, v128relaxed, e4m3, f32, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e4m3, v128relaxed, e4m3, f32, nk_e4m3_to_f32_serial, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_e4m3_, /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e4m3, v128relaxed, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x4_state_v128relaxed_t,
                           nk_b128_vec_t, nk_dot_through_f32x4_init_v128relaxed_, nk_load_e4m3x4_to_f32x4_v128relaxed_,
                           nk_partial_load_e4m3x4_to_f32x4_v128relaxed_, nk_dot_through_f32x4_update_v128relaxed_,
                           nk_dot_through_f32x4_finalize_v128relaxed_, nk_store_b128_v128relaxed_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e4m3, v128relaxed, e4m3, f32, f32, nk_b128_vec_t, nk_dot_e4m3x4_state_v128relaxed_t,
                        nk_b128_vec_t, nk_dot_through_f32x4_init_v128relaxed_, nk_load_e4m3x4_to_f32x4_v128relaxed_,
                        nk_partial_load_e4m3x4_to_f32x4_v128relaxed_, nk_load_b128_v128relaxed_,
                        nk_partial_load_b32x4_serial_, nk_dot_through_f32x4_update_v128relaxed_,
                        nk_dot_through_f32x4_finalize_v128relaxed_, nk_store_b128_v128relaxed_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)

/* E5M2 GEMM: depth_simd_dimensions=4 — upcast to f32x4 via nk_e5m2x4_to_f32x4_v128relaxed_ */
nk_define_cross_pack_size_(dots, e5m2, v128relaxed, e5m2, f32, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e5m2, v128relaxed, e5m2, f32, nk_e5m2_to_f32_serial, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_e5m2_, /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e5m2, v128relaxed, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x4_state_v128relaxed_t,
                           nk_b128_vec_t, nk_dot_through_f32x4_init_v128relaxed_, nk_load_e5m2x4_to_f32x4_v128relaxed_,
                           nk_partial_load_e5m2x4_to_f32x4_v128relaxed_, nk_dot_through_f32x4_update_v128relaxed_,
                           nk_dot_through_f32x4_finalize_v128relaxed_, nk_store_b128_v128relaxed_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e5m2, v128relaxed, e5m2, f32, f32, nk_b128_vec_t, nk_dot_e5m2x4_state_v128relaxed_t,
                        nk_b128_vec_t, nk_dot_through_f32x4_init_v128relaxed_, nk_load_e5m2x4_to_f32x4_v128relaxed_,
                        nk_partial_load_e5m2x4_to_f32x4_v128relaxed_, nk_load_b128_v128relaxed_,
                        nk_partial_load_b32x4_serial_, nk_dot_through_f32x4_update_v128relaxed_,
                        nk_dot_through_f32x4_finalize_v128relaxed_, nk_store_b128_v128relaxed_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)

/* U4 GEMM: depth_simd_dimensions=32 — nibble extraction + relaxed_dot */
nk_define_cross_pack_size_(dots, u4, v128relaxed, u4x2, u4x2, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_pack_(dots, u4, v128relaxed, u4x2, u4x2, nk_assign_from_to_, /*norm_value_type=*/u32,
                      nk_dots_reduce_sumsq_u4_, /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_symmetric_(dots, u4, v128relaxed, u4x2, u32, nk_b128_vec_t, nk_dot_u4x32_state_v128relaxed_t,
                           nk_b128_vec_t, nk_dot_u4x32_init_v128relaxed, nk_load_b128_v128relaxed_,
                           nk_partial_load_b4x32_serial_, nk_dot_u4x32_update_v128relaxed,
                           nk_dot_u4x32_finalize_v128relaxed, nk_store_b128_v128relaxed_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, u4, v128relaxed, u4x2, u4x2, u32, nk_b128_vec_t, nk_dot_u4x32_state_v128relaxed_t,
                        nk_b128_vec_t, nk_dot_u4x32_init_v128relaxed, nk_load_b128_v128relaxed_,
                        nk_partial_load_b4x32_serial_, nk_load_b128_v128relaxed_, nk_partial_load_b4x32_serial_,
                        nk_dot_u4x32_update_v128relaxed, nk_dot_u4x32_finalize_v128relaxed, nk_store_b128_v128relaxed_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)

/* I4 GEMM: depth_simd_dimensions=32 — compensated (XOR-bias algebraic transform) */
nk_define_cross_compensated_pack_size_(dots, i4, v128relaxed, i4x2, i4x2,
                                       /*sum_value_type=*/i32, /*norm_value_type=*/u32,
                                       /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_compensated_pack_(dots, i4, v128relaxed, i4x2, i4x2, nk_assign_from_to_,
                                  /*sum_value_type=*/i32, /*norm_value_type=*/u32, nk_dots_reduce_moments_i4_,
                                  /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_compensated_symmetric_(dots, i4, v128relaxed, i4x2, i32,
                                       /*sum_value_type=*/i32, /*norm_value_type=*/u32, nk_b128_vec_t,
                                       nk_dot_i4x32_state_v128relaxed_t, nk_b128_vec_t, nk_dot_i4x32_init_v128relaxed,
                                       nk_load_b128_v128relaxed_, nk_partial_load_b4x32_serial_,
                                       nk_dot_i4x32_update_v128relaxed, nk_dot_i4x32_finalize_v128relaxed,
                                       nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_,
                                       nk_load_b128_v128relaxed_, nk_partial_load_b32x4_serial_,
                                       nk_sum_i4x32_state_v128relaxed_t, nk_sum_i4x32_init_v128relaxed,
                                       nk_sum_i4x32_update_v128relaxed, nk_sum_i4x32_finalize_v128relaxed,
                                       /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_compensated_packed_(dots, i4, v128relaxed, i4x2, i4x2, i32,
                                    /*sum_value_type=*/i32, /*norm_value_type=*/u32, nk_b128_vec_t,
                                    nk_dot_i4x32_state_v128relaxed_t, nk_b128_vec_t, nk_dot_i4x32_init_v128relaxed,
                                    nk_load_b128_v128relaxed_, nk_partial_load_b4x32_serial_, nk_load_b128_v128relaxed_,
                                    nk_partial_load_b4x32_serial_, nk_dot_i4x32_update_v128relaxed,
                                    nk_dot_i4x32_finalize_v128relaxed, nk_store_b128_v128relaxed_,
                                    nk_partial_store_b32x4_serial_, nk_load_b128_v128relaxed_,
                                    nk_partial_load_b32x4_serial_, nk_dots_reduce_sum_i4_,
                                    /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)

/* U1 GEMM: depth_simd_dimensions=128 (128 bits = 16 bytes = v128 register width) */
nk_define_cross_pack_size_(dots, u1, v128relaxed, u1x8, u1x8, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)
nk_define_cross_pack_(dots, u1, v128relaxed, u1x8, u1x8, nk_assign_from_to_,
                      /*norm_value_type=*/u32, nk_dots_reduce_sum_u1_,
                      /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)
nk_define_cross_symmetric_(dots, u1, v128relaxed, u1x8, u32, nk_b128_vec_t, nk_dot_u1x128_state_v128relaxed_t,
                           nk_b128_vec_t, nk_dot_u1x128_init_v128relaxed, nk_load_b128_v128relaxed_,
                           nk_partial_load_b1x128_serial_, nk_dot_u1x128_update_v128relaxed,
                           nk_dot_u1x128_finalize_v128relaxed, nk_store_b128_v128relaxed_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)
nk_define_cross_packed_(dots, u1, v128relaxed, u1x8, u1x8, u32, nk_b128_vec_t, nk_dot_u1x128_state_v128relaxed_t,
                        nk_b128_vec_t, nk_dot_u1x128_init_v128relaxed, nk_load_b128_v128relaxed_,
                        nk_partial_load_b1x128_serial_, nk_load_b128_v128relaxed_, nk_partial_load_b1x128_serial_,
                        nk_dot_u1x128_update_v128relaxed, nk_dot_u1x128_finalize_v128relaxed,
                        nk_store_b128_v128relaxed_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_DOTS_V128RELAXED_H
