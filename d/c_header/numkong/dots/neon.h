/**
 *  @brief SIMD-accelerated Batched Dot Products for NEON.
 *  @file include/numkong/dots/neon.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dots.h
 */
#ifndef NK_DOTS_NEON_H
#define NK_DOTS_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/dot/neon.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

/* F32 GEMM: depth_simd_dimensions=2 (2 f32s = 8 bytes = 64-bit input for f64 upcast accumulation) */
nk_define_cross_pack_size_(dots, f32, neon, f32, f32, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/2,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f32, neon, f32, f32, nk_assign_from_to_, /*norm_value_type=*/f64, nk_dots_reduce_sumsq_f32_,
                      /*depth_simd_dimensions=*/2,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f32, neon, f32, f64, nk_b64_vec_t, nk_dot_f32x2_state_neon_t, nk_b256_vec_t,
                           nk_dot_f32x2_init_neon, nk_load_b64_neon_, nk_partial_load_b32x2_serial_,
                           nk_dot_f32x2_update_neon, nk_dot_f32x2_finalize_neon, nk_store_b256_neon_,
                           nk_partial_store_b64x4_serial_,
                           /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f32, neon, f32, f32, f64, nk_b64_vec_t, nk_dot_f32x2_state_neon_t, nk_b256_vec_t,
                        nk_dot_f32x2_init_neon, nk_load_b64_neon_, nk_partial_load_b32x2_serial_, nk_load_b64_neon_,
                        nk_partial_load_b32x2_serial_, nk_dot_f32x2_update_neon, nk_dot_f32x2_finalize_neon,
                        nk_store_b256_neon_, nk_partial_store_b64x4_serial_,
                        /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)

/* U1 GEMM: depth_simd_dimensions=128 (128 bits = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, u1, neon, u1x8, u1x8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/128,
                           /*dimensions_per_value=*/8)
nk_define_cross_pack_(dots, u1, neon, u1x8, u1x8, nk_assign_from_to_, /*norm_value_type=*/u32, nk_dots_reduce_sum_u1_,
                      /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)
nk_define_cross_symmetric_(dots, u1, neon, u1x8, u32, nk_b128_vec_t, nk_dot_u1x128_state_neon_t, nk_b128_vec_t,
                           nk_dot_u1x128_init_neon, nk_load_b128_neon_, nk_partial_load_b1x128_serial_,
                           nk_dot_u1x128_update_neon, nk_dot_u1x128_finalize_neon, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)
nk_define_cross_packed_(dots, u1, neon, u1x8, u1x8, u32, nk_b128_vec_t, nk_dot_u1x128_state_neon_t, nk_b128_vec_t,
                        nk_dot_u1x128_init_neon, nk_load_b128_neon_, nk_partial_load_b1x128_serial_, nk_load_b128_neon_,
                        nk_partial_load_b1x128_serial_, nk_dot_u1x128_update_neon, nk_dot_u1x128_finalize_neon,
                        nk_store_b128_neon_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)

/* BF16 GEMM: depth_simd_dimensions=8 (8 bf16s = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, bf16, neon, bf16, bf16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, bf16, neon, bf16, bf16, nk_assign_from_to_, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_bf16_, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, bf16, neon, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_neon_t, nk_b128_vec_t,
                           nk_dot_bf16x8_init_neon, nk_load_b128_neon_, nk_partial_load_b16x8_serial_,
                           nk_dot_bf16x8_update_neon, nk_dot_bf16x8_finalize_neon, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, bf16, neon, bf16, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_neon_t, nk_b128_vec_t,
                        nk_dot_bf16x8_init_neon, nk_load_b128_neon_, nk_partial_load_b16x8_serial_, nk_load_b128_neon_,
                        nk_partial_load_b16x8_serial_, nk_dot_bf16x8_update_neon, nk_dot_bf16x8_finalize_neon,
                        nk_store_b128_neon_, nk_partial_store_b32x4_serial_, /*depth_simd_dimensions=*/8,
                        /*dimensions_per_value=*/1)

/* F16 GEMM: depth_simd_dimensions=8 (8 f16s = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, f16, neon, f16, f16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f16, neon, f16, f16, nk_assign_from_to_, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_f16_,
                      /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f16, neon, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_neon_t, nk_b128_vec_t,
                           nk_dot_f16x8_init_neon, nk_load_b128_neon_, nk_partial_load_b16x8_serial_,
                           nk_dot_f16x8_update_neon, nk_dot_f16x8_finalize_neon, nk_store_b128_neon_,
                           nk_partial_store_b32x4_serial_, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f16, neon, f16, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_neon_t, nk_b128_vec_t,
                        nk_dot_f16x8_init_neon, nk_load_b128_neon_, nk_partial_load_b16x8_serial_, nk_load_b128_neon_,
                        nk_partial_load_b16x8_serial_, nk_dot_f16x8_update_neon, nk_dot_f16x8_finalize_neon,
                        nk_store_b128_neon_, nk_partial_store_b32x4_serial_, /*depth_simd_dimensions=*/8,
                        /*dimensions_per_value=*/1)

/* F64 GEMM: depth_simd_dimensions=2 (2 f64s = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, f64, neon, f64, f64, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/2,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f64, neon, f64, f64, nk_assign_from_to_, /*norm_value_type=*/f64, nk_dots_reduce_sumsq_f64_,
                      /*depth_simd_dimensions=*/2,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f64, neon, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_neon_t, nk_b256_vec_t,
                           nk_dot_f64x2_init_neon, nk_load_b128_neon_, nk_partial_load_b64x2_serial_,
                           nk_dot_f64x2_update_neon, nk_dot_f64x2_finalize_neon, nk_store_b256_neon_,
                           nk_partial_store_b64x4_serial_,
                           /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f64, neon, f64, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_neon_t, nk_b256_vec_t,
                        nk_dot_f64x2_init_neon, nk_load_b128_neon_, nk_partial_load_b64x2_serial_, nk_load_b128_neon_,
                        nk_partial_load_b64x2_serial_, nk_dot_f64x2_update_neon, nk_dot_f64x2_finalize_neon,
                        nk_store_b256_neon_, nk_partial_store_b64x4_serial_,
                        /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_
#endif // NK_DOTS_NEON_H
