/**
 *  @brief SIMD-accelerated Batched Dot Products for Sierra Forest.
 *  @file include/numkong/dots/sierra.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses AVX-VNNI (256-bit) for integer GEMM:
 *  - _mm256_dpbssds_epi32: i8 × i8 → i32 with saturation
 *  - _mm256_dpbuud_epi32: u8 × u8 → u32 without saturation
 */
#ifndef NK_DOTS_SIERRA_H
#define NK_DOTS_SIERRA_H

#if NK_TARGET_X86_
#if NK_TARGET_SIERRA

#include "numkong/dot/sierra.h"  // Sierra-specific dot product helpers
#include "numkong/dot/haswell.h" // Haswell partial load functions
#include "numkong/dots/serial.h" // GEMM macro definitions

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni,avxvnniint8"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni", "avxvnniint8")
#endif

/* I8 GEMM: depth_simd_dimensions=32 (32 i8s = 32 bytes = AVX2 register width) */
nk_define_cross_pack_size_(dots, i8, sierra, i8, i8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, i8, sierra, i8, i8, nk_assign_from_to_, /*norm_value_type=*/u32, nk_dots_reduce_sumsq_i8_,
                      /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, i8, sierra, i8, i32, nk_b256_vec_t, nk_dot_i8x32_state_sierra_t, nk_b128_vec_t,
                           nk_dot_i8x32_init_sierra, nk_load_b256_haswell_, nk_partial_load_b8x32_serial_,
                           nk_dot_i8x32_update_sierra, nk_dot_i8x32_finalize_sierra, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_haswell_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, i8, sierra, i8, i8, i32, nk_b256_vec_t, nk_dot_i8x32_state_sierra_t, nk_b128_vec_t,
                        nk_dot_i8x32_init_sierra, nk_load_b256_haswell_, nk_partial_load_b8x32_serial_,
                        nk_load_b256_haswell_, nk_partial_load_b8x32_serial_, nk_dot_i8x32_update_sierra,
                        nk_dot_i8x32_finalize_sierra, nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=32 (32 u8s = 32 bytes = AVX2 register width) */
nk_define_cross_pack_size_(dots, u8, sierra, u8, u8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, u8, sierra, u8, u8, nk_assign_from_to_, /*norm_value_type=*/u32, nk_dots_reduce_sumsq_u8_,
                      /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, u8, sierra, u8, u32, nk_b256_vec_t, nk_dot_u8x32_state_sierra_t, nk_b128_vec_t,
                           nk_dot_u8x32_init_sierra, nk_load_b256_haswell_, nk_partial_load_b8x32_serial_,
                           nk_dot_u8x32_update_sierra, nk_dot_u8x32_finalize_sierra, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_haswell_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, u8, sierra, u8, u8, u32, nk_b256_vec_t, nk_dot_u8x32_state_sierra_t, nk_b128_vec_t,
                        nk_dot_u8x32_init_sierra, nk_load_b256_haswell_, nk_partial_load_b8x32_serial_,
                        nk_load_b256_haswell_, nk_partial_load_b8x32_serial_, nk_dot_u8x32_update_sierra,
                        nk_dot_u8x32_finalize_sierra, nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/* E2M3 GEMM via DPBUSD integer path: depth_simd_dimensions=32 (32 e2m3s = 32 bytes = AVX2 register width) */
nk_define_cross_pack_size_(dots, e2m3, sierra, e2m3, e2m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e2m3, sierra, e2m3, e2m3, nk_assign_from_to_, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_e2m3_, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e2m3, sierra, e2m3, f32, nk_b256_vec_t, nk_dot_e2m3x32_state_sierra_t, nk_b128_vec_t,
                           nk_dot_e2m3x32_init_sierra, nk_load_b256_haswell_, nk_partial_load_b8x32_serial_,
                           nk_dot_e2m3x32_update_sierra, nk_dot_e2m3x32_finalize_sierra, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_haswell_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e2m3, sierra, e2m3, e2m3, f32, nk_b256_vec_t, nk_dot_e2m3x32_state_sierra_t,
                        nk_b128_vec_t, nk_dot_e2m3x32_init_sierra, nk_load_b256_haswell_, nk_partial_load_b8x32_serial_,
                        nk_load_b256_haswell_, nk_partial_load_b8x32_serial_, nk_dot_e2m3x32_update_sierra,
                        nk_dot_e2m3x32_finalize_sierra, nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SIERRA
#endif // NK_TARGET_X86_
#endif // NK_DOTS_SIERRA_H
