/**
 *  @file include/numkong/dots/sierra.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *  @brief SIMD-accelerated Batched Dot Products for Sierra Forest.
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses AVX-VNNI (256-bit) for integer GEMM:
 *  - _mm256_dpbssds_epi32: i8 × i8 → i32 with saturation
 *  - _mm256_dpbuud_epi32: u8 × u8 → u32 without saturation
 */
#ifndef NUMKONG_DOTS_SIERRA_H
#define NUMKONG_DOTS_SIERRA_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_TARGET_SIERRA

#include "numkong/dot/sierra.h"     // Sierra-specific dot product helpers
#include "numkong/dot/haswell.h"    // Haswell partial load functions
#include "numkong/dots/serial.h"    // GEMM macro definitions
#include "numkong/cast/haswell.h"   // `nk_partial_load_b8x16_haswell_`
#include "numkong/reduce/haswell.h" // `nk_reduce_add_i32x8_haswell_`
#include "numkong/reduce/sierra.h"  // `nk_reduce_moments_i8_sierra_contiguous_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni,avxvnniint8"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni", "avxvnniint8")
#endif

#pragma region Norms

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_i8_sierra_(nk_i8_t const *data, nk_size_t count, nk_size_t stride) {
    nk_i64_t sum;
    nk_u64_t sumsq;
    nk_unused_(stride);
    nk_reduce_moments_i8_sierra_contiguous_(data, count, &sum, &sumsq);
    return (nk_u32_t)sumsq;
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_u8_sierra_(nk_u8_t const *data, nk_size_t count, nk_size_t stride) {
    nk_u64_t sum, sumsq;
    nk_unused_(stride);
    nk_reduce_moments_u8_sierra_contiguous_(data, count, &sum, &sumsq);
    return (nk_u32_t)sumsq;
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e2m3_sierra_(nk_e2m3_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f32_t sum, sumsq;
    nk_unused_(stride);
    nk_reduce_moments_e2m3_sierra_contiguous_(data, count, &sum, &sumsq);
    return sumsq;
}

#pragma endregion Norms

/* I8 GEMM: depth_simd_dimensions=32 (32 i8s = 32 bytes = AVX2 register width) */
nk_define_cross_pack_size_(dots, i8, sierra, i8, i8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, i8, sierra)
nk_define_cross_pack_(dots, i8, sierra, i8, i8, nk_b128_vec_t, nk_load_b128_haswell_, nk_partial_load_b8x16_haswell_,
                      nk_store_b128_haswell_, nk_partial_store_b8x16_haswell_, /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sumsq_i8_sierra_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, i8, sierra, i8, i32, nk_b256_vec_t, nk_dot_i8x32_state_sierra_t, nk_b128_vec_t,
                           nk_dot_i8x32_init_sierra, nk_cross_unscaled_, nk_load_b256_haswell_,
                           nk_partial_load_b8x32_haswell_, nk_dot_i8x32_update_sierra, nk_dot_i8x32_finalize_sierra,
                           nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, i8, sierra, i8, i8, i32, nk_b256_vec_t, nk_dot_i8x32_state_sierra_t, nk_b128_vec_t,
                        nk_dot_i8x32_init_sierra, nk_cross_unscaled_, nk_load_b256_haswell_,
                        nk_partial_load_b8x32_haswell_, nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_,
                        nk_dot_i8x32_update_sierra, nk_dot_i8x32_finalize_sierra, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_haswell_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=32 (32 u8s = 32 bytes = AVX2 register width) */
nk_define_cross_pack_size_(dots, u8, sierra, u8, u8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, u8, sierra)
nk_define_cross_pack_(dots, u8, sierra, u8, u8, nk_b128_vec_t, nk_load_b128_haswell_, nk_partial_load_b8x16_haswell_,
                      nk_store_b128_haswell_, nk_partial_store_b8x16_haswell_, /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sumsq_u8_sierra_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, u8, sierra, u8, u32, nk_b256_vec_t, nk_dot_u8x32_state_sierra_t, nk_b128_vec_t,
                           nk_dot_u8x32_init_sierra, nk_cross_unscaled_, nk_load_b256_haswell_,
                           nk_partial_load_b8x32_haswell_, nk_dot_u8x32_update_sierra, nk_dot_u8x32_finalize_sierra,
                           nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, u8, sierra, u8, u8, u32, nk_b256_vec_t, nk_dot_u8x32_state_sierra_t, nk_b128_vec_t,
                        nk_dot_u8x32_init_sierra, nk_cross_unscaled_, nk_load_b256_haswell_,
                        nk_partial_load_b8x32_haswell_, nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_,
                        nk_dot_u8x32_update_sierra, nk_dot_u8x32_finalize_sierra, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_haswell_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/*  E2M3 GEMM via the DPBUSD integer path: depth_simd_dimensions = 32, as 32 e2m3s span the 32 bytes
 *  of an AVX2 register. */
nk_define_cross_pack_size_(dots, e2m3, sierra, e2m3, e2m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, e2m3, sierra)
nk_define_cross_pack_(dots, e2m3, sierra, e2m3, e2m3, nk_b128_vec_t, nk_load_b128_haswell_,
                      nk_partial_load_b8x16_haswell_, nk_store_b128_haswell_, nk_partial_store_b8x16_haswell_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m3_sierra_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e2m3, sierra, e2m3, f32, nk_b256_vec_t, nk_dot_e2m3x32_state_sierra_t, nk_b128_vec_t,
                           nk_dot_e2m3x32_init_sierra, nk_cross_unscaled_, nk_load_b256_haswell_,
                           nk_partial_load_b8x32_haswell_, nk_dot_e2m3x32_update_sierra, nk_dot_e2m3x32_finalize_sierra,
                           nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e2m3, sierra, e2m3, e2m3, f32, nk_b256_vec_t, nk_dot_e2m3x32_state_sierra_t,
                        nk_b128_vec_t, nk_dot_e2m3x32_init_sierra, nk_cross_unscaled_, nk_load_b256_haswell_,
                        nk_partial_load_b8x32_haswell_, nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_,
                        nk_dot_e2m3x32_update_sierra, nk_dot_e2m3x32_finalize_sierra, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_haswell_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/*  E2M1 GEMM via the DPBSSD integer path: depth_simd_dimensions = 64, as 64 nibbles span the 32
 *  bytes of an AVX2 register. */
nk_define_cross_pack_size_(dots, e2m1, sierra, e2m1x2, e2m1x2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/2)
nk_define_cross_packed_shape_(dots, e2m1, sierra)
nk_define_cross_pack_(dots, e2m1, sierra, e2m1x2, e2m1x2, nk_b256_vec_t, nk_load_b256_haswell_,
                      nk_partial_load_b8x32_haswell_, nk_store_b256_haswell_, nk_partial_store_b8x32_haswell_,
                      /*simd_width=*/32, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m1_,
                      /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2)
nk_define_cross_symmetric_(dots, e2m1, sierra, e2m1x2, f32, nk_b256_vec_t, nk_dot_e2m1x64_state_sierra_t, nk_b128_vec_t,
                           nk_dot_e2m1x64_init_sierra, nk_cross_unscaled_, nk_load_b256_haswell_,
                           nk_partial_load_b4x64_serial_, nk_dot_e2m1x64_update_sierra, nk_dot_e2m1x64_finalize_sierra,
                           nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, e2m1, sierra, e2m1x2, e2m1x2, f32, nk_b256_vec_t, nk_dot_e2m1x64_state_sierra_t,
                        nk_b128_vec_t, nk_dot_e2m1x64_init_sierra, nk_cross_unscaled_, nk_load_b256_haswell_,
                        nk_partial_load_b4x64_serial_, nk_load_b256_haswell_, nk_partial_load_b4x64_serial_,
                        nk_dot_e2m1x64_update_sierra, nk_dot_e2m1x64_finalize_sierra, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_haswell_,
                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_SIERRA
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_DOTS_SIERRA_H
