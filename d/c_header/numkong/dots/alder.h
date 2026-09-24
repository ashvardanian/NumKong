/**
 *  @brief SIMD-accelerated Batched Dot Products for Alder Lake.
 *  @file include/numkong/dots/alder.h
 *  @author Ash Vardanian
 *  @date March 4, 2026
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses AVX-VNNI (256-bit) for integer GEMM via DPBUSD with algebraic
 *  sign transformations for signed*signed and unsigned*unsigned cases.
 */
#ifndef NK_DOTS_ALDER_H
#define NK_DOTS_ALDER_H

#if NK_TARGET_X86_
#if NK_TARGET_ALDER

#include "numkong/dot/alder.h"   // Alder-specific dot product helpers
#include "numkong/dot/haswell.h" // Haswell partial load functions
#include "numkong/dots/serial.h" // GEMM macro definitions

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni")
#endif

/* I8 GEMM: depth_simd_dimensions=32 — compensated (B sums precomputed in pack) */
nk_define_cross_compensated_pack_size_(dots, i8, alder, i8, i8,
                                       /*sum_value_type=*/i32, /*norm_value_type=*/u32,
                                       /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_compensated_pack_(dots, i8, alder, i8, i8, nk_assign_from_to_,
                                  /*sum_value_type=*/i32, /*norm_value_type=*/u32, nk_dots_reduce_moments_i8_,
                                  /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_compensated_symmetric_(dots, i8, alder, i8, i32,
                                       /*sum_value_type=*/i32, /*norm_value_type=*/u32, nk_b256_vec_t,
                                       nk_dot_i8x32_state_alder_t, nk_b128_vec_t, nk_dot_i8x32_init_alder,
                                       nk_load_b256_haswell_, nk_partial_load_b8x32_serial_, nk_dot_i8x32_update_alder,
                                       nk_dot_i8x32_finalize_alder, nk_store_b128_haswell_,
                                       nk_partial_store_b32x4_haswell_, nk_load_b128_haswell_,
                                       nk_partial_load_b32x4_haswell_, nk_sum_i8x32_state_alder_t,
                                       nk_sum_i8x32_init_alder, nk_sum_i8x32_update_alder, nk_sum_i8x32_finalize_alder,
                                       /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_compensated_packed_(dots, i8, alder, i8, i8, i32,
                                    /*sum_value_type=*/i32, /*norm_value_type=*/u32, nk_b256_vec_t,
                                    nk_dot_i8x32_state_alder_t, nk_b128_vec_t, nk_dot_i8x32_init_alder,
                                    nk_load_b256_haswell_, nk_partial_load_b8x32_serial_, nk_load_b256_haswell_,
                                    nk_partial_load_b8x32_serial_, nk_dot_i8x32_update_alder,
                                    nk_dot_i8x32_finalize_alder, nk_store_b128_haswell_,
                                    nk_partial_store_b32x4_haswell_, nk_load_b128_haswell_,
                                    nk_partial_load_b32x4_haswell_, nk_dots_reduce_sum_i8_stub_,
                                    /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=32 — compensated (operand swap, B sums precomputed) */
nk_define_cross_compensated_pack_size_(dots, u8, alder, u8, u8,
                                       /*sum_value_type=*/u32, /*norm_value_type=*/u32,
                                       /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_compensated_pack_(dots, u8, alder, u8, u8, nk_assign_from_to_,
                                  /*sum_value_type=*/u32, /*norm_value_type=*/u32, nk_dots_reduce_moments_u8_,
                                  /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_compensated_symmetric_(dots, u8, alder, u8, u32,
                                       /*sum_value_type=*/u32, /*norm_value_type=*/u32, nk_b256_vec_t,
                                       nk_dot_u8x32_state_alder_t, nk_b128_vec_t, nk_dot_u8x32_init_alder,
                                       nk_load_b256_haswell_, nk_partial_load_b8x32_serial_, nk_dot_u8x32_update_alder,
                                       nk_dot_u8x32_finalize_alder, nk_store_b128_haswell_,
                                       nk_partial_store_b32x4_haswell_, nk_load_b128_haswell_,
                                       nk_partial_load_b32x4_haswell_, nk_sum_u8x32_state_alder_t,
                                       nk_sum_u8x32_init_alder, nk_sum_u8x32_update_alder, nk_sum_u8x32_finalize_alder,
                                       /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_compensated_packed_(dots, u8, alder, u8, u8, u32,
                                    /*sum_value_type=*/u32, /*norm_value_type=*/u32, nk_b256_vec_t,
                                    nk_dot_u8x32_state_alder_t, nk_b128_vec_t, nk_dot_u8x32_init_alder,
                                    nk_load_b256_haswell_, nk_partial_load_b8x32_serial_, nk_load_b256_haswell_,
                                    nk_partial_load_b8x32_serial_, nk_dot_u8x32_update_alder,
                                    nk_dot_u8x32_finalize_alder, nk_store_b128_haswell_,
                                    nk_partial_store_b32x4_haswell_, nk_load_b128_haswell_,
                                    nk_partial_load_b32x4_haswell_, nk_dots_reduce_sum_u8_stub_,
                                    /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/* E2M3 GEMM via DPBUSD integer path: depth_simd_dimensions=32 (32 e2m3s = 32 bytes = AVX2 register width) */
nk_define_cross_pack_size_(dots, e2m3, alder, e2m3, e2m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e2m3, alder, e2m3, e2m3, nk_assign_from_to_, /*norm_value_type=*/f32,
                      nk_dots_reduce_sumsq_e2m3_, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e2m3, alder, e2m3, f32, nk_b256_vec_t, nk_dot_e2m3x32_state_alder_t, nk_b128_vec_t,
                           nk_dot_e2m3x32_init_alder, nk_load_b256_haswell_, nk_partial_load_b8x32_serial_,
                           nk_dot_e2m3x32_update_alder, nk_dot_e2m3x32_finalize_alder, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_haswell_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e2m3, alder, e2m3, e2m3, f32, nk_b256_vec_t, nk_dot_e2m3x32_state_alder_t, nk_b128_vec_t,
                        nk_dot_e2m3x32_init_alder, nk_load_b256_haswell_, nk_partial_load_b8x32_serial_,
                        nk_load_b256_haswell_, nk_partial_load_b8x32_serial_, nk_dot_e2m3x32_update_alder,
                        nk_dot_e2m3x32_finalize_alder, nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_ALDER
#endif // NK_TARGET_X86_
#endif // NK_DOTS_ALDER_H
