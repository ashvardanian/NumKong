/**
 *  @file include/numkong/dots/alder.h
 *  @author Ash Vardanian
 *  @date March 4, 2026
 *  @brief SIMD-accelerated Batched Dot Products for Alder Lake.
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses AVX-VNNI, 256-bit, for integer GEMM via DPBUSD with algebraic sign transformations for
 *  signed*signed and unsigned*unsigned cases.
 */
#ifndef NUMKONG_DOTS_ALDER_H
#define NUMKONG_DOTS_ALDER_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_TARGET_ALDER

#include "numkong/dot/alder.h"      // Alder-specific dot product helpers
#include "numkong/dot/haswell.h"    // Haswell partial load functions
#include "numkong/cast/haswell.h"   // `nk_partial_load_b8x32_haswell_`
#include "numkong/dots/serial.h"    // GEMM macro definitions
#include "numkong/reduce/haswell.h" // `nk_reduce_moments_i8_haswell_contiguous_`
#include "numkong/reduce/alder.h"   // `nk_reduce_moments_u8_alder_contiguous_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni")
#endif

#pragma region Norms

NUMKONG_INLINE void nk_dots_reduce_moments_i8_alder_(nk_i8_t const *data, nk_size_t count, nk_size_t stride,
                                                     nk_i32_t *sum, nk_u32_t *norm) {
    nk_i64_t row_sum;
    nk_u64_t row_sumsq;
    nk_unused_(stride);
    nk_reduce_moments_i8_haswell_contiguous_(data, count, &row_sum, &row_sumsq);
    *sum = (nk_i32_t)row_sum, *norm = (nk_u32_t)row_sumsq;
}

NUMKONG_INLINE void nk_dots_reduce_moments_u8_alder_(nk_u8_t const *data, nk_size_t count, nk_size_t stride,
                                                     nk_u32_t *sum, nk_u32_t *norm) {
    nk_u64_t row_sum, row_sumsq;
    nk_unused_(stride);
    nk_reduce_moments_u8_alder_contiguous_(data, count, &row_sum, &row_sumsq);
    *sum = (nk_u32_t)row_sum, *norm = (nk_u32_t)row_sumsq;
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_i8_alder_(nk_i8_t const *data, nk_size_t count, nk_size_t stride) {
    nk_i32_t sum;
    nk_u32_t norm;
    nk_dots_reduce_moments_i8_alder_(data, count, stride, &sum, &norm);
    return norm;
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_u8_alder_(nk_u8_t const *data, nk_size_t count, nk_size_t stride) {
    nk_u32_t sum, norm;
    nk_dots_reduce_moments_u8_alder_(data, count, stride, &sum, &norm);
    return norm;
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e2m3_alder_(nk_e2m3_t const *data, nk_size_t count, nk_size_t stride) {
    nk_f32_t sum, sumsq;
    nk_unused_(stride);
    nk_reduce_moments_e2m3_alder_contiguous_(data, count, &sum, &sumsq);
    return sumsq;
}

#pragma endregion Norms

/* I8 GEMM: depth_simd_dimensions=32 — compensated (B sums precomputed in pack) */
nk_define_cross_packed_shape_(dots, i8, alder)
nk_define_cross_compensated_pack_size_(dots, i8, alder, i8, i8,
                                       /*sum_value_type=*/i32, /*norm_value_type=*/u32,
                                       /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_compensated_pack_(dots, i8, alder, i8, i8, nk_b128_vec_t, nk_load_b128_haswell_,
                                  nk_partial_load_b8x16_haswell_, nk_store_b128_haswell_,
                                  nk_partial_store_b8x16_haswell_,
                                  /*simd_width=*/16, /*sum_value_type=*/i32, /*norm_value_type=*/u32,
                                  nk_dots_reduce_moments_i8_alder_, /*depth_simd_dimensions=*/16,
                                  /*dimensions_per_value=*/1)
nk_define_cross_compensated_symmetric_(dots, i8, alder, i8, i32,
                                       /*sum_value_type=*/i32, /*norm_value_type=*/u32, nk_b256_vec_t,
                                       nk_dot_i8x32_state_alder_t, nk_b128_vec_t, nk_dot_i8x32_init_alder,
                                       nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_, nk_dot_i8x32_update_alder,
                                       nk_dot_i8x32_finalize_alder, nk_store_b128_haswell_,
                                       nk_partial_store_b32x4_haswell_, nk_load_b128_haswell_,
                                       nk_partial_load_b32x4_haswell_, nk_sum_i8x32_state_alder_t,
                                       nk_sum_i8x32_init_alder, nk_sum_i8x32_update_alder, nk_sum_i8x32_finalize_alder,
                                       /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_compensated_packed_(dots, i8, alder, i8, i8, i32,
                                    /*sum_value_type=*/i32, /*norm_value_type=*/u32, nk_b256_vec_t,
                                    nk_dot_i8x32_state_alder_t, nk_b128_vec_t, nk_dot_i8x32_init_alder,
                                    nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_, nk_load_b256_haswell_,
                                    nk_partial_load_b8x32_haswell_, nk_dot_i8x32_update_alder,
                                    nk_dot_i8x32_finalize_alder, nk_store_b128_haswell_,
                                    nk_partial_store_b32x4_haswell_, nk_load_b128_haswell_,
                                    nk_partial_load_b32x4_haswell_, nk_dots_reduce_sum_i8_stub_,
                                    /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=32 — compensated (operand swap, B sums precomputed) */
nk_define_cross_packed_shape_(dots, u8, alder)
nk_define_cross_compensated_pack_size_(dots, u8, alder, u8, u8,
                                       /*sum_value_type=*/u32, /*norm_value_type=*/u32,
                                       /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_compensated_pack_(dots, u8, alder, u8, u8, nk_b128_vec_t, nk_load_b128_haswell_,
                                  nk_partial_load_b8x16_haswell_, nk_store_b128_haswell_,
                                  nk_partial_store_b8x16_haswell_,
                                  /*simd_width=*/16, /*sum_value_type=*/u32, /*norm_value_type=*/u32,
                                  nk_dots_reduce_moments_u8_alder_, /*depth_simd_dimensions=*/16,
                                  /*dimensions_per_value=*/1)
nk_define_cross_compensated_symmetric_(dots, u8, alder, u8, u32,
                                       /*sum_value_type=*/u32, /*norm_value_type=*/u32, nk_b256_vec_t,
                                       nk_dot_u8x32_state_alder_t, nk_b128_vec_t, nk_dot_u8x32_init_alder,
                                       nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_, nk_dot_u8x32_update_alder,
                                       nk_dot_u8x32_finalize_alder, nk_store_b128_haswell_,
                                       nk_partial_store_b32x4_haswell_, nk_load_b128_haswell_,
                                       nk_partial_load_b32x4_haswell_, nk_sum_u8x32_state_alder_t,
                                       nk_sum_u8x32_init_alder, nk_sum_u8x32_update_alder, nk_sum_u8x32_finalize_alder,
                                       /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_compensated_packed_(dots, u8, alder, u8, u8, u32,
                                    /*sum_value_type=*/u32, /*norm_value_type=*/u32, nk_b256_vec_t,
                                    nk_dot_u8x32_state_alder_t, nk_b128_vec_t, nk_dot_u8x32_init_alder,
                                    nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_, nk_load_b256_haswell_,
                                    nk_partial_load_b8x32_haswell_, nk_dot_u8x32_update_alder,
                                    nk_dot_u8x32_finalize_alder, nk_store_b128_haswell_,
                                    nk_partial_store_b32x4_haswell_, nk_load_b128_haswell_,
                                    nk_partial_load_b32x4_haswell_, nk_dots_reduce_sum_u8_stub_,
                                    /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/*  E2M3 GEMM via the DPBUSD integer path: depth_simd_dimensions = 32, as 32 e2m3s span the 32 bytes
 *  of an AVX2 register. */
nk_define_cross_pack_size_(dots, e2m3, alder, e2m3, e2m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_(dots, e2m3, alder)
nk_define_cross_pack_(dots, e2m3, alder, e2m3, e2m3, nk_b256_vec_t, nk_load_b256_haswell_,
                      nk_partial_load_b8x32_haswell_, nk_store_b256_haswell_, nk_partial_store_b8x32_haswell_,
                      /*simd_width=*/32, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m3_alder_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e2m3, alder, e2m3, f32, nk_b256_vec_t, nk_dot_e2m3x32_state_alder_t, nk_b128_vec_t,
                           nk_dot_e2m3x32_init_alder, nk_cross_unscaled_, nk_load_b256_haswell_,
                           nk_partial_load_b8x32_haswell_, nk_dot_e2m3x32_update_alder, nk_dot_e2m3x32_finalize_alder,
                           nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e2m3, alder, e2m3, e2m3, f32, nk_b256_vec_t, nk_dot_e2m3x32_state_alder_t, nk_b128_vec_t,
                        nk_dot_e2m3x32_init_alder, nk_cross_unscaled_, nk_load_b256_haswell_,
                        nk_partial_load_b8x32_haswell_, nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_,
                        nk_dot_e2m3x32_update_alder, nk_dot_e2m3x32_finalize_alder, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_haswell_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/*  E2M1 GEMM via the DPBUSD integer path: depth_simd_dimensions = 64, as 64 nibbles span the 32
 *  bytes of an AVX2 register. */
nk_define_cross_pack_size_(dots, e2m1, alder, e2m1x2, e2m1x2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/2)
nk_define_cross_packed_shape_(dots, e2m1, alder)
nk_define_cross_pack_(dots, e2m1, alder, e2m1x2, e2m1x2, nk_b256_vec_t, nk_load_b256_haswell_,
                      nk_partial_load_b8x32_haswell_, nk_store_b256_haswell_, nk_partial_store_b8x32_haswell_,
                      /*simd_width=*/32, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m1_,
                      /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2)
nk_define_cross_symmetric_(dots, e2m1, alder, e2m1x2, f32, nk_b256_vec_t, nk_dot_e2m1x64_state_alder_t, nk_b128_vec_t,
                           nk_dot_e2m1x64_init_alder, nk_cross_unscaled_, nk_load_b256_haswell_,
                           nk_partial_load_b4x64_serial_, nk_dot_e2m1x64_update_alder, nk_dot_e2m1x64_finalize_alder,
                           nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, e2m1, alder, e2m1x2, e2m1x2, f32, nk_b256_vec_t, nk_dot_e2m1x64_state_alder_t,
                        nk_b128_vec_t, nk_dot_e2m1x64_init_alder, nk_cross_unscaled_, nk_load_b256_haswell_,
                        nk_partial_load_b4x64_serial_, nk_load_b256_haswell_, nk_partial_load_b4x64_serial_,
                        nk_dot_e2m1x64_update_alder, nk_dot_e2m1x64_finalize_alder, nk_store_b128_haswell_,
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

#endif // NUMKONG_TARGET_ALDER
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_DOTS_ALDER_H
