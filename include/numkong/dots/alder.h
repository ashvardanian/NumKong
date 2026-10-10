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
#include "numkong/reduce/haswell.h" // `nk_reduce_moments_contiguous_i8_haswell_`
#include "numkong/reduce/alder.h"   // `nk_reduce_moments_contiguous_u8_alder_`

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
    nk_reduce_moments_contiguous_i8_haswell_(data, count, &row_sum, &row_sumsq);
    *sum = (nk_i32_t)row_sum, *norm = (nk_u32_t)row_sumsq;
}

NUMKONG_INLINE void nk_dots_reduce_moments_u8_alder_(nk_u8_t const *data, nk_size_t count, nk_size_t stride,
                                                     nk_u32_t *sum, nk_u32_t *norm) {
    nk_u64_t row_sum, row_sumsq;
    nk_unused_(stride);
    nk_reduce_moments_contiguous_u8_alder_(data, count, &row_sum, &row_sumsq);
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
    nk_reduce_moments_contiguous_e2m3_alder_(data, count, &sum, &sumsq);
    return sumsq;
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e2m1_alder_(nk_e2m1x2_t const *data, nk_size_t count, nk_size_t stride) {
    nk_unused_(stride);
    // Quadrupled squares of the doubled magnitudes, indexed by the nibble without its sign bit.
    __m256i squares_lut_u8x32 = _mm256_broadcastsi128_si256(_mm_setr_epi8(0, 1, 4, 9, 16, 36, 64, (char)144, //
                                                                          0, 1, 4, 9, 16, 36, 64, (char)144));
    __m256i mask_0f_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i zero_u8x32 = _mm256_setzero_si256();
    __m256i squares_u64x4 = zero_u8x32;
    nk_size_t bytes = nk_size_divide_round_up_(count, 2);
    unsigned char const *ptr = (unsigned char const *)data;
    while (bytes > 0) {
        nk_b256_vec_t raw_vec;
        nk_size_t chunk = bytes < 32 ? bytes : 32;
        nk_partial_load_b8x32_haswell_(ptr, &raw_vec, chunk);
        __m256i low_u8x32 = _mm256_and_si256(raw_vec.ymm, mask_0f_u8x32);
        __m256i high_u8x32 = _mm256_and_si256(_mm256_srli_epi16(raw_vec.ymm, 4), mask_0f_u8x32);
        // An odd count leaves the last low nibble outside the row.
        if ((count & 1) && chunk == bytes) {
            nk_b256_vec_t keep_vec;
            keep_vec.ymm = mask_0f_u8x32;
            keep_vec.u8s[chunk - 1] = 0;
            low_u8x32 = _mm256_and_si256(low_u8x32, keep_vec.ymm);
        }
        squares_u64x4 = _mm256_add_epi64(
            squares_u64x4, _mm256_sad_epu8(_mm256_shuffle_epi8(squares_lut_u8x32, low_u8x32), zero_u8x32));
        squares_u64x4 = _mm256_add_epi64(
            squares_u64x4, _mm256_sad_epu8(_mm256_shuffle_epi8(squares_lut_u8x32, high_u8x32), zero_u8x32));
        ptr += chunk, bytes -= chunk;
    }
    return (nk_f32_t)(nk_u64_t)nk_reduce_add_i64x4_haswell_(squares_u64x4) * 0.25f;
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
                                  nk_dots_reduce_moments_i8_alder_, /*depth_simd_dimensions=*/32,
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
                                    nk_partial_load_b32x4_haswell_, nk_dots_reduce_sum_stub_i8_serial_,
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
                                  nk_dots_reduce_moments_u8_alder_, /*depth_simd_dimensions=*/32,
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
                                    nk_partial_load_b32x4_haswell_, nk_dots_reduce_sum_stub_u8_serial_,
                                    /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/*  E2M3 GEMM via the DPBUSD integer path: depth_simd_dimensions = 32, as 32 e2m3s span the 32 bytes
 *  of an AVX2 register. */
nk_define_cross_pack_size_(dots, e2m3, alder, e2m3, e2m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, e2m3, alder)
nk_define_cross_pack_(dots, e2m3, alder, e2m3, e2m3, nk_b256_vec_t, nk_load_b256_haswell_,
                      nk_partial_load_b8x32_haswell_, nk_store_b256_haswell_, nk_partial_store_b8x32_haswell_,
                      /*simd_width=*/32, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m3_alder_,
                      /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_serial_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, e2m3, alder, e2m3, f32, nk_b256_vec_t, nk_dot_e2m3x32_state_alder_t, nk_b128_vec_t,
                           nk_dot_e2m3x32_init_alder, nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_,
                           nk_dot_e2m3x32_update_alder, nk_dot_e2m3x32_finalize_alder, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_haswell_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e2m3, alder, e2m3, e2m3, f32, nk_b256_vec_t, nk_dot_e2m3x32_state_alder_t, nk_b128_vec_t,
                        nk_dot_e2m3x32_init_alder, nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_,
                        nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_, nk_dot_e2m3x32_update_alder,
                        nk_dot_e2m3x32_finalize_alder, nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/*  E2M1 GEMM via the DPBUSD integer path: depth_simd_dimensions = 64, as 64 nibbles span the 32
 *  bytes of an AVX2 register. */
nk_define_cross_pack_size_(dots, e2m1, alder, e2m1x2, e2m1x2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/2, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, e2m1, alder)
nk_define_cross_pack_(dots, e2m1, alder, e2m1x2, e2m1x2, nk_b256_vec_t, nk_load_b256_haswell_,
                      nk_partial_load_b8x32_haswell_, nk_store_b256_haswell_, nk_partial_store_b8x32_haswell_,
                      /*simd_width=*/32, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m1_alder_,
                      /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, nk_cross_pack_scales_bytes_serial_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, e2m1, alder, e2m1x2, f32, nk_b256_vec_t, nk_dot_e2m1x64_state_alder_t, nk_b128_vec_t,
                           nk_dot_e2m1x64_init_alder, nk_load_b256_haswell_, nk_partial_load_b4x64_serial_,
                           nk_dot_e2m1x64_update_alder, nk_dot_e2m1x64_finalize_alder, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_haswell_,
                           /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, e2m1, alder, e2m1x2, e2m1x2, f32, nk_b256_vec_t, nk_dot_e2m1x64_state_alder_t,
                        nk_b128_vec_t, nk_dot_e2m1x64_init_alder, nk_load_b256_haswell_, nk_partial_load_b4x64_serial_,
                        nk_load_b256_haswell_, nk_partial_load_b4x64_serial_, nk_dot_e2m1x64_update_alder,
                        nk_dot_e2m1x64_finalize_alder, nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_,
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
