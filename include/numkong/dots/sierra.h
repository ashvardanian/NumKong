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

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e2m1_sierra_(nk_e2m1x2_t const *data, nk_size_t count, nk_size_t stride) {
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

/* I8 GEMM: depth_simd_dimensions=32 (32 i8s = 32 bytes = AVX2 register width) */
nk_define_cross_pack_size_(dots, i8, sierra, i8, i8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, i8, sierra)
nk_define_cross_pack_(dots, i8, sierra, i8, i8, nk_b128_vec_t, nk_load_b128_haswell_, nk_partial_load_b8x16_haswell_,
                      nk_store_b128_haswell_, nk_partial_store_b8x16_haswell_, /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sumsq_i8_sierra_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_, /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, i8, sierra, i8, i32, nk_b256_vec_t, nk_dot_i8x32_state_sierra_t, nk_b128_vec_t,
                           nk_dot_i8x32_init_sierra, nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_,
                           nk_dot_i8x32_update_sierra, nk_dot_i8x32_finalize_sierra, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_haswell_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, i8, sierra, i8, i8, i32, nk_b256_vec_t, nk_dot_i8x32_state_sierra_t, nk_b128_vec_t,
                        nk_dot_i8x32_init_sierra, nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_,
                        nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_, nk_dot_i8x32_update_sierra,
                        nk_dot_i8x32_finalize_sierra, nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=32 (32 u8s = 32 bytes = AVX2 register width) */
nk_define_cross_pack_size_(dots, u8, sierra, u8, u8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, u8, sierra)
nk_define_cross_pack_(dots, u8, sierra, u8, u8, nk_b128_vec_t, nk_load_b128_haswell_, nk_partial_load_b8x16_haswell_,
                      nk_store_b128_haswell_, nk_partial_store_b8x16_haswell_, /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sumsq_u8_sierra_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_, /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, u8, sierra, u8, u32, nk_b256_vec_t, nk_dot_u8x32_state_sierra_t, nk_b128_vec_t,
                           nk_dot_u8x32_init_sierra, nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_,
                           nk_dot_u8x32_update_sierra, nk_dot_u8x32_finalize_sierra, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_haswell_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, u8, sierra, u8, u8, u32, nk_b256_vec_t, nk_dot_u8x32_state_sierra_t, nk_b128_vec_t,
                        nk_dot_u8x32_init_sierra, nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_,
                        nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_, nk_dot_u8x32_update_sierra,
                        nk_dot_u8x32_finalize_sierra, nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/*  E2M3 GEMM via the DPBUSD integer path: depth_simd_dimensions = 32, as 32 e2m3s span the 32 bytes
 *  of an AVX2 register. */
nk_define_cross_pack_size_(dots, e2m3, sierra, e2m3, e2m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, e2m3, sierra)
nk_define_cross_pack_(dots, e2m3, sierra, e2m3, e2m3, nk_b128_vec_t, nk_load_b128_haswell_,
                      nk_partial_load_b8x16_haswell_, nk_store_b128_haswell_, nk_partial_store_b8x16_haswell_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m3_sierra_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, e2m3, sierra, e2m3, f32, nk_b256_vec_t, nk_dot_e2m3x32_state_sierra_t, nk_b128_vec_t,
                           nk_dot_e2m3x32_init_sierra, nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_,
                           nk_dot_e2m3x32_update_sierra, nk_dot_e2m3x32_finalize_sierra, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_haswell_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e2m3, sierra, e2m3, e2m3, f32, nk_b256_vec_t, nk_dot_e2m3x32_state_sierra_t,
                        nk_b128_vec_t, nk_dot_e2m3x32_init_sierra, nk_load_b256_haswell_,
                        nk_partial_load_b8x32_haswell_, nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_,
                        nk_dot_e2m3x32_update_sierra, nk_dot_e2m3x32_finalize_sierra, nk_store_b128_haswell_,
                        nk_partial_store_b32x4_haswell_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/*  E2M1 GEMM via the DPBSSD integer path: depth_simd_dimensions = 64, as 64 nibbles span the 32
 *  bytes of an AVX2 register. */
nk_define_cross_pack_size_(dots, e2m1, sierra, e2m1x2, e2m1x2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/2, /*scale_bytes=*/1)
nk_define_cross_packed_shape_(dots, e2m1, sierra)
nk_define_cross_pack_(dots, e2m1, sierra, e2m1x2, e2m1x2, nk_b256_vec_t, nk_load_b256_haswell_,
                      nk_partial_load_b8x32_haswell_, nk_store_b256_haswell_, nk_partial_store_b8x32_haswell_,
                      /*simd_width=*/32, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m1_sierra_,
                      /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/2, nk_cross_pack_scales_bytes_,
                      /*scale_bytes=*/1)
nk_define_cross_symmetric_(dots, e2m1, sierra, e2m1x2, f32, nk_b256_vec_t, nk_dot_e2m1x64_state_sierra_t, nk_b128_vec_t,
                           nk_dot_e2m1x64_init_sierra, nk_load_b256_haswell_, nk_partial_load_b4x64_serial_,
                           nk_dot_e2m1x64_update_sierra, nk_dot_e2m1x64_finalize_sierra, nk_store_b128_haswell_,
                           nk_partial_store_b32x4_haswell_,
                           /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, e2m1, sierra, e2m1x2, e2m1x2, f32, nk_b256_vec_t, nk_dot_e2m1x64_state_sierra_t,
                        nk_b128_vec_t, nk_dot_e2m1x64_init_sierra, nk_load_b256_haswell_, nk_partial_load_b4x64_serial_,
                        nk_load_b256_haswell_, nk_partial_load_b4x64_serial_, nk_dot_e2m1x64_update_sierra,
                        nk_dot_e2m1x64_finalize_sierra, nk_store_b128_haswell_, nk_partial_store_b32x4_haswell_,
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
