/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for Sierra Forest.
 *  @file include/numkong/spatial/sierra.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_sierra_instructions AVXVNNIINT8 Instructions Performance
 *
 *      Intrinsic                   Instruction                     Sierra Forest
 *      _mm256_dpbssds_epi32        VPDPBSSDS (YMM, YMM, YMM)       4cy @ p05
 *      _mm256_dpbssd_epi32         VPDPBSSD (YMM, YMM, YMM)        4cy @ p05
 *      _mm256_dpbuud_epi32         VPDPBUUD (YMM, YMM, YMM)        4cy @ p05
 *      _mm_rsqrt_ps                VRSQRTPS (XMM, XMM)             5cy @ p0
 *      _mm_sqrt_ss                 VSQRTSS (XMM, XMM, XMM)        12cy @ p0
 *
 *  Sierra Forest (AVXVNNIINT8) provides native signed x signed and unsigned x unsigned
 *  dot products, eliminating the need for algebraic corrections required on Alder Lake.
 *  This gives ~2.6x throughput over Haswell and ~1.3x over Alder for spatial kernels.
 */
#ifndef NK_SPATIAL_SIERRA_H
#define NK_SPATIAL_SIERRA_H

#if NK_TARGET_X86_
#if NK_TARGET_SIERRA

#include "numkong/types.h"
#include "numkong/scalar/haswell.h" // `nk_f32_sqrt_haswell`
#include "numkong/reduce/haswell.h" // `nk_reduce_add_i32x8_haswell_`
#include "numkong/cast/serial.h"    // `nk_partial_load_b8x32_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni,avxvnniint8"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni", "avxvnniint8")
#endif

NK_PUBLIC void nk_angular_i8_sierra(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {

    __m256i dot_product_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_i32x8 = _mm256_setzero_si256();

    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8x32 = _mm256_loadu_si256((__m256i const *)(a + i));
        __m256i b_i8x32 = _mm256_loadu_si256((__m256i const *)(b + i));
        dot_product_i32x8 = _mm256_dpbssds_epi32(dot_product_i32x8, a_i8x32, b_i8x32);
        a_norm_sq_i32x8 = _mm256_dpbssds_epi32(a_norm_sq_i32x8, a_i8x32, a_i8x32);
        b_norm_sq_i32x8 = _mm256_dpbssds_epi32(b_norm_sq_i32x8, b_i8x32, b_i8x32);
    }

    nk_i32_t dot_product_i32 = nk_reduce_add_i32x8_haswell_(dot_product_i32x8);
    nk_i32_t a_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(a_norm_sq_i32x8);
    nk_i32_t b_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(b_norm_sq_i32x8);

    for (; i < n; ++i) {
        nk_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = nk_angular_normalize_f32_haswell_(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}

NK_PUBLIC void nk_sqeuclidean_i8_sierra(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result) {
    // ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b) using dpbssds (signed x signed)

    __m256i dot_product_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_i32x8 = _mm256_setzero_si256();

    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8x32 = _mm256_loadu_si256((__m256i const *)(a + i));
        __m256i b_i8x32 = _mm256_loadu_si256((__m256i const *)(b + i));
        dot_product_i32x8 = _mm256_dpbssds_epi32(dot_product_i32x8, a_i8x32, b_i8x32);
        a_norm_sq_i32x8 = _mm256_dpbssds_epi32(a_norm_sq_i32x8, a_i8x32, a_i8x32);
        b_norm_sq_i32x8 = _mm256_dpbssds_epi32(b_norm_sq_i32x8, b_i8x32, b_i8x32);
    }

    nk_i32_t dot_product_i32 = nk_reduce_add_i32x8_haswell_(dot_product_i32x8);
    nk_i32_t a_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(a_norm_sq_i32x8);
    nk_i32_t b_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(b_norm_sq_i32x8);

    for (; i < n; ++i) {
        nk_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = (nk_u32_t)(a_norm_sq_i32 + b_norm_sq_i32 - 2 * dot_product_i32);
}

NK_PUBLIC void nk_euclidean_i8_sierra(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq_u32;
    nk_sqeuclidean_i8_sierra(a, b, n, &distance_sq_u32);
    *result = nk_f32_sqrt_haswell((nk_f32_t)distance_sq_u32);
}

NK_PUBLIC void nk_angular_u8_sierra(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {

    __m256i dot_product_u32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_u32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_u32x8 = _mm256_setzero_si256();

    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_u8x32 = _mm256_loadu_si256((__m256i const *)(a + i));
        __m256i b_u8x32 = _mm256_loadu_si256((__m256i const *)(b + i));
        dot_product_u32x8 = _mm256_dpbuud_epi32(dot_product_u32x8, a_u8x32, b_u8x32);
        a_norm_sq_u32x8 = _mm256_dpbuud_epi32(a_norm_sq_u32x8, a_u8x32, a_u8x32);
        b_norm_sq_u32x8 = _mm256_dpbuud_epi32(b_norm_sq_u32x8, b_u8x32, b_u8x32);
    }

    nk_i32_t dot_product_i32 = nk_reduce_add_i32x8_haswell_(dot_product_u32x8);
    nk_i32_t a_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(a_norm_sq_u32x8);
    nk_i32_t b_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(b_norm_sq_u32x8);

    for (; i < n; ++i) {
        nk_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = nk_angular_normalize_f32_haswell_(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}

NK_PUBLIC void nk_sqeuclidean_u8_sierra(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    // ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b) using dpbuud (unsigned x unsigned)

    __m256i dot_product_u32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_u32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_u32x8 = _mm256_setzero_si256();

    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_u8x32 = _mm256_loadu_si256((__m256i const *)(a + i));
        __m256i b_u8x32 = _mm256_loadu_si256((__m256i const *)(b + i));
        dot_product_u32x8 = _mm256_dpbuud_epi32(dot_product_u32x8, a_u8x32, b_u8x32);
        a_norm_sq_u32x8 = _mm256_dpbuud_epi32(a_norm_sq_u32x8, a_u8x32, a_u8x32);
        b_norm_sq_u32x8 = _mm256_dpbuud_epi32(b_norm_sq_u32x8, b_u8x32, b_u8x32);
    }

    nk_i32_t dot_product_i32 = nk_reduce_add_i32x8_haswell_(dot_product_u32x8);
    nk_i32_t a_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(a_norm_sq_u32x8);
    nk_i32_t b_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(b_norm_sq_u32x8);

    for (; i < n; ++i) {
        nk_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = (nk_u32_t)(a_norm_sq_i32 + b_norm_sq_i32 - 2 * dot_product_i32);
}

NK_PUBLIC void nk_euclidean_u8_sierra(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq_u32;
    nk_sqeuclidean_u8_sierra(a, b, n, &distance_sq_u32);
    *result = nk_f32_sqrt_haswell((nk_f32_t)distance_sq_u32);
}

NK_PUBLIC void nk_angular_e2m3_sierra(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                                      nk_f32_t *result) {
    // Angular distance for e2m3 using dual-VPSHUFB LUT + VPDPBSSD norm decomposition.
    // Every e2m3 value × 16 is an exact integer in [-120, +120].
    // DPBSSD(signed, signed) eliminates the need for unsigned conversion tricks.
    //
    __m256i const lut_lower_u8x32 = _mm256_set_epi8(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 30, 28,
                                                    26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m256i const lut_upper_u8x32 = _mm256_set_epi8(120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32,
                                                    120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32);
    __m256i const nibble_mask_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const magnitude_mask_u8x32 = _mm256_set1_epi8(0x1F);
    __m256i const half_select_u8x32 = _mm256_set1_epi8(0x10);
    __m256i const sign_mask_u8x32 = _mm256_set1_epi8(0x20);
    __m256i dot_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_i32x8 = _mm256_setzero_si256();
    __m256i a_e2m3_u8x32, b_e2m3_u8x32;

nk_angular_e2m3_sierra_cycle:
    if (count_scalars < 32) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b8x32_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x32_serial_(b_scalars, &b_vec, count_scalars);
        a_e2m3_u8x32 = a_vec.ymm;
        b_e2m3_u8x32 = b_vec.ymm;
        count_scalars = 0;
    }
    else {
        a_e2m3_u8x32 = _mm256_loadu_si256((__m256i const *)a_scalars);
        b_e2m3_u8x32 = _mm256_loadu_si256((__m256i const *)b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }

    // Decode a: extract magnitude, dual-VPSHUFB LUT, apply sign
    __m256i a_magnitude_u8x32 = _mm256_and_si256(a_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i a_shuffle_idx = _mm256_and_si256(a_magnitude_u8x32, nibble_mask_u8x32);
    __m256i a_upper_sel = _mm256_cmpeq_epi8(_mm256_and_si256(a_magnitude_u8x32, half_select_u8x32), half_select_u8x32);
    __m256i a_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, a_shuffle_idx),
                                                  _mm256_shuffle_epi8(lut_upper_u8x32, a_shuffle_idx), a_upper_sel);
    __m256i a_negate = _mm256_cmpeq_epi8(_mm256_and_si256(a_e2m3_u8x32, sign_mask_u8x32), sign_mask_u8x32);
    __m256i a_signed_i8x32 = _mm256_blendv_epi8(a_unsigned_u8x32,
                                                _mm256_sub_epi8(_mm256_setzero_si256(), a_unsigned_u8x32), a_negate);

    // Decode b: same LUT decode + sign
    __m256i b_magnitude_u8x32 = _mm256_and_si256(b_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i b_shuffle_idx = _mm256_and_si256(b_magnitude_u8x32, nibble_mask_u8x32);
    __m256i b_upper_sel = _mm256_cmpeq_epi8(_mm256_and_si256(b_magnitude_u8x32, half_select_u8x32), half_select_u8x32);
    __m256i b_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, b_shuffle_idx),
                                                  _mm256_shuffle_epi8(lut_upper_u8x32, b_shuffle_idx), b_upper_sel);
    __m256i b_negate = _mm256_cmpeq_epi8(_mm256_and_si256(b_e2m3_u8x32, sign_mask_u8x32), sign_mask_u8x32);
    __m256i b_signed_i8x32 = _mm256_blendv_epi8(b_unsigned_u8x32,
                                                _mm256_sub_epi8(_mm256_setzero_si256(), b_unsigned_u8x32), b_negate);

    // VPDPBSSD: signed × signed → i32
    dot_i32x8 = _mm256_dpbssd_epi32(dot_i32x8, a_signed_i8x32, b_signed_i8x32);
    a_norm_i32x8 = _mm256_dpbssd_epi32(a_norm_i32x8, a_signed_i8x32, a_signed_i8x32);
    b_norm_i32x8 = _mm256_dpbssd_epi32(b_norm_i32x8, b_signed_i8x32, b_signed_i8x32);

    if (count_scalars) goto nk_angular_e2m3_sierra_cycle;

    nk_i32_t dot_i32 = nk_reduce_add_i32x8_haswell_(dot_i32x8);
    nk_i32_t a_norm_i32 = nk_reduce_add_i32x8_haswell_(a_norm_i32x8);
    nk_i32_t b_norm_i32 = nk_reduce_add_i32x8_haswell_(b_norm_i32x8);
    *result = nk_angular_normalize_f32_haswell_(dot_i32, a_norm_i32, b_norm_i32);
}

NK_PUBLIC void nk_sqeuclidean_e2m3_sierra(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars,
                                          nk_size_t count_scalars, nk_f32_t *result) {
    // Squared Euclidean distance for e2m3 using norm decomposition + VPDPBSSD.
    // ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b)
    //
    __m256i const lut_lower_u8x32 = _mm256_set_epi8(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 30, 28,
                                                    26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m256i const lut_upper_u8x32 = _mm256_set_epi8(120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32,
                                                    120, 112, 104, 96, 88, 80, 72, 64, 60, 56, 52, 48, 44, 40, 36, 32);
    __m256i const nibble_mask_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const magnitude_mask_u8x32 = _mm256_set1_epi8(0x1F);
    __m256i const half_select_u8x32 = _mm256_set1_epi8(0x10);
    __m256i const sign_mask_u8x32 = _mm256_set1_epi8(0x20);
    __m256i dot_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_i32x8 = _mm256_setzero_si256();
    __m256i a_e2m3_u8x32, b_e2m3_u8x32;

nk_sqeuclidean_e2m3_sierra_cycle:
    if (count_scalars < 32) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b8x32_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x32_serial_(b_scalars, &b_vec, count_scalars);
        a_e2m3_u8x32 = a_vec.ymm;
        b_e2m3_u8x32 = b_vec.ymm;
        count_scalars = 0;
    }
    else {
        a_e2m3_u8x32 = _mm256_loadu_si256((__m256i const *)a_scalars);
        b_e2m3_u8x32 = _mm256_loadu_si256((__m256i const *)b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }

    // Decode a
    __m256i a_magnitude_u8x32 = _mm256_and_si256(a_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i a_shuffle_idx = _mm256_and_si256(a_magnitude_u8x32, nibble_mask_u8x32);
    __m256i a_upper_sel = _mm256_cmpeq_epi8(_mm256_and_si256(a_magnitude_u8x32, half_select_u8x32), half_select_u8x32);
    __m256i a_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, a_shuffle_idx),
                                                  _mm256_shuffle_epi8(lut_upper_u8x32, a_shuffle_idx), a_upper_sel);
    __m256i a_negate = _mm256_cmpeq_epi8(_mm256_and_si256(a_e2m3_u8x32, sign_mask_u8x32), sign_mask_u8x32);
    __m256i a_signed_i8x32 = _mm256_blendv_epi8(a_unsigned_u8x32,
                                                _mm256_sub_epi8(_mm256_setzero_si256(), a_unsigned_u8x32), a_negate);

    // Decode b
    __m256i b_magnitude_u8x32 = _mm256_and_si256(b_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i b_shuffle_idx = _mm256_and_si256(b_magnitude_u8x32, nibble_mask_u8x32);
    __m256i b_upper_sel = _mm256_cmpeq_epi8(_mm256_and_si256(b_magnitude_u8x32, half_select_u8x32), half_select_u8x32);
    __m256i b_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, b_shuffle_idx),
                                                  _mm256_shuffle_epi8(lut_upper_u8x32, b_shuffle_idx), b_upper_sel);
    __m256i b_negate = _mm256_cmpeq_epi8(_mm256_and_si256(b_e2m3_u8x32, sign_mask_u8x32), sign_mask_u8x32);
    __m256i b_signed_i8x32 = _mm256_blendv_epi8(b_unsigned_u8x32,
                                                _mm256_sub_epi8(_mm256_setzero_si256(), b_unsigned_u8x32), b_negate);

    dot_i32x8 = _mm256_dpbssd_epi32(dot_i32x8, a_signed_i8x32, b_signed_i8x32);
    a_norm_i32x8 = _mm256_dpbssd_epi32(a_norm_i32x8, a_signed_i8x32, a_signed_i8x32);
    b_norm_i32x8 = _mm256_dpbssd_epi32(b_norm_i32x8, b_signed_i8x32, b_signed_i8x32);

    if (count_scalars) goto nk_sqeuclidean_e2m3_sierra_cycle;

    nk_i32_t dot_i32 = nk_reduce_add_i32x8_haswell_(dot_i32x8);
    nk_i32_t a_norm_i32 = nk_reduce_add_i32x8_haswell_(a_norm_i32x8);
    nk_i32_t b_norm_i32 = nk_reduce_add_i32x8_haswell_(b_norm_i32x8);
    *result = (nk_f32_t)(a_norm_i32 + b_norm_i32 - 2 * dot_i32) / 256.0f;
}

NK_PUBLIC void nk_euclidean_e2m3_sierra(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e2m3_sierra(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

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
#endif // NK_SPATIAL_SIERRA_H
