/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for Alder Lake.
 *  @file include/numkong/spatial/alder.h
 *  @author Ash Vardanian
 *  @date March 4, 2026
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_alder_instructions AVX-VNNI Instructions Performance
 *
 *      Intrinsic                   Instruction                     Alder Lake  Raptor Lake
 *      _mm256_dpbusd_epi32         VPDPBUSD (YMM, YMM, YMM)        4cy @ p05   4cy @ p05
 *      _mm256_sad_epu8             VPSADBW (YMM, YMM, YMM)         3cy @ p5    3cy @ p5
 *      _mm256_xor_si256            VPXOR (YMM, YMM, YMM)           1cy @ p015  1cy @ p015
 *      _mm256_add_epi64            VPADDQ (YMM, YMM, YMM)          1cy @ p015  1cy @ p015
 *      _mm_rsqrt_ps                VRSQRTPS (XMM, XMM)             5cy @ p0    5cy @ p0
 *      _mm_sqrt_ss                 VSQRTSS (XMM, XMM, XMM)        12cy @ p0   12cy @ p0
 *
 *  All spatial kernels use the dpbusd norm-decomposition approach:
 *    ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b)
 *  This avoids the p5 bottleneck from unpack operations, achieving ~2x throughput
 *  over Haswell's subs+unpack+madd approach (16 elem/cy vs 8 elem/cy).
 */
#ifndef NK_SPATIAL_ALDER_H
#define NK_SPATIAL_ALDER_H

#if NK_TARGET_X86_
#if NK_TARGET_ALDER

#include "numkong/types.h"
#include "numkong/dot/alder.h"      // VEX compat macros + dpbusd helpers
#include "numkong/scalar/haswell.h" // `nk_f32_sqrt_haswell`
#include "numkong/reduce/haswell.h"
#include "numkong/cast/serial.h" // `nk_partial_load_b8x32_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni")
#endif

NK_PUBLIC void nk_angular_i8_alder(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
    // Angular distance using DPBUSD with algebraic transformation for signed x signed.
    //
    // For angular distance we need: dot(a,b), ||a||^2, ||b||^2
    // Using dpbusd(u8, i8) for asymmetric unsigned x signed:
    //   a' = a XOR 0x80 (signed -> unsigned), then dpbusd(a', b) = (a+128)*b
    //   a*b = dpbusd(a',b) - 128*sum(b)
    //
    // For norms: dpbusd(a', a) = (a+128)*a, so a^2 = dpbusd(a',a) - 128*sum(a)
    // Similarly for b: dpbusd(b', b) = (b+128)*b
    //
    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const zeros_u8x32 = _mm256_setzero_si256();
    __m256i dot_product_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_i32x8 = _mm256_setzero_si256();
    __m256i sum_a_biased_i64x4 = _mm256_setzero_si256();
    __m256i sum_b_biased_i64x4 = _mm256_setzero_si256();

    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8x32 = _mm256_loadu_si256((__m256i const *)(a + i));
        __m256i b_i8x32 = _mm256_loadu_si256((__m256i const *)(b + i));

        // Convert to unsigned for dpbusd
        __m256i a_unsigned_u8x32 = _mm256_xor_si256(a_i8x32, xor_mask_u8x32);
        __m256i b_unsigned_u8x32 = _mm256_xor_si256(b_i8x32, xor_mask_u8x32);

        // dpbusd: (a+128)*b, (a+128)*a, (b+128)*b
        dot_product_i32x8 = _mm256_dpbusd_avx_epi32(dot_product_i32x8, a_unsigned_u8x32, b_i8x32);
        a_norm_sq_i32x8 = _mm256_dpbusd_avx_epi32(a_norm_sq_i32x8, a_unsigned_u8x32, a_i8x32);
        b_norm_sq_i32x8 = _mm256_dpbusd_avx_epi32(b_norm_sq_i32x8, b_unsigned_u8x32, b_i8x32);

        // Accumulate biased sums for correction: sum(a+128), sum(b+128) via SAD
        sum_a_biased_i64x4 = _mm256_add_epi64(sum_a_biased_i64x4, _mm256_sad_epu8(a_unsigned_u8x32, zeros_u8x32));
        sum_b_biased_i64x4 = _mm256_add_epi64(sum_b_biased_i64x4, _mm256_sad_epu8(b_unsigned_u8x32, zeros_u8x32));
    }

    // Reduce and apply corrections inline:
    // correction_x = 128 * sum_x_biased - 16384 * elements_processed
    // value = reduce(accumulator) - correction
    nk_i64_t sum_a_biased = nk_reduce_add_i64x4_haswell_(sum_a_biased_i64x4);
    nk_i64_t sum_b_biased = nk_reduce_add_i64x4_haswell_(sum_b_biased_i64x4);
    nk_i64_t correction_a = 128LL * sum_a_biased - 16384LL * (nk_i64_t)i;
    nk_i64_t correction_b = 128LL * sum_b_biased - 16384LL * (nk_i64_t)i;

    nk_i32_t dot_product_i32 = nk_reduce_add_i32x8_haswell_(dot_product_i32x8) - (nk_i32_t)correction_b;
    nk_i32_t a_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(a_norm_sq_i32x8) - (nk_i32_t)correction_a;
    nk_i32_t b_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(b_norm_sq_i32x8) - (nk_i32_t)correction_b;

    // Scalar tail
    for (; i < n; ++i) {
        nk_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = nk_angular_normalize_f32_haswell_(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}

NK_PUBLIC void nk_sqeuclidean_i8_alder(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result) {
    // Squared Euclidean distance for i8 using DPBUSD with norm decomposition.
    // ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b)
    //
    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const zeros_u8x32 = _mm256_setzero_si256();
    __m256i dot_product_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_i32x8 = _mm256_setzero_si256();
    __m256i sum_a_biased_i64x4 = _mm256_setzero_si256();
    __m256i sum_b_biased_i64x4 = _mm256_setzero_si256();

    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8x32 = _mm256_loadu_si256((__m256i const *)(a + i));
        __m256i b_i8x32 = _mm256_loadu_si256((__m256i const *)(b + i));
        __m256i a_unsigned_u8x32 = _mm256_xor_si256(a_i8x32, xor_mask_u8x32);
        __m256i b_unsigned_u8x32 = _mm256_xor_si256(b_i8x32, xor_mask_u8x32);

        dot_product_i32x8 = _mm256_dpbusd_avx_epi32(dot_product_i32x8, a_unsigned_u8x32, b_i8x32);
        a_norm_sq_i32x8 = _mm256_dpbusd_avx_epi32(a_norm_sq_i32x8, a_unsigned_u8x32, a_i8x32);
        b_norm_sq_i32x8 = _mm256_dpbusd_avx_epi32(b_norm_sq_i32x8, b_unsigned_u8x32, b_i8x32);

        sum_a_biased_i64x4 = _mm256_add_epi64(sum_a_biased_i64x4, _mm256_sad_epu8(a_unsigned_u8x32, zeros_u8x32));
        sum_b_biased_i64x4 = _mm256_add_epi64(sum_b_biased_i64x4, _mm256_sad_epu8(b_unsigned_u8x32, zeros_u8x32));
    }

    nk_i64_t sum_a_biased = nk_reduce_add_i64x4_haswell_(sum_a_biased_i64x4);
    nk_i64_t sum_b_biased = nk_reduce_add_i64x4_haswell_(sum_b_biased_i64x4);
    nk_i64_t correction_a = 128LL * sum_a_biased - 16384LL * (nk_i64_t)i;
    nk_i64_t correction_b = 128LL * sum_b_biased - 16384LL * (nk_i64_t)i;

    nk_i32_t dot_product_i32 = nk_reduce_add_i32x8_haswell_(dot_product_i32x8) - (nk_i32_t)correction_b;
    nk_i32_t a_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(a_norm_sq_i32x8) - (nk_i32_t)correction_a;
    nk_i32_t b_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(b_norm_sq_i32x8) - (nk_i32_t)correction_b;

    for (; i < n; ++i) {
        nk_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    // ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b)
    *result = (nk_u32_t)(a_norm_sq_i32 + b_norm_sq_i32 - 2 * dot_product_i32);
}

NK_PUBLIC void nk_euclidean_i8_alder(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq_u32;
    nk_sqeuclidean_i8_alder(a, b, n, &distance_sq_u32);
    *result = nk_f32_sqrt_haswell((nk_f32_t)distance_sq_u32);
}

NK_PUBLIC void nk_sqeuclidean_u8_alder(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    // Squared Euclidean distance for u8 using DPBUSD with norm decomposition.
    // ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b)
    //
    // For u8 x u8: dpbusd(a, b'^0x80) = a*(b-128), so dot(a,b) = dpbusd(a,b') + 128*sum(a)
    // For norms: dpbusd(a, a'^0x80) = a*(a-128), so ||a||^2 = dpbusd(a,a') + 128*sum(a)
    //
    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const zeros_u8x32 = _mm256_setzero_si256();
    __m256i dot_product_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_i32x8 = _mm256_setzero_si256();
    __m256i sum_a_u64x4 = _mm256_setzero_si256();
    __m256i sum_b_u64x4 = _mm256_setzero_si256();

    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_u8x32 = _mm256_loadu_si256((__m256i const *)(a + i));
        __m256i b_u8x32 = _mm256_loadu_si256((__m256i const *)(b + i));
        __m256i a_signed_i8x32 = _mm256_xor_si256(a_u8x32, xor_mask_u8x32);
        __m256i b_signed_i8x32 = _mm256_xor_si256(b_u8x32, xor_mask_u8x32);

        // dpbusd(a, b-128) = a*(b-128), dpbusd(a, a-128) = a*(a-128), etc.
        dot_product_i32x8 = _mm256_dpbusd_avx_epi32(dot_product_i32x8, a_u8x32, b_signed_i8x32);
        a_norm_sq_i32x8 = _mm256_dpbusd_avx_epi32(a_norm_sq_i32x8, a_u8x32, a_signed_i8x32);
        b_norm_sq_i32x8 = _mm256_dpbusd_avx_epi32(b_norm_sq_i32x8, b_u8x32, b_signed_i8x32);

        sum_a_u64x4 = _mm256_add_epi64(sum_a_u64x4, _mm256_sad_epu8(a_u8x32, zeros_u8x32));
        sum_b_u64x4 = _mm256_add_epi64(sum_b_u64x4, _mm256_sad_epu8(b_u8x32, zeros_u8x32));
    }

    // Corrections: x*(y-128) + 128*sum(x) = x*y
    nk_i64_t sum_a_i64 = nk_reduce_add_i64x4_haswell_(sum_a_u64x4);
    nk_i64_t sum_b_i64 = nk_reduce_add_i64x4_haswell_(sum_b_u64x4);
    nk_i32_t dot_product_i32 = (nk_i32_t)((nk_i64_t)nk_reduce_add_i32x8_haswell_(dot_product_i32x8) +
                                          128LL * sum_a_i64);
    nk_i32_t a_norm_sq_i32 = (nk_i32_t)((nk_i64_t)nk_reduce_add_i32x8_haswell_(a_norm_sq_i32x8) + 128LL * sum_a_i64);
    nk_i32_t b_norm_sq_i32 = (nk_i32_t)((nk_i64_t)nk_reduce_add_i32x8_haswell_(b_norm_sq_i32x8) + 128LL * sum_b_i64);

    for (; i < n; ++i) {
        nk_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = (nk_u32_t)(a_norm_sq_i32 + b_norm_sq_i32 - 2 * dot_product_i32);
}

NK_PUBLIC void nk_euclidean_u8_alder(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq_u32;
    nk_sqeuclidean_u8_alder(a, b, n, &distance_sq_u32);
    *result = nk_f32_sqrt_haswell((nk_f32_t)distance_sq_u32);
}

NK_PUBLIC void nk_angular_u8_alder(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    // Angular distance for u8 using DPBUSD with algebraic transformation.
    // dpbusd(a, b'^0x80) = a*(b-128), so dot(a,b) = dpbusd(a,b') + 128*sum(a)
    //
    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const zeros_u8x32 = _mm256_setzero_si256();
    __m256i dot_product_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_i32x8 = _mm256_setzero_si256();
    __m256i sum_a_u64x4 = _mm256_setzero_si256();
    __m256i sum_b_u64x4 = _mm256_setzero_si256();

    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_u8x32 = _mm256_loadu_si256((__m256i const *)(a + i));
        __m256i b_u8x32 = _mm256_loadu_si256((__m256i const *)(b + i));
        __m256i a_signed_i8x32 = _mm256_xor_si256(a_u8x32, xor_mask_u8x32);
        __m256i b_signed_i8x32 = _mm256_xor_si256(b_u8x32, xor_mask_u8x32);

        dot_product_i32x8 = _mm256_dpbusd_avx_epi32(dot_product_i32x8, a_u8x32, b_signed_i8x32);
        a_norm_sq_i32x8 = _mm256_dpbusd_avx_epi32(a_norm_sq_i32x8, a_u8x32, a_signed_i8x32);
        b_norm_sq_i32x8 = _mm256_dpbusd_avx_epi32(b_norm_sq_i32x8, b_u8x32, b_signed_i8x32);

        sum_a_u64x4 = _mm256_add_epi64(sum_a_u64x4, _mm256_sad_epu8(a_u8x32, zeros_u8x32));
        sum_b_u64x4 = _mm256_add_epi64(sum_b_u64x4, _mm256_sad_epu8(b_u8x32, zeros_u8x32));
    }

    nk_i64_t sum_a_i64 = nk_reduce_add_i64x4_haswell_(sum_a_u64x4);
    nk_i64_t sum_b_i64 = nk_reduce_add_i64x4_haswell_(sum_b_u64x4);
    nk_i32_t dot_product_i32 = (nk_i32_t)((nk_i64_t)nk_reduce_add_i32x8_haswell_(dot_product_i32x8) +
                                          128LL * sum_a_i64);
    nk_i32_t a_norm_sq_i32 = (nk_i32_t)((nk_i64_t)nk_reduce_add_i32x8_haswell_(a_norm_sq_i32x8) + 128LL * sum_a_i64);
    nk_i32_t b_norm_sq_i32 = (nk_i32_t)((nk_i64_t)nk_reduce_add_i32x8_haswell_(b_norm_sq_i32x8) + 128LL * sum_b_i64);

    for (; i < n; ++i) {
        nk_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = nk_angular_normalize_f32_haswell_(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}

NK_PUBLIC void nk_angular_e2m3_alder(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                                     nk_f32_t *result) {
    // Angular distance for e2m3 using dual-VPSHUFB LUT + VPDPBUSD norm decomposition.
    // Every e2m3 value × 16 is an exact integer in [-120, +120].
    // We compute dot(a,b), ||a||^2, ||b||^2 in scaled integer domain,
    // then normalize: angular = 1 - dot / sqrt(||a||^2 * ||b||^2).
    // Final division by 256.0f for dot and norms cancels in the ratio.
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

nk_angular_e2m3_alder_cycle:
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

    // Decode a: extract magnitude, dual-VPSHUFB LUT
    __m256i a_magnitude_u8x32 = _mm256_and_si256(a_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i a_shuffle_idx = _mm256_and_si256(a_magnitude_u8x32, nibble_mask_u8x32);
    __m256i a_upper_sel = _mm256_cmpeq_epi8(_mm256_and_si256(a_magnitude_u8x32, half_select_u8x32), half_select_u8x32);
    __m256i a_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, a_shuffle_idx),
                                                  _mm256_shuffle_epi8(lut_upper_u8x32, a_shuffle_idx), a_upper_sel);

    // Decode b: same LUT decode
    __m256i b_magnitude_u8x32 = _mm256_and_si256(b_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i b_shuffle_idx = _mm256_and_si256(b_magnitude_u8x32, nibble_mask_u8x32);
    __m256i b_upper_sel = _mm256_cmpeq_epi8(_mm256_and_si256(b_magnitude_u8x32, half_select_u8x32), half_select_u8x32);
    __m256i b_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, b_shuffle_idx),
                                                  _mm256_shuffle_epi8(lut_upper_u8x32, b_shuffle_idx), b_upper_sel);

    // Dot product with sign: combined sign from (a XOR b) & 0x20
    __m256i sign_combined = _mm256_and_si256(_mm256_xor_si256(a_e2m3_u8x32, b_e2m3_u8x32), sign_mask_u8x32);
    __m256i negate_mask = _mm256_cmpeq_epi8(sign_combined, sign_mask_u8x32);
    __m256i b_negated = _mm256_sub_epi8(_mm256_setzero_si256(), b_unsigned_u8x32);
    __m256i b_dot_i8x32 = _mm256_blendv_epi8(b_unsigned_u8x32, b_negated, negate_mask);

    // DPBUSD: a_unsigned[u8] × b_signed[i8] → i32 for dot product
    dot_i32x8 = _mm256_dpbusd_avx_epi32(dot_i32x8, a_unsigned_u8x32, b_dot_i8x32);
    // Norms: magnitude² is always positive, DPBUSD(unsigned, unsigned-as-signed) works since max=120 < 127
    a_norm_i32x8 = _mm256_dpbusd_avx_epi32(a_norm_i32x8, a_unsigned_u8x32, a_unsigned_u8x32);
    b_norm_i32x8 = _mm256_dpbusd_avx_epi32(b_norm_i32x8, b_unsigned_u8x32, b_unsigned_u8x32);

    if (count_scalars) goto nk_angular_e2m3_alder_cycle;

    nk_i32_t dot_i32 = nk_reduce_add_i32x8_haswell_(dot_i32x8);
    nk_i32_t a_norm_i32 = nk_reduce_add_i32x8_haswell_(a_norm_i32x8);
    nk_i32_t b_norm_i32 = nk_reduce_add_i32x8_haswell_(b_norm_i32x8);
    // The 256.0f factor cancels in the angular normalization ratio
    *result = nk_angular_normalize_f32_haswell_(dot_i32, a_norm_i32, b_norm_i32);
}

NK_PUBLIC void nk_sqeuclidean_e2m3_alder(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars,
                                         nk_size_t count_scalars, nk_f32_t *result) {
    // Squared Euclidean distance for e2m3 using norm decomposition:
    // ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b)
    // Each value × 16 is exact integer, so result = integer_result / 256.0f
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

nk_sqeuclidean_e2m3_alder_cycle:
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

    // Decode a and b magnitudes via LUT
    __m256i a_magnitude_u8x32 = _mm256_and_si256(a_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i a_shuffle_idx = _mm256_and_si256(a_magnitude_u8x32, nibble_mask_u8x32);
    __m256i a_upper_sel = _mm256_cmpeq_epi8(_mm256_and_si256(a_magnitude_u8x32, half_select_u8x32), half_select_u8x32);
    __m256i a_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, a_shuffle_idx),
                                                  _mm256_shuffle_epi8(lut_upper_u8x32, a_shuffle_idx), a_upper_sel);

    __m256i b_magnitude_u8x32 = _mm256_and_si256(b_e2m3_u8x32, magnitude_mask_u8x32);
    __m256i b_shuffle_idx = _mm256_and_si256(b_magnitude_u8x32, nibble_mask_u8x32);
    __m256i b_upper_sel = _mm256_cmpeq_epi8(_mm256_and_si256(b_magnitude_u8x32, half_select_u8x32), half_select_u8x32);
    __m256i b_unsigned_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lower_u8x32, b_shuffle_idx),
                                                  _mm256_shuffle_epi8(lut_upper_u8x32, b_shuffle_idx), b_upper_sel);

    // Signed dot product: combined sign from (a XOR b) & 0x20
    __m256i sign_combined = _mm256_and_si256(_mm256_xor_si256(a_e2m3_u8x32, b_e2m3_u8x32), sign_mask_u8x32);
    __m256i negate_mask = _mm256_cmpeq_epi8(sign_combined, sign_mask_u8x32);
    __m256i b_negated = _mm256_sub_epi8(_mm256_setzero_si256(), b_unsigned_u8x32);
    __m256i b_dot_i8x32 = _mm256_blendv_epi8(b_unsigned_u8x32, b_negated, negate_mask);

    dot_i32x8 = _mm256_dpbusd_avx_epi32(dot_i32x8, a_unsigned_u8x32, b_dot_i8x32);
    a_norm_i32x8 = _mm256_dpbusd_avx_epi32(a_norm_i32x8, a_unsigned_u8x32, a_unsigned_u8x32);
    b_norm_i32x8 = _mm256_dpbusd_avx_epi32(b_norm_i32x8, b_unsigned_u8x32, b_unsigned_u8x32);

    if (count_scalars) goto nk_sqeuclidean_e2m3_alder_cycle;

    nk_i32_t dot_i32 = nk_reduce_add_i32x8_haswell_(dot_i32x8);
    nk_i32_t a_norm_i32 = nk_reduce_add_i32x8_haswell_(a_norm_i32x8);
    nk_i32_t b_norm_i32 = nk_reduce_add_i32x8_haswell_(b_norm_i32x8);
    // ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b), scaled by 256
    *result = (nk_f32_t)(a_norm_i32 + b_norm_i32 - 2 * dot_i32) / 256.0f;
}

NK_PUBLIC void nk_euclidean_e2m3_alder(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e2m3_alder(a, b, n, result);
    *result = nk_f32_sqrt_haswell(*result);
}

NK_PUBLIC void nk_angular_e3m2_alder(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars, nk_size_t count_scalars,
                                     nk_f32_t *result) {
    // Angular distance for e3m2 using dual-VPSHUFB LUT decode to i16 + VPDPWSSD norm decomposition.
    // Every e3m2 value × 16 is an exact integer (max magnitude 448), requiring i16.
    // VPDPWSSD replaces Haswell's VPMADDWD + VPADDD, saving one instruction per accumulation.
    //
    __m256i const lut_lo_lower_u8x32 = _mm256_set_epi8(        //
        28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0, //
        28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m256i const lut_lo_upper_u8x32 = _mm256_set_epi8(                                                           //
        (char)192, (char)128, 64, 0, (char)224, (char)192, (char)160, (char)128, 112, 96, 80, 64, 56, 48, 40, 32, //
        (char)192, (char)128, 64, 0, (char)224, (char)192, (char)160, (char)128, 112, 96, 80, 64, 56, 48, 40, 32);
    __m256i const nibble_mask_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const magnitude_mask_u8x32 = _mm256_set1_epi8(0x1F);
    __m256i const half_select_u8x32 = _mm256_set1_epi8(0x10);
    __m256i const sign_mask_u8x32 = _mm256_set1_epi8(0x20);
    __m256i const high_threshold_u8x32 = _mm256_set1_epi8(27);
    __m256i const ones_u8x32 = _mm256_set1_epi8(1);
    __m256i const ones_i16x16 = _mm256_set1_epi16(1);
    __m256i dot_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_i32x8 = _mm256_setzero_si256();
    __m256i a_e3m2_u8x32, b_e3m2_u8x32;

nk_angular_e3m2_alder_cycle:
    if (count_scalars < 32) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b8x32_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x32_serial_(b_scalars, &b_vec, count_scalars);
        a_e3m2_u8x32 = a_vec.ymm;
        b_e3m2_u8x32 = b_vec.ymm;
        count_scalars = 0;
    }
    else {
        a_e3m2_u8x32 = _mm256_loadu_si256((__m256i const *)a_scalars);
        b_e3m2_u8x32 = _mm256_loadu_si256((__m256i const *)b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }

    // Extract 5-bit magnitude, split into low 4 bits and bit 4
    __m256i a_magnitude_u8x32 = _mm256_and_si256(a_e3m2_u8x32, magnitude_mask_u8x32);
    __m256i b_magnitude_u8x32 = _mm256_and_si256(b_e3m2_u8x32, magnitude_mask_u8x32);
    __m256i a_shuffle_index_u8x32 = _mm256_and_si256(a_magnitude_u8x32, nibble_mask_u8x32);
    __m256i b_shuffle_index_u8x32 = _mm256_and_si256(b_magnitude_u8x32, nibble_mask_u8x32);
    __m256i a_upper_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(a_magnitude_u8x32, half_select_u8x32),
                                                     half_select_u8x32);
    __m256i b_upper_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(b_magnitude_u8x32, half_select_u8x32),
                                                     half_select_u8x32);

    // Dual VPSHUFB: lookup low bytes in both halves, blend based on bit 4
    __m256i a_lo_bytes_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lo_lower_u8x32, a_shuffle_index_u8x32),
                                                  _mm256_shuffle_epi8(lut_lo_upper_u8x32, a_shuffle_index_u8x32),
                                                  a_upper_select_u8x32);
    __m256i b_lo_bytes_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lo_lower_u8x32, b_shuffle_index_u8x32),
                                                  _mm256_shuffle_epi8(lut_lo_upper_u8x32, b_shuffle_index_u8x32),
                                                  b_upper_select_u8x32);

    // High byte: 1 iff magnitude >= 28 (signed compare safe: 27 < 128)
    __m256i a_hi_bytes_u8x32 = _mm256_and_si256(_mm256_cmpgt_epi8(a_magnitude_u8x32, high_threshold_u8x32), ones_u8x32);
    __m256i b_hi_bytes_u8x32 = _mm256_and_si256(_mm256_cmpgt_epi8(b_magnitude_u8x32, high_threshold_u8x32), ones_u8x32);

    // Interleave low and high bytes into i16 (little-endian: low byte first)
    __m256i a_lo_i16x16 = _mm256_unpacklo_epi8(a_lo_bytes_u8x32, a_hi_bytes_u8x32);
    __m256i a_hi_i16x16 = _mm256_unpackhi_epi8(a_lo_bytes_u8x32, a_hi_bytes_u8x32);
    __m256i b_lo_i16x16 = _mm256_unpacklo_epi8(b_lo_bytes_u8x32, b_hi_bytes_u8x32);
    __m256i b_hi_i16x16 = _mm256_unpackhi_epi8(b_lo_bytes_u8x32, b_hi_bytes_u8x32);

    // Combined sign: (a ^ b) & 0x20, widen to i16 via unpack, create +1/-1 sign vector
    __m256i sign_combined_u8x32 = _mm256_and_si256(_mm256_xor_si256(a_e3m2_u8x32, b_e3m2_u8x32), sign_mask_u8x32);
    __m256i negate_mask_u8x32 = _mm256_cmpeq_epi8(sign_combined_u8x32, sign_mask_u8x32);
    __m256i negate_lo_i16x16 = _mm256_unpacklo_epi8(negate_mask_u8x32, negate_mask_u8x32);
    __m256i negate_hi_i16x16 = _mm256_unpackhi_epi8(negate_mask_u8x32, negate_mask_u8x32);
    __m256i sign_lo_i16x16 = _mm256_or_si256(negate_lo_i16x16, ones_i16x16);
    __m256i sign_hi_i16x16 = _mm256_or_si256(negate_hi_i16x16, ones_i16x16);
    __m256i b_signed_lo_i16x16 = _mm256_sign_epi16(b_lo_i16x16, sign_lo_i16x16);
    __m256i b_signed_hi_i16x16 = _mm256_sign_epi16(b_hi_i16x16, sign_hi_i16x16);

    // VPDPWSSD: i16×i16→i32 fused dot-product-accumulate (replaces VPMADDWD + VPADDD)
    dot_i32x8 = _mm256_dpwssd_avx_epi32(dot_i32x8, a_lo_i16x16, b_signed_lo_i16x16);
    dot_i32x8 = _mm256_dpwssd_avx_epi32(dot_i32x8, a_hi_i16x16, b_signed_hi_i16x16);
    a_norm_i32x8 = _mm256_dpwssd_avx_epi32(a_norm_i32x8, a_lo_i16x16, a_lo_i16x16);
    a_norm_i32x8 = _mm256_dpwssd_avx_epi32(a_norm_i32x8, a_hi_i16x16, a_hi_i16x16);
    b_norm_i32x8 = _mm256_dpwssd_avx_epi32(b_norm_i32x8, b_lo_i16x16, b_lo_i16x16);
    b_norm_i32x8 = _mm256_dpwssd_avx_epi32(b_norm_i32x8, b_hi_i16x16, b_hi_i16x16);

    if (count_scalars) goto nk_angular_e3m2_alder_cycle;

    nk_i32_t dot_i32 = nk_reduce_add_i32x8_haswell_(dot_i32x8);
    nk_i32_t a_norm_i32 = nk_reduce_add_i32x8_haswell_(a_norm_i32x8);
    nk_i32_t b_norm_i32 = nk_reduce_add_i32x8_haswell_(b_norm_i32x8);
    // The 256.0f factor cancels in the angular normalization ratio
    *result = nk_angular_normalize_f32_haswell_(dot_i32, a_norm_i32, b_norm_i32);
}

NK_PUBLIC void nk_sqeuclidean_e3m2_alder(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars,
                                         nk_size_t count_scalars, nk_f32_t *result) {
    // Squared Euclidean distance for e3m2 using norm decomposition + VPDPWSSD:
    // ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b)
    // Each value × 16 is exact integer, so result = integer_result / 256.0f
    //
    __m256i const lut_lo_lower_u8x32 = _mm256_set_epi8(        //
        28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0, //
        28, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m256i const lut_lo_upper_u8x32 = _mm256_set_epi8(                                                           //
        (char)192, (char)128, 64, 0, (char)224, (char)192, (char)160, (char)128, 112, 96, 80, 64, 56, 48, 40, 32, //
        (char)192, (char)128, 64, 0, (char)224, (char)192, (char)160, (char)128, 112, 96, 80, 64, 56, 48, 40, 32);
    __m256i const nibble_mask_u8x32 = _mm256_set1_epi8(0x0F);
    __m256i const magnitude_mask_u8x32 = _mm256_set1_epi8(0x1F);
    __m256i const half_select_u8x32 = _mm256_set1_epi8(0x10);
    __m256i const sign_mask_u8x32 = _mm256_set1_epi8(0x20);
    __m256i const high_threshold_u8x32 = _mm256_set1_epi8(27);
    __m256i const ones_u8x32 = _mm256_set1_epi8(1);
    __m256i const ones_i16x16 = _mm256_set1_epi16(1);
    __m256i dot_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_i32x8 = _mm256_setzero_si256();
    __m256i a_e3m2_u8x32, b_e3m2_u8x32;

nk_sqeuclidean_e3m2_alder_cycle:
    if (count_scalars < 32) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b8x32_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x32_serial_(b_scalars, &b_vec, count_scalars);
        a_e3m2_u8x32 = a_vec.ymm;
        b_e3m2_u8x32 = b_vec.ymm;
        count_scalars = 0;
    }
    else {
        a_e3m2_u8x32 = _mm256_loadu_si256((__m256i const *)a_scalars);
        b_e3m2_u8x32 = _mm256_loadu_si256((__m256i const *)b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }

    // Extract 5-bit magnitude, split into low 4 bits and bit 4
    __m256i a_magnitude_u8x32 = _mm256_and_si256(a_e3m2_u8x32, magnitude_mask_u8x32);
    __m256i b_magnitude_u8x32 = _mm256_and_si256(b_e3m2_u8x32, magnitude_mask_u8x32);
    __m256i a_shuffle_index_u8x32 = _mm256_and_si256(a_magnitude_u8x32, nibble_mask_u8x32);
    __m256i b_shuffle_index_u8x32 = _mm256_and_si256(b_magnitude_u8x32, nibble_mask_u8x32);
    __m256i a_upper_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(a_magnitude_u8x32, half_select_u8x32),
                                                     half_select_u8x32);
    __m256i b_upper_select_u8x32 = _mm256_cmpeq_epi8(_mm256_and_si256(b_magnitude_u8x32, half_select_u8x32),
                                                     half_select_u8x32);

    // Dual VPSHUFB: lookup low bytes in both halves, blend based on bit 4
    __m256i a_lo_bytes_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lo_lower_u8x32, a_shuffle_index_u8x32),
                                                  _mm256_shuffle_epi8(lut_lo_upper_u8x32, a_shuffle_index_u8x32),
                                                  a_upper_select_u8x32);
    __m256i b_lo_bytes_u8x32 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut_lo_lower_u8x32, b_shuffle_index_u8x32),
                                                  _mm256_shuffle_epi8(lut_lo_upper_u8x32, b_shuffle_index_u8x32),
                                                  b_upper_select_u8x32);

    // High byte: 1 iff magnitude >= 28
    __m256i a_hi_bytes_u8x32 = _mm256_and_si256(_mm256_cmpgt_epi8(a_magnitude_u8x32, high_threshold_u8x32), ones_u8x32);
    __m256i b_hi_bytes_u8x32 = _mm256_and_si256(_mm256_cmpgt_epi8(b_magnitude_u8x32, high_threshold_u8x32), ones_u8x32);

    // Interleave low and high bytes into i16
    __m256i a_lo_i16x16 = _mm256_unpacklo_epi8(a_lo_bytes_u8x32, a_hi_bytes_u8x32);
    __m256i a_hi_i16x16 = _mm256_unpackhi_epi8(a_lo_bytes_u8x32, a_hi_bytes_u8x32);
    __m256i b_lo_i16x16 = _mm256_unpacklo_epi8(b_lo_bytes_u8x32, b_hi_bytes_u8x32);
    __m256i b_hi_i16x16 = _mm256_unpackhi_epi8(b_lo_bytes_u8x32, b_hi_bytes_u8x32);

    // Combined sign for dot product
    __m256i sign_combined_u8x32 = _mm256_and_si256(_mm256_xor_si256(a_e3m2_u8x32, b_e3m2_u8x32), sign_mask_u8x32);
    __m256i negate_mask_u8x32 = _mm256_cmpeq_epi8(sign_combined_u8x32, sign_mask_u8x32);
    __m256i negate_lo_i16x16 = _mm256_unpacklo_epi8(negate_mask_u8x32, negate_mask_u8x32);
    __m256i negate_hi_i16x16 = _mm256_unpackhi_epi8(negate_mask_u8x32, negate_mask_u8x32);
    __m256i sign_lo_i16x16 = _mm256_or_si256(negate_lo_i16x16, ones_i16x16);
    __m256i sign_hi_i16x16 = _mm256_or_si256(negate_hi_i16x16, ones_i16x16);
    __m256i b_signed_lo_i16x16 = _mm256_sign_epi16(b_lo_i16x16, sign_lo_i16x16);
    __m256i b_signed_hi_i16x16 = _mm256_sign_epi16(b_hi_i16x16, sign_hi_i16x16);

    // VPDPWSSD: i16×i16→i32 fused dot-product-accumulate
    dot_i32x8 = _mm256_dpwssd_avx_epi32(dot_i32x8, a_lo_i16x16, b_signed_lo_i16x16);
    dot_i32x8 = _mm256_dpwssd_avx_epi32(dot_i32x8, a_hi_i16x16, b_signed_hi_i16x16);
    a_norm_i32x8 = _mm256_dpwssd_avx_epi32(a_norm_i32x8, a_lo_i16x16, a_lo_i16x16);
    a_norm_i32x8 = _mm256_dpwssd_avx_epi32(a_norm_i32x8, a_hi_i16x16, a_hi_i16x16);
    b_norm_i32x8 = _mm256_dpwssd_avx_epi32(b_norm_i32x8, b_lo_i16x16, b_lo_i16x16);
    b_norm_i32x8 = _mm256_dpwssd_avx_epi32(b_norm_i32x8, b_hi_i16x16, b_hi_i16x16);

    if (count_scalars) goto nk_sqeuclidean_e3m2_alder_cycle;

    nk_i32_t dot_i32 = nk_reduce_add_i32x8_haswell_(dot_i32x8);
    nk_i32_t a_norm_i32 = nk_reduce_add_i32x8_haswell_(a_norm_i32x8);
    nk_i32_t b_norm_i32 = nk_reduce_add_i32x8_haswell_(b_norm_i32x8);
    // ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b), scaled by 256
    *result = (nk_f32_t)(a_norm_i32 + b_norm_i32 - 2 * dot_i32) / 256.0f;
}

NK_PUBLIC void nk_euclidean_e3m2_alder(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e3m2_alder(a, b, n, result);
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

#endif // NK_TARGET_ALDER
#endif // NK_TARGET_X86_
#endif // NK_SPATIAL_ALDER_H
