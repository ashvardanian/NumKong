/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for NEON FP8DOT4.
 *  @file include/numkong/spatial/neonfp8.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/spatial.h
 *
 *  For L2 distance, we use the identity: (a−b)² = a² + b² − 2 × a × b,
 *  computing all three terms via FP8DOT4 without FP8 subtraction.
 *  Angular distance uses three DOT4 accumulators (a·b, ‖a‖², ‖b‖²) in parallel.
 */
#ifndef NK_SPATIAL_NEONFP8_H
#define NK_SPATIAL_NEONFP8_H

#if NK_TARGET_ARM64_
#if NK_TARGET_NEONFP8

#include "numkong/types.h"
#include "numkong/dot/neonfp8.h"  // `nk_e2m3x16_to_e4m3x16_neonfp8_`, `nk_e3m2x16_to_e5m2x16_neonfp8_`
#include "numkong/spatial/neon.h" // `nk_f32_sqrt_neon`, `nk_angular_normalize_f32_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd+fp8dot4"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd+fp8dot4")
#endif

NK_API_COMPTIME void nk_sqeuclidean_e4m3_neonfp8(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                 nk_f32_t *result) {
    mfloat8x16_t a_mf8x16, b_mf8x16;
    float32x4_t a2_f32x4 = vdupq_n_f32(0), ab_f32x4 = vdupq_n_f32(0), b2_f32x4 = vdupq_n_f32(0);
nk_sqeuclidean_e4m3_neonfp8_cycle:
    if (n < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n);
        nk_partial_load_b8x16_serial_(b, &b_vec, n);
        a_mf8x16 = vreinterpretq_mf8_u8(a_vec.u8x16);
        b_mf8x16 = vreinterpretq_mf8_u8(b_vec.u8x16);
        n = 0;
    }
    else {
        a_mf8x16 = vreinterpretq_mf8_u8(vld1q_u8((nk_u8_t const *)a));
        b_mf8x16 = vreinterpretq_mf8_u8(vld1q_u8((nk_u8_t const *)b));
        a += 16, b += 16, n -= 16;
    }
    a2_f32x4 = vdotq_f32_mf8_fpm(a2_f32x4, a_mf8x16, a_mf8x16, NK_FPM_E4M3_);
    ab_f32x4 = vdotq_f32_mf8_fpm(ab_f32x4, a_mf8x16, b_mf8x16, NK_FPM_E4M3_);
    b2_f32x4 = vdotq_f32_mf8_fpm(b2_f32x4, b_mf8x16, b_mf8x16, NK_FPM_E4M3_);
    if (n) goto nk_sqeuclidean_e4m3_neonfp8_cycle;
    *result = vaddvq_f32(a2_f32x4) - 2 * vaddvq_f32(ab_f32x4) + vaddvq_f32(b2_f32x4);
}

NK_API_COMPTIME void nk_euclidean_e4m3_neonfp8(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e4m3_neonfp8(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_API_COMPTIME void nk_angular_e4m3_neonfp8(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    mfloat8x16_t a_mf8x16, b_mf8x16;
    float32x4_t ab_f32x4 = vdupq_n_f32(0), a2_f32x4 = vdupq_n_f32(0), b2_f32x4 = vdupq_n_f32(0);
nk_angular_e4m3_neonfp8_cycle:
    if (n < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n);
        nk_partial_load_b8x16_serial_(b, &b_vec, n);
        a_mf8x16 = vreinterpretq_mf8_u8(a_vec.u8x16);
        b_mf8x16 = vreinterpretq_mf8_u8(b_vec.u8x16);
        n = 0;
    }
    else {
        a_mf8x16 = vreinterpretq_mf8_u8(vld1q_u8((nk_u8_t const *)a));
        b_mf8x16 = vreinterpretq_mf8_u8(vld1q_u8((nk_u8_t const *)b));
        a += 16, b += 16, n -= 16;
    }
    ab_f32x4 = vdotq_f32_mf8_fpm(ab_f32x4, a_mf8x16, b_mf8x16, NK_FPM_E4M3_);
    a2_f32x4 = vdotq_f32_mf8_fpm(a2_f32x4, a_mf8x16, a_mf8x16, NK_FPM_E4M3_);
    b2_f32x4 = vdotq_f32_mf8_fpm(b2_f32x4, b_mf8x16, b_mf8x16, NK_FPM_E4M3_);
    if (n) goto nk_angular_e4m3_neonfp8_cycle;
    *result = nk_angular_normalize_f32_neon_(vaddvq_f32(ab_f32x4), vaddvq_f32(a2_f32x4), vaddvq_f32(b2_f32x4));
}

NK_API_COMPTIME void nk_sqeuclidean_e5m2_neonfp8(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                 nk_f32_t *result) {
    mfloat8x16_t a_mf8x16, b_mf8x16;
    float32x4_t a2_f32x4 = vdupq_n_f32(0), ab_f32x4 = vdupq_n_f32(0), b2_f32x4 = vdupq_n_f32(0);
nk_sqeuclidean_e5m2_neonfp8_cycle:
    if (n < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n);
        nk_partial_load_b8x16_serial_(b, &b_vec, n);
        a_mf8x16 = vreinterpretq_mf8_u8(a_vec.u8x16);
        b_mf8x16 = vreinterpretq_mf8_u8(b_vec.u8x16);
        n = 0;
    }
    else {
        a_mf8x16 = vreinterpretq_mf8_u8(vld1q_u8((nk_u8_t const *)a));
        b_mf8x16 = vreinterpretq_mf8_u8(vld1q_u8((nk_u8_t const *)b));
        a += 16, b += 16, n -= 16;
    }
    a2_f32x4 = vdotq_f32_mf8_fpm(a2_f32x4, a_mf8x16, a_mf8x16, NK_FPM_E5M2_);
    ab_f32x4 = vdotq_f32_mf8_fpm(ab_f32x4, a_mf8x16, b_mf8x16, NK_FPM_E5M2_);
    b2_f32x4 = vdotq_f32_mf8_fpm(b2_f32x4, b_mf8x16, b_mf8x16, NK_FPM_E5M2_);
    if (n) goto nk_sqeuclidean_e5m2_neonfp8_cycle;
    *result = vaddvq_f32(a2_f32x4) - 2 * vaddvq_f32(ab_f32x4) + vaddvq_f32(b2_f32x4);
}

NK_API_COMPTIME void nk_euclidean_e5m2_neonfp8(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e5m2_neonfp8(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_API_COMPTIME void nk_angular_e5m2_neonfp8(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    mfloat8x16_t a_mf8x16, b_mf8x16;
    float32x4_t ab_f32x4 = vdupq_n_f32(0), a2_f32x4 = vdupq_n_f32(0), b2_f32x4 = vdupq_n_f32(0);
nk_angular_e5m2_neonfp8_cycle:
    if (n < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n);
        nk_partial_load_b8x16_serial_(b, &b_vec, n);
        a_mf8x16 = vreinterpretq_mf8_u8(a_vec.u8x16);
        b_mf8x16 = vreinterpretq_mf8_u8(b_vec.u8x16);
        n = 0;
    }
    else {
        a_mf8x16 = vreinterpretq_mf8_u8(vld1q_u8((nk_u8_t const *)a));
        b_mf8x16 = vreinterpretq_mf8_u8(vld1q_u8((nk_u8_t const *)b));
        a += 16, b += 16, n -= 16;
    }
    ab_f32x4 = vdotq_f32_mf8_fpm(ab_f32x4, a_mf8x16, b_mf8x16, NK_FPM_E5M2_);
    a2_f32x4 = vdotq_f32_mf8_fpm(a2_f32x4, a_mf8x16, a_mf8x16, NK_FPM_E5M2_);
    b2_f32x4 = vdotq_f32_mf8_fpm(b2_f32x4, b_mf8x16, b_mf8x16, NK_FPM_E5M2_);
    if (n) goto nk_angular_e5m2_neonfp8_cycle;
    *result = nk_angular_normalize_f32_neon_(vaddvq_f32(ab_f32x4), vaddvq_f32(a2_f32x4), vaddvq_f32(b2_f32x4));
}

NK_API_COMPTIME void nk_sqeuclidean_e2m3_neonfp8(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                 nk_f32_t *result) {
    mfloat8x16_t a_mf8x16, b_mf8x16;
    float32x4_t a2_f32x4 = vdupq_n_f32(0), ab_f32x4 = vdupq_n_f32(0), b2_f32x4 = vdupq_n_f32(0);
nk_sqeuclidean_e2m3_neonfp8_cycle:
    if (n < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n);
        nk_partial_load_b8x16_serial_(b, &b_vec, n);
        a_mf8x16 = vreinterpretq_mf8_u8(nk_e2m3x16_to_e4m3x16_neonfp8_(a_vec.u8x16));
        b_mf8x16 = vreinterpretq_mf8_u8(nk_e2m3x16_to_e4m3x16_neonfp8_(b_vec.u8x16));
        n = 0;
    }
    else {
        a_mf8x16 = vreinterpretq_mf8_u8(nk_e2m3x16_to_e4m3x16_neonfp8_(vld1q_u8((nk_u8_t const *)a)));
        b_mf8x16 = vreinterpretq_mf8_u8(nk_e2m3x16_to_e4m3x16_neonfp8_(vld1q_u8((nk_u8_t const *)b)));
        a += 16, b += 16, n -= 16;
    }
    a2_f32x4 = vdotq_f32_mf8_fpm(a2_f32x4, a_mf8x16, a_mf8x16, NK_FPM_E4M3_);
    ab_f32x4 = vdotq_f32_mf8_fpm(ab_f32x4, a_mf8x16, b_mf8x16, NK_FPM_E4M3_);
    b2_f32x4 = vdotq_f32_mf8_fpm(b2_f32x4, b_mf8x16, b_mf8x16, NK_FPM_E4M3_);
    if (n) goto nk_sqeuclidean_e2m3_neonfp8_cycle;
    *result = vaddvq_f32(a2_f32x4) - 2 * vaddvq_f32(ab_f32x4) + vaddvq_f32(b2_f32x4);
}

NK_API_COMPTIME void nk_euclidean_e2m3_neonfp8(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e2m3_neonfp8(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_API_COMPTIME void nk_angular_e2m3_neonfp8(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    mfloat8x16_t a_mf8x16, b_mf8x16;
    float32x4_t ab_f32x4 = vdupq_n_f32(0), a2_f32x4 = vdupq_n_f32(0), b2_f32x4 = vdupq_n_f32(0);
nk_angular_e2m3_neonfp8_cycle:
    if (n < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n);
        nk_partial_load_b8x16_serial_(b, &b_vec, n);
        a_mf8x16 = vreinterpretq_mf8_u8(nk_e2m3x16_to_e4m3x16_neonfp8_(a_vec.u8x16));
        b_mf8x16 = vreinterpretq_mf8_u8(nk_e2m3x16_to_e4m3x16_neonfp8_(b_vec.u8x16));
        n = 0;
    }
    else {
        a_mf8x16 = vreinterpretq_mf8_u8(nk_e2m3x16_to_e4m3x16_neonfp8_(vld1q_u8((nk_u8_t const *)a)));
        b_mf8x16 = vreinterpretq_mf8_u8(nk_e2m3x16_to_e4m3x16_neonfp8_(vld1q_u8((nk_u8_t const *)b)));
        a += 16, b += 16, n -= 16;
    }
    ab_f32x4 = vdotq_f32_mf8_fpm(ab_f32x4, a_mf8x16, b_mf8x16, NK_FPM_E4M3_);
    a2_f32x4 = vdotq_f32_mf8_fpm(a2_f32x4, a_mf8x16, a_mf8x16, NK_FPM_E4M3_);
    b2_f32x4 = vdotq_f32_mf8_fpm(b2_f32x4, b_mf8x16, b_mf8x16, NK_FPM_E4M3_);
    if (n) goto nk_angular_e2m3_neonfp8_cycle;
    *result = nk_angular_normalize_f32_neon_(vaddvq_f32(ab_f32x4), vaddvq_f32(a2_f32x4), vaddvq_f32(b2_f32x4));
}

NK_API_COMPTIME void nk_sqeuclidean_e3m2_neonfp8(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                 nk_f32_t *result) {
    mfloat8x16_t a_mf8x16, b_mf8x16;
    float32x4_t a2_f32x4 = vdupq_n_f32(0), ab_f32x4 = vdupq_n_f32(0), b2_f32x4 = vdupq_n_f32(0);
nk_sqeuclidean_e3m2_neonfp8_cycle:
    if (n < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n);
        nk_partial_load_b8x16_serial_(b, &b_vec, n);
        a_mf8x16 = vreinterpretq_mf8_u8(nk_e3m2x16_to_e5m2x16_neonfp8_(a_vec.u8x16));
        b_mf8x16 = vreinterpretq_mf8_u8(nk_e3m2x16_to_e5m2x16_neonfp8_(b_vec.u8x16));
        n = 0;
    }
    else {
        a_mf8x16 = vreinterpretq_mf8_u8(nk_e3m2x16_to_e5m2x16_neonfp8_(vld1q_u8((nk_u8_t const *)a)));
        b_mf8x16 = vreinterpretq_mf8_u8(nk_e3m2x16_to_e5m2x16_neonfp8_(vld1q_u8((nk_u8_t const *)b)));
        a += 16, b += 16, n -= 16;
    }
    a2_f32x4 = vdotq_f32_mf8_fpm(a2_f32x4, a_mf8x16, a_mf8x16, NK_FPM_E5M2_);
    ab_f32x4 = vdotq_f32_mf8_fpm(ab_f32x4, a_mf8x16, b_mf8x16, NK_FPM_E5M2_);
    b2_f32x4 = vdotq_f32_mf8_fpm(b2_f32x4, b_mf8x16, b_mf8x16, NK_FPM_E5M2_);
    if (n) goto nk_sqeuclidean_e3m2_neonfp8_cycle;
    *result = vaddvq_f32(a2_f32x4) - 2 * vaddvq_f32(ab_f32x4) + vaddvq_f32(b2_f32x4);
}

NK_API_COMPTIME void nk_euclidean_e3m2_neonfp8(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e3m2_neonfp8(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_API_COMPTIME void nk_angular_e3m2_neonfp8(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    mfloat8x16_t a_mf8x16, b_mf8x16;
    float32x4_t ab_f32x4 = vdupq_n_f32(0), a2_f32x4 = vdupq_n_f32(0), b2_f32x4 = vdupq_n_f32(0);
nk_angular_e3m2_neonfp8_cycle:
    if (n < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a, &a_vec, n);
        nk_partial_load_b8x16_serial_(b, &b_vec, n);
        a_mf8x16 = vreinterpretq_mf8_u8(nk_e3m2x16_to_e5m2x16_neonfp8_(a_vec.u8x16));
        b_mf8x16 = vreinterpretq_mf8_u8(nk_e3m2x16_to_e5m2x16_neonfp8_(b_vec.u8x16));
        n = 0;
    }
    else {
        a_mf8x16 = vreinterpretq_mf8_u8(nk_e3m2x16_to_e5m2x16_neonfp8_(vld1q_u8((nk_u8_t const *)a)));
        b_mf8x16 = vreinterpretq_mf8_u8(nk_e3m2x16_to_e5m2x16_neonfp8_(vld1q_u8((nk_u8_t const *)b)));
        a += 16, b += 16, n -= 16;
    }
    ab_f32x4 = vdotq_f32_mf8_fpm(ab_f32x4, a_mf8x16, b_mf8x16, NK_FPM_E5M2_);
    a2_f32x4 = vdotq_f32_mf8_fpm(a2_f32x4, a_mf8x16, a_mf8x16, NK_FPM_E5M2_);
    b2_f32x4 = vdotq_f32_mf8_fpm(b2_f32x4, b_mf8x16, b_mf8x16, NK_FPM_E5M2_);
    if (n) goto nk_angular_e3m2_neonfp8_cycle;
    *result = nk_angular_normalize_f32_neon_(vaddvq_f32(ab_f32x4), vaddvq_f32(a2_f32x4), vaddvq_f32(b2_f32x4));
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONFP8
#endif // NK_TARGET_ARM64_
#endif // NK_SPATIAL_NEONFP8_H
