/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for NEON.
 *  @file include/numkong/spatial/neon.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_neon_instructions Key NEON Spatial Instructions
 *
 *  ARM NEON instructions for distance computations:
 *
 *      Intrinsic         Instruction                   Latency     Throughput
 *                                                                  A76     M4+/V1+/Oryon
 *      vfmaq_f32         FMLA (V.4S, V.4S, V.4S)       4cy         2/cy    4/cy
 *      vmulq_f32         FMUL (V.4S, V.4S, V.4S)       3cy         2/cy    4/cy
 *      vaddq_f32         FADD (V.4S, V.4S, V.4S)       2cy         2/cy    4/cy
 *      vsubq_f32         FSUB (V.4S, V.4S, V.4S)       2cy         2/cy    4/cy
 *      vrsqrteq_f32      FRSQRTE (V.4S, V.4S)          2cy         2/cy    2/cy
 *      vsqrtq_f32        FSQRT (V.4S, V.4S)            9-12cy      0.25/cy 0.25/cy
 *      vrecpeq_f32       FRECPE (V.4S, V.4S)           2cy         2/cy    2/cy
 *
 *  FRSQRTE provides ~8-bit precision; two Newton-Raphson iterations via vrsqrtsq_f32 achieve
 *  ~23-bit precision, sufficient for f32. This is much faster than FSQRT (0.25/cy).
 *
 *  Distance computations (L2, angular) benefit from 2x throughput on 4-pipe cores (Apple M4+,
 *  Graviton3+, Oryon), but FSQRT remains slow on all cores. Use rsqrt+NR when precision allows.
 */
#ifndef NK_SPATIAL_NEON_H
#define NK_SPATIAL_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/types.h"
#include "numkong/scalar/neon.h" // `nk_f32_sqrt_neon`
#include "numkong/dot/neon.h"    // `nk_dot_stable_sum_f64x2_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

/**
 *  @brief Reciprocal square root of 4 floats with Newton-Raphson refinement.
 *
 *  Uses `vrsqrteq_f32` (~8-bit initial estimate) followed by two Newton-Raphson iterations
 *  via `vrsqrtsq_f32`, achieving ~23-bit precision — sufficient for f32.
 *  Much faster than `vsqrtq_f32` (2 cy vs 9-12 cy latency, 2/cy vs 0.25/cy throughput).
 */
NK_INTERNAL float32x4_t nk_rsqrt_f32x4_neon_(float32x4_t x) {
    float32x4_t rsqrt = vrsqrteq_f32(x);
    rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(x, rsqrt), rsqrt));
    rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(x, rsqrt), rsqrt));
    return rsqrt;
}

/**
 *  @brief Reciprocal square root of 2 doubles with Newton-Raphson refinement.
 *
 *  Uses `vrsqrteq_f64` (~8-bit initial estimate) followed by three Newton-Raphson iterations
 *  via `vrsqrtsq_f64`, achieving ~48-bit precision — reasonable for f64 distance computations
 *  where the final result is often narrowed to f32.  For full 52-bit mantissa fidelity,
 *  prefer `vsqrtq_f64` instead.
 */
NK_INTERNAL float64x2_t nk_rsqrt_f64x2_neon_(float64x2_t x) {
    float64x2_t rsqrt = vrsqrteq_f64(x);
    rsqrt = vmulq_f64(rsqrt, vrsqrtsq_f64(vmulq_f64(x, rsqrt), rsqrt));
    rsqrt = vmulq_f64(rsqrt, vrsqrtsq_f64(vmulq_f64(x, rsqrt), rsqrt));
    rsqrt = vmulq_f64(rsqrt, vrsqrtsq_f64(vmulq_f64(x, rsqrt), rsqrt));
    return rsqrt;
}

NK_INTERNAL nk_f32_t nk_angular_normalize_f32_neon_(nk_f32_t ab, nk_f32_t a2, nk_f32_t b2) {
    if (a2 == 0 && b2 == 0) return 0;
    if (ab == 0) return 1;
    nk_f32_t squares_arr[2] = {a2, b2};
    float32x2_t squares = vld1_f32(squares_arr);
    // Unlike x86, Arm NEON manuals don't explicitly mention the accuracy of their `rsqrt` approximation.
    // Third-party research suggests that it's less accurate than SSE instructions, having an error of 1.5×2⁻¹².
    // One or two rounds of Newton-Raphson refinement are recommended to improve the accuracy.
    // https://github.com/lighttransport/embree-aarch64/issues/24
    // https://github.com/lighttransport/embree-aarch64/blob/3f75f8cb4e553d13dced941b5fefd4c826835a6b/common/math/math.h#L137-L145
    float32x2_t rsqrts = vrsqrte_f32(squares);
    // Perform two rounds of Newton-Raphson refinement:
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts = vmul_f32(rsqrts, vrsqrts_f32(vmul_f32(squares, rsqrts), rsqrts));
    rsqrts = vmul_f32(rsqrts, vrsqrts_f32(vmul_f32(squares, rsqrts), rsqrts));
    vst1_f32(squares_arr, rsqrts);
    nk_f32_t result = 1 - ab * squares_arr[0] * squares_arr[1];
    return result > 0 ? result : 0;
}

NK_INTERNAL nk_f64_t nk_angular_normalize_f64_neon_(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {
    if (a2 == 0 && b2 == 0) return 0;
    if (ab == 0) return 1;
    nk_f64_t squares_arr[2] = {a2, b2};
    float64x2_t squares = vld1q_f64(squares_arr);

    // Unlike x86, Arm NEON manuals don't explicitly mention the accuracy of their `rsqrt` approximation.
    // Third-party research suggests that it's less accurate than SSE instructions, having an error of 1.5×2⁻¹².
    // One or two rounds of Newton-Raphson refinement are recommended to improve the accuracy.
    // https://github.com/lighttransport/embree-aarch64/issues/24
    // https://github.com/lighttransport/embree-aarch64/blob/3f75f8cb4e553d13dced941b5fefd4c826835a6b/common/math/math.h#L137-L145
    float64x2_t rsqrts_f64x2 = vrsqrteq_f64(squares);
    // Perform three rounds of Newton-Raphson refinement for f64 precision (~48 bits):
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts_f64x2 = vmulq_f64(rsqrts_f64x2, vrsqrtsq_f64(vmulq_f64(squares, rsqrts_f64x2), rsqrts_f64x2));
    rsqrts_f64x2 = vmulq_f64(rsqrts_f64x2, vrsqrtsq_f64(vmulq_f64(squares, rsqrts_f64x2), rsqrts_f64x2));
    rsqrts_f64x2 = vmulq_f64(rsqrts_f64x2, vrsqrtsq_f64(vmulq_f64(squares, rsqrts_f64x2), rsqrts_f64x2));
    vst1q_f64(squares_arr, rsqrts_f64x2);
    nk_f64_t result = 1 - ab * squares_arr[0] * squares_arr[1];
    return result > 0 ? result : 0;
}

#pragma region - Traditional Floats

NK_PUBLIC void nk_sqeuclidean_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    // Accumulate in f64 for numerical stability (2 f32s per iteration, avoids slow vget_low/high)
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float32x2_t a_f32x2 = vld1_f32(a + i);
        float32x2_t b_f32x2 = vld1_f32(b + i);
        float32x2_t diff_f32x2 = vsub_f32(a_f32x2, b_f32x2);
        float64x2_t diff_f64x2 = vcvt_f64_f32(diff_f32x2);
        sum_f64x2 = vfmaq_f64(sum_f64x2, diff_f64x2, diff_f64x2);
    }
    nk_f64_t sum_f64 = vaddvq_f64(sum_f64x2);
    for (; i < n; ++i) {
        nk_f64_t diff_f64 = (nk_f64_t)a[i] - (nk_f64_t)b[i];
        sum_f64 += diff_f64 * diff_f64;
    }
    *result = sum_f64;
}

NK_PUBLIC void nk_euclidean_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_sqeuclidean_f32_neon(a, b, n, result);
    *result = nk_f64_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    // Accumulate in f64 for numerical stability (2 f32s per iteration, avoids slow vget_low/high)
    float64x2_t ab_f64x2 = vdupq_n_f64(0);
    float64x2_t a2_f64x2 = vdupq_n_f64(0);
    float64x2_t b2_f64x2 = vdupq_n_f64(0);
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float32x2_t a_f32x2 = vld1_f32(a + i);
        float32x2_t b_f32x2 = vld1_f32(b + i);
        float64x2_t a_f64x2 = vcvt_f64_f32(a_f32x2);
        float64x2_t b_f64x2 = vcvt_f64_f32(b_f32x2);
        ab_f64x2 = vfmaq_f64(ab_f64x2, a_f64x2, b_f64x2);
        a2_f64x2 = vfmaq_f64(a2_f64x2, a_f64x2, a_f64x2);
        b2_f64x2 = vfmaq_f64(b2_f64x2, b_f64x2, b_f64x2);
    }
    nk_f64_t ab_f64 = vaddvq_f64(ab_f64x2);
    nk_f64_t a2_f64 = vaddvq_f64(a2_f64x2);
    nk_f64_t b2_f64 = vaddvq_f64(b2_f64x2);
    for (; i < n; ++i) {
        nk_f64_t ai = (nk_f64_t)a[i], bi = (nk_f64_t)b[i];
        ab_f64 += ai * bi, a2_f64 += ai * ai, b2_f64 += bi * bi;
    }
    *result = nk_angular_normalize_f64_neon_(ab_f64, a2_f64, b2_f64);
}

NK_PUBLIC void nk_sqeuclidean_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    float64x2_t a_f64x2, b_f64x2;

nk_sqeuclidean_f64_neon_cycle:
    if (n < 2) {
        nk_b128_vec_t a_tail, b_tail;
        nk_partial_load_b64x2_serial_(a, &a_tail, n);
        nk_partial_load_b64x2_serial_(b, &b_tail, n);
        a_f64x2 = a_tail.f64x2;
        b_f64x2 = b_tail.f64x2;
        n = 0;
    }
    else {
        a_f64x2 = vld1q_f64(a);
        b_f64x2 = vld1q_f64(b);
        a += 2, b += 2, n -= 2;
    }
    float64x2_t diff_f64x2 = vsubq_f64(a_f64x2, b_f64x2);
    sum_f64x2 = vfmaq_f64(sum_f64x2, diff_f64x2, diff_f64x2);
    if (n) goto nk_sqeuclidean_f64_neon_cycle;

    *result = vaddvq_f64(sum_f64x2);
}

NK_PUBLIC void nk_euclidean_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_sqeuclidean_f64_neon(a, b, n, result);
    *result = nk_f64_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    // Dot2 (Ogita-Rump-Oishi) for cross-product ab (may have cancellation),
    // simple FMA for self-products a2/b2 (all positive, no cancellation)
    float64x2_t ab_sum_f64x2 = vdupq_n_f64(0);
    float64x2_t ab_compensation_f64x2 = vdupq_n_f64(0);
    float64x2_t a2_f64x2 = vdupq_n_f64(0);
    float64x2_t b2_f64x2 = vdupq_n_f64(0);
    float64x2_t a_f64x2, b_f64x2;

nk_angular_f64_neon_cycle:
    if (n < 2) {
        nk_b128_vec_t a_tail, b_tail;
        nk_partial_load_b64x2_serial_(a, &a_tail, n);
        nk_partial_load_b64x2_serial_(b, &b_tail, n);
        a_f64x2 = a_tail.f64x2;
        b_f64x2 = b_tail.f64x2;
        n = 0;
    }
    else {
        a_f64x2 = vld1q_f64(a);
        b_f64x2 = vld1q_f64(b);
        a += 2, b += 2, n -= 2;
    }
    // TwoProd for ab: product = a*b, error = fma(a,b,-product)
    float64x2_t product_f64x2 = vmulq_f64(a_f64x2, b_f64x2);
    float64x2_t product_error_f64x2 = vnegq_f64(vfmsq_f64(product_f64x2, a_f64x2, b_f64x2));
    // TwoSum: (t, q) = TwoSum(sum, product)
    float64x2_t tentative_sum_f64x2 = vaddq_f64(ab_sum_f64x2, product_f64x2);
    float64x2_t virtual_addend_f64x2 = vsubq_f64(tentative_sum_f64x2, ab_sum_f64x2);
    float64x2_t sum_error_f64x2 = vaddq_f64(
        vsubq_f64(ab_sum_f64x2, vsubq_f64(tentative_sum_f64x2, virtual_addend_f64x2)),
        vsubq_f64(product_f64x2, virtual_addend_f64x2));
    ab_sum_f64x2 = tentative_sum_f64x2;
    ab_compensation_f64x2 = vaddq_f64(ab_compensation_f64x2, vaddq_f64(sum_error_f64x2, product_error_f64x2));
    // Simple FMA for self-products (no cancellation)
    a2_f64x2 = vfmaq_f64(a2_f64x2, a_f64x2, a_f64x2);
    b2_f64x2 = vfmaq_f64(b2_f64x2, b_f64x2, b_f64x2);
    if (n) goto nk_angular_f64_neon_cycle;

    *result = nk_angular_normalize_f64_neon_( //
        nk_dot_stable_sum_f64x2_neon_(ab_sum_f64x2, ab_compensation_f64x2), vaddvq_f64(a2_f64x2), vaddvq_f64(b2_f64x2));
}

#pragma endregion - Traditional Floats
#pragma region - Smaller Floats

NK_PUBLIC void nk_sqeuclidean_bf16_neon(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    uint16x8_t a_u16x8, b_u16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_sqeuclidean_bf16_neon_cycle:
    if (n < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b16x8_serial_(a, &a_vec, n);
        nk_partial_load_b16x8_serial_(b, &b_vec, n);
        a_u16x8 = a_vec.u16x8;
        b_u16x8 = b_vec.u16x8;
        n = 0;
    }
    else {
        a_u16x8 = vld1q_u16((nk_u16_t const *)a);
        b_u16x8 = vld1q_u16((nk_u16_t const *)b);
        a += 8, b += 8, n -= 8;
    }
    float32x4_t a_low_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(a_u16x8), 16));
    float32x4_t a_high_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(a_u16x8), 16));
    float32x4_t b_low_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b_u16x8), 16));
    float32x4_t b_high_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b_u16x8), 16));
    float32x4_t diff_low_f32x4 = vsubq_f32(a_low_f32x4, b_low_f32x4);
    float32x4_t diff_high_f32x4 = vsubq_f32(a_high_f32x4, b_high_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_low_f32x4, diff_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_high_f32x4, diff_high_f32x4);
    if (n) goto nk_sqeuclidean_bf16_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_bf16_neon(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_bf16_neon(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_bf16_neon(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    uint16x8_t a_u16x8, b_u16x8;
    float32x4_t ab_f32x4 = vdupq_n_f32(0);
    float32x4_t a2_f32x4 = vdupq_n_f32(0);
    float32x4_t b2_f32x4 = vdupq_n_f32(0);
nk_angular_bf16_neon_cycle:
    if (n < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b16x8_serial_(a, &a_vec, n);
        nk_partial_load_b16x8_serial_(b, &b_vec, n);
        a_u16x8 = a_vec.u16x8;
        b_u16x8 = b_vec.u16x8;
        n = 0;
    }
    else {
        a_u16x8 = vld1q_u16((nk_u16_t const *)a);
        b_u16x8 = vld1q_u16((nk_u16_t const *)b);
        a += 8, b += 8, n -= 8;
    }
    float32x4_t a_low_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(a_u16x8), 16));
    float32x4_t a_high_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(a_u16x8), 16));
    float32x4_t b_low_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b_u16x8), 16));
    float32x4_t b_high_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b_u16x8), 16));
    ab_f32x4 = vfmaq_f32(ab_f32x4, a_low_f32x4, b_low_f32x4);
    ab_f32x4 = vfmaq_f32(ab_f32x4, a_high_f32x4, b_high_f32x4);
    a2_f32x4 = vfmaq_f32(a2_f32x4, a_low_f32x4, a_low_f32x4);
    a2_f32x4 = vfmaq_f32(a2_f32x4, a_high_f32x4, a_high_f32x4);
    b2_f32x4 = vfmaq_f32(b2_f32x4, b_low_f32x4, b_low_f32x4);
    b2_f32x4 = vfmaq_f32(b2_f32x4, b_high_f32x4, b_high_f32x4);
    if (n) goto nk_angular_bf16_neon_cycle;
    nk_f32_t ab = vaddvq_f32(ab_f32x4);
    nk_f32_t a2 = vaddvq_f32(a2_f32x4);
    nk_f32_t b2 = vaddvq_f32(b2_f32x4);
    *result = nk_angular_normalize_f32_neon_(ab, a2, b2);
}

NK_PUBLIC void nk_sqeuclidean_e2m3_neon(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    float16x8_t a_f16x8, b_f16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_sqeuclidean_e2m3_neon_cycle:
    if (n < 8) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b8x8_serial_(a, &a_vec, n);
        nk_partial_load_b8x8_serial_(b, &b_vec, n);
        a_f16x8 = nk_e2m3x8_to_f16x8_neon_(a_vec.u8x8);
        b_f16x8 = nk_e2m3x8_to_f16x8_neon_(b_vec.u8x8);
        n = 0;
    }
    else {
        a_f16x8 = nk_e2m3x8_to_f16x8_neon_(vld1_u8(a));
        b_f16x8 = nk_e2m3x8_to_f16x8_neon_(vld1_u8(b));
        a += 8, b += 8, n -= 8;
    }
    float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
    float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_f32_f16(vget_high_f16(b_f16x8));
    float32x4_t diff_low_f32x4 = vsubq_f32(a_low_f32x4, b_low_f32x4);
    float32x4_t diff_high_f32x4 = vsubq_f32(a_high_f32x4, b_high_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_low_f32x4, diff_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_high_f32x4, diff_high_f32x4);
    if (n) goto nk_sqeuclidean_e2m3_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_e2m3_neon(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e2m3_neon(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_e2m3_neon(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    float16x8_t a_f16x8, b_f16x8;
    float32x4_t ab_f32x4 = vdupq_n_f32(0);
    float32x4_t a2_f32x4 = vdupq_n_f32(0);
    float32x4_t b2_f32x4 = vdupq_n_f32(0);
nk_angular_e2m3_neon_cycle:
    if (n < 8) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b8x8_serial_(a, &a_vec, n);
        nk_partial_load_b8x8_serial_(b, &b_vec, n);
        a_f16x8 = nk_e2m3x8_to_f16x8_neon_(a_vec.u8x8);
        b_f16x8 = nk_e2m3x8_to_f16x8_neon_(b_vec.u8x8);
        n = 0;
    }
    else {
        a_f16x8 = nk_e2m3x8_to_f16x8_neon_(vld1_u8(a));
        b_f16x8 = nk_e2m3x8_to_f16x8_neon_(vld1_u8(b));
        a += 8, b += 8, n -= 8;
    }
    float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
    float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_f32_f16(vget_high_f16(b_f16x8));
    ab_f32x4 = vfmaq_f32(ab_f32x4, a_low_f32x4, b_low_f32x4);
    ab_f32x4 = vfmaq_f32(ab_f32x4, a_high_f32x4, b_high_f32x4);
    a2_f32x4 = vfmaq_f32(a2_f32x4, a_low_f32x4, a_low_f32x4);
    a2_f32x4 = vfmaq_f32(a2_f32x4, a_high_f32x4, a_high_f32x4);
    b2_f32x4 = vfmaq_f32(b2_f32x4, b_low_f32x4, b_low_f32x4);
    b2_f32x4 = vfmaq_f32(b2_f32x4, b_high_f32x4, b_high_f32x4);
    if (n) goto nk_angular_e2m3_neon_cycle;
    nk_f32_t ab = vaddvq_f32(ab_f32x4);
    nk_f32_t a2 = vaddvq_f32(a2_f32x4);
    nk_f32_t b2 = vaddvq_f32(b2_f32x4);
    *result = nk_angular_normalize_f32_neon_(ab, a2, b2);
}

NK_PUBLIC void nk_sqeuclidean_e3m2_neon(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    float16x8_t a_f16x8, b_f16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_sqeuclidean_e3m2_neon_cycle:
    if (n < 8) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b8x8_serial_(a, &a_vec, n);
        nk_partial_load_b8x8_serial_(b, &b_vec, n);
        a_f16x8 = nk_e3m2x8_to_f16x8_neon_(a_vec.u8x8);
        b_f16x8 = nk_e3m2x8_to_f16x8_neon_(b_vec.u8x8);
        n = 0;
    }
    else {
        a_f16x8 = nk_e3m2x8_to_f16x8_neon_(vld1_u8(a));
        b_f16x8 = nk_e3m2x8_to_f16x8_neon_(vld1_u8(b));
        a += 8, b += 8, n -= 8;
    }
    float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
    float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_f32_f16(vget_high_f16(b_f16x8));
    float32x4_t diff_low_f32x4 = vsubq_f32(a_low_f32x4, b_low_f32x4);
    float32x4_t diff_high_f32x4 = vsubq_f32(a_high_f32x4, b_high_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_low_f32x4, diff_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_high_f32x4, diff_high_f32x4);
    if (n) goto nk_sqeuclidean_e3m2_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_e3m2_neon(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e3m2_neon(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_e3m2_neon(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    float16x8_t a_f16x8, b_f16x8;
    float32x4_t ab_f32x4 = vdupq_n_f32(0);
    float32x4_t a2_f32x4 = vdupq_n_f32(0);
    float32x4_t b2_f32x4 = vdupq_n_f32(0);
nk_angular_e3m2_neon_cycle:
    if (n < 8) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b8x8_serial_(a, &a_vec, n);
        nk_partial_load_b8x8_serial_(b, &b_vec, n);
        a_f16x8 = nk_e3m2x8_to_f16x8_neon_(a_vec.u8x8);
        b_f16x8 = nk_e3m2x8_to_f16x8_neon_(b_vec.u8x8);
        n = 0;
    }
    else {
        a_f16x8 = nk_e3m2x8_to_f16x8_neon_(vld1_u8(a));
        b_f16x8 = nk_e3m2x8_to_f16x8_neon_(vld1_u8(b));
        a += 8, b += 8, n -= 8;
    }
    float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
    float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_f32_f16(vget_high_f16(b_f16x8));
    ab_f32x4 = vfmaq_f32(ab_f32x4, a_low_f32x4, b_low_f32x4);
    ab_f32x4 = vfmaq_f32(ab_f32x4, a_high_f32x4, b_high_f32x4);
    a2_f32x4 = vfmaq_f32(a2_f32x4, a_low_f32x4, a_low_f32x4);
    a2_f32x4 = vfmaq_f32(a2_f32x4, a_high_f32x4, a_high_f32x4);
    b2_f32x4 = vfmaq_f32(b2_f32x4, b_low_f32x4, b_low_f32x4);
    b2_f32x4 = vfmaq_f32(b2_f32x4, b_high_f32x4, b_high_f32x4);
    if (n) goto nk_angular_e3m2_neon_cycle;
    nk_f32_t ab = vaddvq_f32(ab_f32x4);
    nk_f32_t a2 = vaddvq_f32(a2_f32x4);
    nk_f32_t b2 = vaddvq_f32(b2_f32x4);
    *result = nk_angular_normalize_f32_neon_(ab, a2, b2);
}

NK_PUBLIC void nk_sqeuclidean_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    float16x8_t a_f16x8, b_f16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_sqeuclidean_e4m3_neon_cycle:
    if (n < 8) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b8x8_serial_(a, &a_vec, n);
        nk_partial_load_b8x8_serial_(b, &b_vec, n);
        a_f16x8 = nk_e4m3x8_to_f16x8_neon_(a_vec.u8x8);
        b_f16x8 = nk_e4m3x8_to_f16x8_neon_(b_vec.u8x8);
        n = 0;
    }
    else {
        a_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(a));
        b_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(b));
        a += 8, b += 8, n -= 8;
    }
    float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
    float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_f32_f16(vget_high_f16(b_f16x8));
    float32x4_t diff_low_f32x4 = vsubq_f32(a_low_f32x4, b_low_f32x4);
    float32x4_t diff_high_f32x4 = vsubq_f32(a_high_f32x4, b_high_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_low_f32x4, diff_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_high_f32x4, diff_high_f32x4);
    if (n) goto nk_sqeuclidean_e4m3_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e4m3_neon(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    float16x8_t a_f16x8, b_f16x8;
    float32x4_t ab_f32x4 = vdupq_n_f32(0);
    float32x4_t a2_f32x4 = vdupq_n_f32(0);
    float32x4_t b2_f32x4 = vdupq_n_f32(0);
nk_angular_e4m3_neon_cycle:
    if (n < 8) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b8x8_serial_(a, &a_vec, n);
        nk_partial_load_b8x8_serial_(b, &b_vec, n);
        a_f16x8 = nk_e4m3x8_to_f16x8_neon_(a_vec.u8x8);
        b_f16x8 = nk_e4m3x8_to_f16x8_neon_(b_vec.u8x8);
        n = 0;
    }
    else {
        a_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(a));
        b_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(b));
        a += 8, b += 8, n -= 8;
    }
    float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
    float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_f32_f16(vget_high_f16(b_f16x8));
    ab_f32x4 = vfmaq_f32(ab_f32x4, a_low_f32x4, b_low_f32x4);
    ab_f32x4 = vfmaq_f32(ab_f32x4, a_high_f32x4, b_high_f32x4);
    a2_f32x4 = vfmaq_f32(a2_f32x4, a_low_f32x4, a_low_f32x4);
    a2_f32x4 = vfmaq_f32(a2_f32x4, a_high_f32x4, a_high_f32x4);
    b2_f32x4 = vfmaq_f32(b2_f32x4, b_low_f32x4, b_low_f32x4);
    b2_f32x4 = vfmaq_f32(b2_f32x4, b_high_f32x4, b_high_f32x4);
    if (n) goto nk_angular_e4m3_neon_cycle;
    nk_f32_t ab = vaddvq_f32(ab_f32x4);
    nk_f32_t a2 = vaddvq_f32(a2_f32x4);
    nk_f32_t b2 = vaddvq_f32(b2_f32x4);
    *result = nk_angular_normalize_f32_neon_(ab, a2, b2);
}

NK_PUBLIC void nk_sqeuclidean_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    float16x8_t a_f16x8, b_f16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_sqeuclidean_e5m2_neon_cycle:
    if (n < 8) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b8x8_serial_(a, &a_vec, n);
        nk_partial_load_b8x8_serial_(b, &b_vec, n);
        a_f16x8 = nk_e5m2x8_to_f16x8_neon_(a_vec.u8x8);
        b_f16x8 = nk_e5m2x8_to_f16x8_neon_(b_vec.u8x8);
        n = 0;
    }
    else {
        a_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8(a));
        b_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8(b));
        a += 8, b += 8, n -= 8;
    }
    float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
    float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_f32_f16(vget_high_f16(b_f16x8));
    float32x4_t diff_low_f32x4 = vsubq_f32(a_low_f32x4, b_low_f32x4);
    float32x4_t diff_high_f32x4 = vsubq_f32(a_high_f32x4, b_high_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_low_f32x4, diff_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_high_f32x4, diff_high_f32x4);
    if (n) goto nk_sqeuclidean_e5m2_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e5m2_neon(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    float16x8_t a_f16x8, b_f16x8;
    float32x4_t ab_f32x4 = vdupq_n_f32(0);
    float32x4_t a2_f32x4 = vdupq_n_f32(0);
    float32x4_t b2_f32x4 = vdupq_n_f32(0);
nk_angular_e5m2_neon_cycle:
    if (n < 8) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b8x8_serial_(a, &a_vec, n);
        nk_partial_load_b8x8_serial_(b, &b_vec, n);
        a_f16x8 = nk_e5m2x8_to_f16x8_neon_(a_vec.u8x8);
        b_f16x8 = nk_e5m2x8_to_f16x8_neon_(b_vec.u8x8);
        n = 0;
    }
    else {
        a_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8(a));
        b_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8(b));
        a += 8, b += 8, n -= 8;
    }
    float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
    float32x4_t a_high_f32x4 = vcvt_f32_f16(vget_high_f16(a_f16x8));
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_f32_f16(vget_high_f16(b_f16x8));
    ab_f32x4 = vfmaq_f32(ab_f32x4, a_low_f32x4, b_low_f32x4);
    ab_f32x4 = vfmaq_f32(ab_f32x4, a_high_f32x4, b_high_f32x4);
    a2_f32x4 = vfmaq_f32(a2_f32x4, a_low_f32x4, a_low_f32x4);
    a2_f32x4 = vfmaq_f32(a2_f32x4, a_high_f32x4, a_high_f32x4);
    b2_f32x4 = vfmaq_f32(b2_f32x4, b_low_f32x4, b_low_f32x4);
    b2_f32x4 = vfmaq_f32(b2_f32x4, b_high_f32x4, b_high_f32x4);
    if (n) goto nk_angular_e5m2_neon_cycle;
    nk_f32_t ab = vaddvq_f32(ab_f32x4);
    nk_f32_t a2 = vaddvq_f32(a2_f32x4);
    nk_f32_t b2 = vaddvq_f32(b2_f32x4);
    *result = nk_angular_normalize_f32_neon_(ab, a2, b2);
}

/** @brief Angular from_dot: computes 1 − dot × rsqrt(query_sumsq × target_sumsq) for 4 pairs in f64. */
NK_INTERNAL void nk_angular_through_f64_from_dot_neon_(nk_b256_vec_t dots, nk_f64_t query_sumsq,
                                                       nk_b256_vec_t target_sumsqs, nk_b256_vec_t *results) {
    float64x2_t dots_ab_f64x2 = dots.f64x2s[0];
    float64x2_t dots_cd_f64x2 = dots.f64x2s[1];
    float64x2_t query_sumsq_f64x2 = vdupq_n_f64(query_sumsq);
    float64x2_t target_sumsqs_ab_f64x2 = target_sumsqs.f64x2s[0];
    float64x2_t target_sumsqs_cd_f64x2 = target_sumsqs.f64x2s[1];

    // products = query_sumsq * target_sumsq
    float64x2_t products_ab_f64x2 = vmulq_f64(query_sumsq_f64x2, target_sumsqs_ab_f64x2);
    float64x2_t products_cd_f64x2 = vmulq_f64(query_sumsq_f64x2, target_sumsqs_cd_f64x2);

    // rsqrt with Newton-Raphson (2 iterations for ~48-bit precision)
    float64x2_t rsqrt_ab_f64x2 = vrsqrteq_f64(products_ab_f64x2);
    float64x2_t rsqrt_cd_f64x2 = vrsqrteq_f64(products_cd_f64x2);
    rsqrt_ab_f64x2 = vmulq_f64(rsqrt_ab_f64x2,
                               vrsqrtsq_f64(vmulq_f64(products_ab_f64x2, rsqrt_ab_f64x2), rsqrt_ab_f64x2));
    rsqrt_cd_f64x2 = vmulq_f64(rsqrt_cd_f64x2,
                               vrsqrtsq_f64(vmulq_f64(products_cd_f64x2, rsqrt_cd_f64x2), rsqrt_cd_f64x2));
    rsqrt_ab_f64x2 = vmulq_f64(rsqrt_ab_f64x2,
                               vrsqrtsq_f64(vmulq_f64(products_ab_f64x2, rsqrt_ab_f64x2), rsqrt_ab_f64x2));
    rsqrt_cd_f64x2 = vmulq_f64(rsqrt_cd_f64x2,
                               vrsqrtsq_f64(vmulq_f64(products_cd_f64x2, rsqrt_cd_f64x2), rsqrt_cd_f64x2));

    // angular = 1 − dot × rsqrt(product)
    float64x2_t ones_f64x2 = vdupq_n_f64(1.0);
    float64x2_t zeros_f64x2 = vdupq_n_f64(0.0);
    float64x2_t result_ab_f64x2 = vsubq_f64(ones_f64x2, vmulq_f64(dots_ab_f64x2, rsqrt_ab_f64x2));
    float64x2_t result_cd_f64x2 = vsubq_f64(ones_f64x2, vmulq_f64(dots_cd_f64x2, rsqrt_cd_f64x2));

    // Clamp to [0, inf)
    result_ab_f64x2 = vmaxq_f64(result_ab_f64x2, zeros_f64x2);
    result_cd_f64x2 = vmaxq_f64(result_cd_f64x2, zeros_f64x2);

    // Handle edge cases with vectorized selects
    uint64x2_t products_zero_ab_u64x2 = vceqq_f64(products_ab_f64x2, zeros_f64x2);
    uint64x2_t products_zero_cd_u64x2 = vceqq_f64(products_cd_f64x2, zeros_f64x2);
    uint64x2_t dots_zero_ab_u64x2 = vceqq_f64(dots_ab_f64x2, zeros_f64x2);
    uint64x2_t dots_zero_cd_u64x2 = vceqq_f64(dots_cd_f64x2, zeros_f64x2);

    // Both zero → result = 0; products zero but dots nonzero → result = 1
    uint64x2_t both_zero_ab_u64x2 = vandq_u64(products_zero_ab_u64x2, dots_zero_ab_u64x2);
    uint64x2_t both_zero_cd_u64x2 = vandq_u64(products_zero_cd_u64x2, dots_zero_cd_u64x2);
    result_ab_f64x2 = vbslq_f64(both_zero_ab_u64x2, zeros_f64x2, result_ab_f64x2);
    result_cd_f64x2 = vbslq_f64(both_zero_cd_u64x2, zeros_f64x2, result_cd_f64x2);

    uint64x2_t prod_zero_dot_nonzero_ab_u64x2 = vandq_u64(
        products_zero_ab_u64x2, vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(dots_zero_ab_u64x2))));
    uint64x2_t prod_zero_dot_nonzero_cd_u64x2 = vandq_u64(
        products_zero_cd_u64x2, vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(dots_zero_cd_u64x2))));
    result_ab_f64x2 = vbslq_f64(prod_zero_dot_nonzero_ab_u64x2, ones_f64x2, result_ab_f64x2);
    result_cd_f64x2 = vbslq_f64(prod_zero_dot_nonzero_cd_u64x2, ones_f64x2, result_cd_f64x2);

    results->f64x2s[0] = result_ab_f64x2;
    results->f64x2s[1] = result_cd_f64x2;
}

/** @brief Euclidean from_dot: computes √(query_sumsq + target_sumsq − 2 × dot) for 4 pairs in f64. */
NK_INTERNAL void nk_euclidean_through_f64_from_dot_neon_(nk_b256_vec_t dots, nk_f64_t query_sumsq,
                                                         nk_b256_vec_t target_sumsqs, nk_b256_vec_t *results) {
    float64x2_t dots_ab_f64x2 = dots.f64x2s[0];
    float64x2_t dots_cd_f64x2 = dots.f64x2s[1];
    float64x2_t query_sumsq_f64x2 = vdupq_n_f64(query_sumsq);
    float64x2_t target_sumsqs_ab_f64x2 = target_sumsqs.f64x2s[0];
    float64x2_t target_sumsqs_cd_f64x2 = target_sumsqs.f64x2s[1];

    // dist_sq = query_sumsq + target_sumsq − 2 × dot
    float64x2_t neg_two_f64x2 = vdupq_n_f64(-2.0);
    float64x2_t sum_sq_ab_f64x2 = vaddq_f64(query_sumsq_f64x2, target_sumsqs_ab_f64x2);
    float64x2_t sum_sq_cd_f64x2 = vaddq_f64(query_sumsq_f64x2, target_sumsqs_cd_f64x2);
    float64x2_t dist_sq_ab_f64x2 = vfmaq_f64(sum_sq_ab_f64x2, neg_two_f64x2, dots_ab_f64x2);
    float64x2_t dist_sq_cd_f64x2 = vfmaq_f64(sum_sq_cd_f64x2, neg_two_f64x2, dots_cd_f64x2);

    // Clamp and sqrt in f64
    float64x2_t zeros_f64x2 = vdupq_n_f64(0.0);
    dist_sq_ab_f64x2 = vmaxq_f64(dist_sq_ab_f64x2, zeros_f64x2);
    dist_sq_cd_f64x2 = vmaxq_f64(dist_sq_cd_f64x2, zeros_f64x2);
    float64x2_t dist_ab_f64x2 = vsqrtq_f64(dist_sq_ab_f64x2);
    float64x2_t dist_cd_f64x2 = vsqrtq_f64(dist_sq_cd_f64x2);

    results->f64x2s[0] = dist_ab_f64x2;
    results->f64x2s[1] = dist_cd_f64x2;
}

/** @brief Angular from_dot: computes 1 − dot × rsqrt(query_sumsq × target_sumsq) for 4 pairs in f32. */
NK_INTERNAL void nk_angular_through_f32_from_dot_neon_(nk_b128_vec_t dots, nk_f32_t query_sumsq,
                                                       nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    float32x4_t dots_f32x4 = dots.f32x4;
    float32x4_t query_sumsq_f32x4 = vdupq_n_f32(query_sumsq);
    float32x4_t products_f32x4 = vmulq_f32(query_sumsq_f32x4, target_sumsqs.f32x4);

    // rsqrt with Newton-Raphson refinement (2 iterations)
    float32x4_t rsqrt_f32x4 = vrsqrteq_f32(products_f32x4);
    rsqrt_f32x4 = vmulq_f32(rsqrt_f32x4, vrsqrtsq_f32(vmulq_f32(products_f32x4, rsqrt_f32x4), rsqrt_f32x4));
    rsqrt_f32x4 = vmulq_f32(rsqrt_f32x4, vrsqrtsq_f32(vmulq_f32(products_f32x4, rsqrt_f32x4), rsqrt_f32x4));

    float32x4_t normalized_f32x4 = vmulq_f32(dots_f32x4, rsqrt_f32x4);
    float32x4_t angular_f32x4 = vsubq_f32(vdupq_n_f32(1.0f), normalized_f32x4);
    results->f32x4 = vmaxq_f32(angular_f32x4, vdupq_n_f32(0.0f));
}

/** @brief Euclidean from_dot: computes √(query_sumsq + target_sumsq − 2 × dot) for 4 pairs in f32. */
NK_INTERNAL void nk_euclidean_through_f32_from_dot_neon_(nk_b128_vec_t dots, nk_f32_t query_sumsq,
                                                         nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    float32x4_t dots_f32x4 = dots.f32x4;
    float32x4_t query_sumsq_f32x4 = vdupq_n_f32(query_sumsq);
    float32x4_t sum_sq_f32x4 = vaddq_f32(query_sumsq_f32x4, target_sumsqs.f32x4);
    // dist_sq = sum_sq − 2 × dot
    float32x4_t dist_sq_f32x4 = vfmsq_f32(sum_sq_f32x4, vdupq_n_f32(2.0f), dots_f32x4);
    // Clamp and sqrt
    dist_sq_f32x4 = vmaxq_f32(dist_sq_f32x4, vdupq_n_f32(0.0f));
    results->f32x4 = vsqrtq_f32(dist_sq_f32x4);
}

/** @brief Angular from_dot for i32 accumulators: cast to f32, rsqrt+NR, clamp. 4 pairs. */
NK_INTERNAL void nk_angular_through_i32_from_dot_neon_(nk_b128_vec_t dots, nk_i32_t query_sumsq,
                                                       nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    float32x4_t dots_f32x4 = vcvtq_f32_s32(dots.i32x4);
    float32x4_t query_sumsq_f32x4 = vdupq_n_f32((nk_f32_t)query_sumsq);
    float32x4_t products_f32x4 = vmulq_f32(query_sumsq_f32x4, vcvtq_f32_s32(target_sumsqs.i32x4));
    float32x4_t rsqrt_f32x4 = nk_rsqrt_f32x4_neon_(products_f32x4);
    float32x4_t normalized_f32x4 = vmulq_f32(dots_f32x4, rsqrt_f32x4);
    float32x4_t angular_f32x4 = vsubq_f32(vdupq_n_f32(1.0f), normalized_f32x4);
    results->f32x4 = vmaxq_f32(angular_f32x4, vdupq_n_f32(0.0f));
}

/** @brief Euclidean from_dot for i32 accumulators: cast to f32, then √(a² + b² − 2ab). 4 pairs. */
NK_INTERNAL void nk_euclidean_through_i32_from_dot_neon_(nk_b128_vec_t dots, nk_i32_t query_sumsq,
                                                         nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    float32x4_t dots_f32x4 = vcvtq_f32_s32(dots.i32x4);
    float32x4_t query_sumsq_f32x4 = vdupq_n_f32((nk_f32_t)query_sumsq);
    float32x4_t sum_sq_f32x4 = vaddq_f32(query_sumsq_f32x4, vcvtq_f32_s32(target_sumsqs.i32x4));
    float32x4_t dist_sq_f32x4 = vfmsq_f32(sum_sq_f32x4, vdupq_n_f32(2.0f), dots_f32x4);
    dist_sq_f32x4 = vmaxq_f32(dist_sq_f32x4, vdupq_n_f32(0.0f));
    results->f32x4 = vsqrtq_f32(dist_sq_f32x4);
}

/** @brief Angular from_dot for u32 accumulators: cast to f32, rsqrt+NR, clamp. 4 pairs. */
NK_INTERNAL void nk_angular_through_u32_from_dot_neon_(nk_b128_vec_t dots, nk_u32_t query_sumsq,
                                                       nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    float32x4_t dots_f32x4 = vcvtq_f32_u32(dots.u32x4);
    float32x4_t query_sumsq_f32x4 = vdupq_n_f32((nk_f32_t)query_sumsq);
    float32x4_t products_f32x4 = vmulq_f32(query_sumsq_f32x4, vcvtq_f32_u32(target_sumsqs.u32x4));
    float32x4_t rsqrt_f32x4 = nk_rsqrt_f32x4_neon_(products_f32x4);
    float32x4_t normalized_f32x4 = vmulq_f32(dots_f32x4, rsqrt_f32x4);
    float32x4_t angular_f32x4 = vsubq_f32(vdupq_n_f32(1.0f), normalized_f32x4);
    results->f32x4 = vmaxq_f32(angular_f32x4, vdupq_n_f32(0.0f));
}

/** @brief Euclidean from_dot for u32 accumulators: cast to f32, then √(a² + b² − 2ab). 4 pairs. */
NK_INTERNAL void nk_euclidean_through_u32_from_dot_neon_(nk_b128_vec_t dots, nk_u32_t query_sumsq,
                                                         nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    float32x4_t dots_f32x4 = vcvtq_f32_u32(dots.u32x4);
    float32x4_t query_sumsq_f32x4 = vdupq_n_f32((nk_f32_t)query_sumsq);
    float32x4_t sum_sq_f32x4 = vaddq_f32(query_sumsq_f32x4, vcvtq_f32_u32(target_sumsqs.u32x4));
    float32x4_t dist_sq_f32x4 = vfmsq_f32(sum_sq_f32x4, vdupq_n_f32(2.0f), dots_f32x4);
    dist_sq_f32x4 = vmaxq_f32(dist_sq_f32x4, vdupq_n_f32(0.0f));
    results->f32x4 = vsqrtq_f32(dist_sq_f32x4);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma endregion - Smaller Floats
#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_
#endif // NK_SPATIAL_NEON_H
