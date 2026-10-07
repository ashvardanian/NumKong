/**
 *  @file include/numkong/spatial/neon.h
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief SIMD-accelerated spatial similarity measures for NEON.
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_neon_instructions Key NEON Spatial Instructions
 *
 *  ARM NEON instructions for distance computations:
 *
 *  @verbatim
 *  Intrinsic     Instruction              A76        M5
 *  vfmaq_f32     FMLA (V.4S, V.4S, V.4S)  4cy @ 2p   3cy @ 4p
 *  vmulq_f32     FMUL (V.4S, V.4S, V.4S)  3cy @ 2p   3cy @ 4p
 *  vaddq_f32     FADD (V.4S, V.4S, V.4S)  2cy @ 2p   2cy @ 4p
 *  vsubq_f32     FSUB (V.4S, V.4S, V.4S)  2cy @ 2p   2cy @ 4p
 *  vrsqrteq_f32  FRSQRTE (V.4S, V.4S)     2cy @ 2p   3cy @ 1p
 *  vsqrtq_f32    FSQRT (V.4S, V.4S)       12cy @ 1p  9cy @ 1p
 *  vrecpeq_f32   FRECPE (V.4S, V.4S)      2cy @ 2p   3cy @ 1p
 *  @endverbatim
 *
 *  FRSQRTE provides ~8-bit precision; two Newton-Raphson iterations via vrsqrtsq_f32 achieve
 *  ~23-bit precision, sufficient for f32. This is much faster than FSQRT (0.25/cy).
 *
 *  Distance computations (L2, angular) benefit from 2x throughput on 4-pipe cores (Apple M4+,
 *  Graviton3+, Oryon), but FSQRT remains slow on all cores. Use rsqrt+NR when precision allows.
 */
#ifndef NUMKONG_SPATIAL_NEON_H
#define NUMKONG_SPATIAL_NEON_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_NEON_

#include "numkong/types.h"
#include "numkong/dot/neon.h" // `nk_dot_stable_sum_f64x2_neon_`

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
 *  Uses @c vrsqrteq_f32 (~8-bit initial estimate) followed by two Newton-Raphson iterations
 *  via @c vrsqrtsq_f32, achieving ~23-bit precision — sufficient for f32.
 *  Much faster than @c vsqrtq_f32 (2 cy vs 9-12 cy latency, 2/cy vs 0.25/cy throughput).
 *  @c vmulxq_f32 takes ∞ × 0 as 2, so an infinite input keeps its estimate of 0.
 */
NUMKONG_INLINE float32x4_t nk_rsqrt_f32x4_neon_(float32x4_t x) {
    float32x4_t rsqrt_f32x4 = vrsqrteq_f32(x);
    rsqrt_f32x4 = vmulq_f32(rsqrt_f32x4, vrsqrtsq_f32(vmulxq_f32(x, rsqrt_f32x4), rsqrt_f32x4));
    rsqrt_f32x4 = vmulq_f32(rsqrt_f32x4, vrsqrtsq_f32(vmulxq_f32(x, rsqrt_f32x4), rsqrt_f32x4));
    return rsqrt_f32x4;
}

/**
 *  @brief Reciprocal square root of 2 doubles with Newton-Raphson refinement.
 *
 *  Uses @c vrsqrteq_f64 (~8-bit initial estimate) followed by three Newton-Raphson iterations via
 *  @c vrsqrtsq_f64, achieving ~48-bit precision. That is reasonable for f64 distance computations,
 *  whose result is often narrowed to f32, while full 52-bit mantissa fidelity needs @c vsqrtq_f64.
 *  @c vmulxq_f64 keeps an infinite input's estimate of 0, like @c nk_rsqrt_f32x4_neon_.
 */
NUMKONG_INLINE float64x2_t nk_rsqrt_f64x2_neon_(float64x2_t x) {
    float64x2_t rsqrt_f64x2 = vrsqrteq_f64(x);
    rsqrt_f64x2 = vmulq_f64(rsqrt_f64x2, vrsqrtsq_f64(vmulxq_f64(x, rsqrt_f64x2), rsqrt_f64x2));
    rsqrt_f64x2 = vmulq_f64(rsqrt_f64x2, vrsqrtsq_f64(vmulxq_f64(x, rsqrt_f64x2), rsqrt_f64x2));
    rsqrt_f64x2 = vmulq_f64(rsqrt_f64x2, vrsqrtsq_f64(vmulxq_f64(x, rsqrt_f64x2), rsqrt_f64x2));
    return rsqrt_f64x2;
}

NUMKONG_INLINE nk_f32_t nk_angular_normalize_f32_neon_(nk_f32_t ab, nk_f32_t a2, nk_f32_t b2) {
    if (a2 == 0 && b2 == 0) return 0;
    if (ab == 0) return 1;
    nk_f32_t squares_arr[2] = {a2, b2};
    float32x2_t squares_f32x2 = vld1_f32(squares_arr);
    // Unlike x86, Arm NEON manuals don't explicitly mention the accuracy of their `rsqrt` approximation.
    // Third-party research suggests that it's less accurate than SSE instructions, having an error of 1.5×2⁻¹².
    // One or two rounds of Newton-Raphson refinement are recommended to improve the accuracy.
    // https://github.com/lighttransport/embree-aarch64/issues/24
    // https://github.com/lighttransport/embree-aarch64/blob/3f75f8cb4e553d13dced941b5fefd4c826835a6b/common/math/math.h#L137-L145
    float32x2_t rsqrts_f32x2 = vrsqrte_f32(squares_f32x2);
    // Perform two rounds of Newton-Raphson refinement:
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts_f32x2 = vmul_f32(rsqrts_f32x2, vrsqrts_f32(vmul_f32(squares_f32x2, rsqrts_f32x2), rsqrts_f32x2));
    rsqrts_f32x2 = vmul_f32(rsqrts_f32x2, vrsqrts_f32(vmul_f32(squares_f32x2, rsqrts_f32x2), rsqrts_f32x2));
    vst1_f32(squares_arr, rsqrts_f32x2);
    nk_f32_t result = 1 - ab * squares_arr[0] * squares_arr[1];
    return result > 0 ? result : 0;
}

NUMKONG_INLINE nk_f64_t nk_angular_normalize_f64_neon_(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {
    if (a2 == 0 && b2 == 0) return 0;
    if (ab == 0) return 1;
    nk_f64_t squares_arr[2] = {a2, b2};
    float64x2_t squares_f64x2 = vld1q_f64(squares_arr);

    // Unlike x86, Arm NEON manuals don't explicitly mention the accuracy of their `rsqrt` approximation.
    // Third-party research suggests that it's less accurate than SSE instructions, having an error of 1.5×2⁻¹².
    // One or two rounds of Newton-Raphson refinement are recommended to improve the accuracy.
    // https://github.com/lighttransport/embree-aarch64/issues/24
    // https://github.com/lighttransport/embree-aarch64/blob/3f75f8cb4e553d13dced941b5fefd4c826835a6b/common/math/math.h#L137-L145
    float64x2_t rsqrts_f64x2 = vrsqrteq_f64(squares_f64x2);
    // Perform three rounds of Newton-Raphson refinement for f64 precision (~48 bits):
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts_f64x2 = vmulq_f64(rsqrts_f64x2, vrsqrtsq_f64(vmulq_f64(squares_f64x2, rsqrts_f64x2), rsqrts_f64x2));
    rsqrts_f64x2 = vmulq_f64(rsqrts_f64x2, vrsqrtsq_f64(vmulq_f64(squares_f64x2, rsqrts_f64x2), rsqrts_f64x2));
    rsqrts_f64x2 = vmulq_f64(rsqrts_f64x2, vrsqrtsq_f64(vmulq_f64(squares_f64x2, rsqrts_f64x2), rsqrts_f64x2));
    vst1q_f64(squares_arr, rsqrts_f64x2);
    nk_f64_t result = 1 - ab * squares_arr[0] * squares_arr[1];
    return result > 0 ? result : 0;
}

#pragma region F32 and F64 Floats

/** Sums the squared differences of @p n F32 pairs, widened to F64. */
NUMKONG_INLINE void nk_squared_distance_f32_neon_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t diff_f64x2 = vsubq_f64(vcvt_f64_f32(vld1_f32(a + i)), vcvt_f64_f32(vld1_f32(b + i)));
        sum_f64x2 = vfmaq_f64(sum_f64x2, diff_f64x2, diff_f64x2);
    }
    nk_f64_t sum = vaddvq_f64(sum_f64x2);
    if (i < n) {
        nk_f64_t const diff = (nk_f64_t)a[i] - (nk_f64_t)b[i];
        sum += diff * diff;
    }
    *result = sum;
}

/** Sums the squared differences of @p n F64 pairs in Dot2. */
NUMKONG_INLINE void nk_squared_distance_f64_neon_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    float64x2_t sum_f64x2 = vdupq_n_f64(0), compensation_f64x2 = vdupq_n_f64(0);
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
    nk_dot2_f64x2_neon_(&sum_f64x2, &compensation_f64x2, diff_f64x2, diff_f64x2);
    if (n) goto nk_sqeuclidean_f64_neon_cycle;

    *result = nk_dot_stable_sum_f64x2_neon_(sum_f64x2, compensation_f64x2);
}

#pragma endregion F32 and F64 Floats
#pragma region F16 and BF16 Floats

/** Sums the squared differences of @p n BF16 pairs, widened to F32. */
NUMKONG_INLINE void nk_squared_distance_bf16_neon_(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                   nk_f32_t *result) {
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
    float32x4_t a_high_f32x4 = vreinterpretq_f32_u32(vshll_high_n_u16(a_u16x8, 16));
    float32x4_t b_low_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b_u16x8), 16));
    float32x4_t b_high_f32x4 = vreinterpretq_f32_u32(vshll_high_n_u16(b_u16x8, 16));
    float32x4_t diff_low_f32x4 = vsubq_f32(a_low_f32x4, b_low_f32x4);
    float32x4_t diff_high_f32x4 = vsubq_f32(a_high_f32x4, b_high_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_low_f32x4, diff_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_high_f32x4, diff_high_f32x4);
    if (n) goto nk_sqeuclidean_bf16_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

/** Sums the squared differences of @p n F16 pairs, widened to F32. */
NUMKONG_INLINE void nk_squared_distance_f16_neon_(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    uint16x8_t a_u16x8, b_u16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_sqeuclidean_f16_neon_cycle:
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
    float16x8_t a_f16x8 = vreinterpretq_f16_u16(a_u16x8);
    float16x8_t b_f16x8 = vreinterpretq_f16_u16(b_u16x8);
    float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
    float32x4_t a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
    float32x4_t diff_low_f32x4 = vsubq_f32(a_low_f32x4, b_low_f32x4);
    float32x4_t diff_high_f32x4 = vsubq_f32(a_high_f32x4, b_high_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_low_f32x4, diff_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_high_f32x4, diff_high_f32x4);
    if (n) goto nk_sqeuclidean_f16_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

/** Sums the squared differences of @p n E2M3 pairs, widened to F32. */
NUMKONG_INLINE void nk_squared_distance_e2m3_neon_(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n,
                                                   nk_f32_t *result) {
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
    float32x4_t a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
    float32x4_t diff_low_f32x4 = vsubq_f32(a_low_f32x4, b_low_f32x4);
    float32x4_t diff_high_f32x4 = vsubq_f32(a_high_f32x4, b_high_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_low_f32x4, diff_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_high_f32x4, diff_high_f32x4);
    if (n) goto nk_sqeuclidean_e2m3_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

/** Sums the squared differences of @p n E3M2 pairs, widened to F32. */
NUMKONG_INLINE void nk_squared_distance_e3m2_neon_(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n,
                                                   nk_f32_t *result) {
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
    float32x4_t a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
    float32x4_t diff_low_f32x4 = vsubq_f32(a_low_f32x4, b_low_f32x4);
    float32x4_t diff_high_f32x4 = vsubq_f32(a_high_f32x4, b_high_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_low_f32x4, diff_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_high_f32x4, diff_high_f32x4);
    if (n) goto nk_sqeuclidean_e3m2_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

/** Sums the squared differences of @p n E4M3 pairs, widened to F32. */
NUMKONG_INLINE void nk_squared_distance_e4m3_neon_(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n,
                                                   nk_f32_t *result) {
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
    float32x4_t a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
    float32x4_t diff_low_f32x4 = vsubq_f32(a_low_f32x4, b_low_f32x4);
    float32x4_t diff_high_f32x4 = vsubq_f32(a_high_f32x4, b_high_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_low_f32x4, diff_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_high_f32x4, diff_high_f32x4);
    if (n) goto nk_sqeuclidean_e4m3_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

/** Sums the squared differences of @p n E5M2 pairs, widened to F32. */
NUMKONG_INLINE void nk_squared_distance_e5m2_neon_(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n,
                                                   nk_f32_t *result) {
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
    float32x4_t a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
    float32x4_t diff_low_f32x4 = vsubq_f32(a_low_f32x4, b_low_f32x4);
    float32x4_t diff_high_f32x4 = vsubq_f32(a_high_f32x4, b_high_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_low_f32x4, diff_low_f32x4);
    sum_f32x4 = vfmaq_f32(sum_f32x4, diff_high_f32x4, diff_high_f32x4);
    if (n) goto nk_sqeuclidean_e5m2_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

/** Angular from_dot: computes 1 − dot × rsqrt(q) × rsqrt(t) for 4 pairs in f64, where q is
 *  @p query_sumsq and t each target's sum of squares, with the rules of the serial variant.
 *  Separate reciprocal square roots avoid overflowing the product of two finite-but-large norms. */
NUMKONG_INLINE void nk_angular_through_f64_from_dot_neon_(nk_b256_vec_t const *dots_vec, nk_f64_t query_sumsq,
                                                          nk_b256_vec_t const *target_sumsqs_vec,
                                                          nk_b256_vec_t *result_vec) {
    float64x2_t const zeros_f64x2 = vdupq_n_f64(0), ones_f64x2 = vdupq_n_f64(1);
    float64x2_t const dots_ab_f64x2 = dots_vec->f64x2s[0], dots_cd_f64x2 = dots_vec->f64x2s[1];
    float64x2_t const query_sumsq_f64x2 = vdupq_n_f64(query_sumsq);
    float64x2_t const target_sumsqs_ab_f64x2 = target_sumsqs_vec->f64x2s[0];
    float64x2_t const target_sumsqs_cd_f64x2 = target_sumsqs_vec->f64x2s[1];

    float64x2_t const query_rsqrt_f64x2 = nk_rsqrt_f64x2_neon_(query_sumsq_f64x2);
    float64x2_t const rsqrt_ab_f64x2 = vmulq_f64(query_rsqrt_f64x2, nk_rsqrt_f64x2_neon_(target_sumsqs_ab_f64x2));
    float64x2_t const rsqrt_cd_f64x2 = vmulq_f64(query_rsqrt_f64x2, nk_rsqrt_f64x2_neon_(target_sumsqs_cd_f64x2));
    float64x2_t angular_ab_f64x2 = vmaxq_f64(vsubq_f64(ones_f64x2, vmulq_f64(dots_ab_f64x2, rsqrt_ab_f64x2)),
                                             zeros_f64x2);
    float64x2_t angular_cd_f64x2 = vmaxq_f64(vsubq_f64(ones_f64x2, vmulq_f64(dots_cd_f64x2, rsqrt_cd_f64x2)),
                                             zeros_f64x2);

    // A zero norm or an exactly zero dot gives 1, and two zero norms give 0
    uint64x2_t const query_zero_u64x2 = vceqzq_f64(query_sumsq_f64x2);
    uint64x2_t const unit_ab_u64x2 = vorrq_u64(vceqzq_f64(dots_ab_f64x2),
                                               vorrq_u64(query_zero_u64x2, vceqzq_f64(target_sumsqs_ab_f64x2)));
    uint64x2_t const unit_cd_u64x2 = vorrq_u64(vceqzq_f64(dots_cd_f64x2),
                                               vorrq_u64(query_zero_u64x2, vceqzq_f64(target_sumsqs_cd_f64x2)));
    angular_ab_f64x2 = vbslq_f64(unit_ab_u64x2, ones_f64x2, angular_ab_f64x2);
    angular_cd_f64x2 = vbslq_f64(unit_cd_u64x2, ones_f64x2, angular_cd_f64x2);
    angular_ab_f64x2 = vbslq_f64(vceqzq_f64(vaddq_f64(query_sumsq_f64x2, target_sumsqs_ab_f64x2)), zeros_f64x2,
                                 angular_ab_f64x2);
    angular_cd_f64x2 = vbslq_f64(vceqzq_f64(vaddq_f64(query_sumsq_f64x2, target_sumsqs_cd_f64x2)), zeros_f64x2,
                                 angular_cd_f64x2);

    // A NaN dot outranks the zero-norm cases
    result_vec->f64x2s[0] = vbslq_f64(vceqq_f64(dots_ab_f64x2, dots_ab_f64x2), angular_ab_f64x2, dots_ab_f64x2);
    result_vec->f64x2s[1] = vbslq_f64(vceqq_f64(dots_cd_f64x2, dots_cd_f64x2), angular_cd_f64x2, dots_cd_f64x2);
}

/** Euclidean from_dot: computes √(q + t − 2 × dot) for 4 pairs in f64, where q is @p query_sumsq
 *  and t each target's sum of squares. */
NUMKONG_INLINE void nk_euclidean_through_f64_from_dot_neon_(nk_b256_vec_t const *dots_vec, nk_f64_t query_sumsq,
                                                            nk_b256_vec_t const *target_sumsqs_vec,
                                                            nk_b256_vec_t *result_vec) {
    float64x2_t dots_ab_f64x2 = dots_vec->f64x2s[0];
    float64x2_t dots_cd_f64x2 = dots_vec->f64x2s[1];
    float64x2_t query_sumsq_f64x2 = vdupq_n_f64(query_sumsq);
    float64x2_t target_sumsqs_ab_f64x2 = target_sumsqs_vec->f64x2s[0];
    float64x2_t target_sumsqs_cd_f64x2 = target_sumsqs_vec->f64x2s[1];

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

    result_vec->f64x2s[0] = dist_ab_f64x2;
    result_vec->f64x2s[1] = dist_cd_f64x2;
}

/** Angular from_dot: computes 1 − dot × rsqrt(q) × rsqrt(t) for 4 pairs in f32, where q is
 *  @p query_sumsq and t each target's sum of squares, with the rules of the serial variant.
 *  Separate reciprocal square roots avoid overflowing the product of two finite-but-large norms. */
NUMKONG_INLINE void nk_angular_through_f32_from_dot_neon_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
                                                          nk_b128_vec_t const *target_sumsqs_vec,
                                                          nk_b128_vec_t *result_vec) {
    float32x4_t const zeros_f32x4 = vdupq_n_f32(0), ones_f32x4 = vdupq_n_f32(1), dots_f32x4 = dots_vec->f32x4;
    float32x4_t const query_sumsq_f32x4 = vdupq_n_f32(query_sumsq), target_sumsqs_f32x4 = target_sumsqs_vec->f32x4;
    float32x4_t const rsqrt_f32x4 = vmulq_f32(nk_rsqrt_f32x4_neon_(query_sumsq_f32x4),
                                              nk_rsqrt_f32x4_neon_(target_sumsqs_f32x4));
    float32x4_t angular_f32x4 = vmaxq_f32(vsubq_f32(ones_f32x4, vmulq_f32(dots_f32x4, rsqrt_f32x4)), zeros_f32x4);
    uint32x4_t const unit_u32x4 = vorrq_u32(vceqzq_f32(dots_f32x4),
                                            vorrq_u32(vceqzq_f32(query_sumsq_f32x4), vceqzq_f32(target_sumsqs_f32x4)));
    angular_f32x4 = vbslq_f32(unit_u32x4, ones_f32x4, angular_f32x4);
    angular_f32x4 = vbslq_f32(vceqzq_f32(vaddq_f32(query_sumsq_f32x4, target_sumsqs_f32x4)), zeros_f32x4,
                              angular_f32x4);
    // A NaN dot outranks the zero-norm cases
    result_vec->f32x4 = vbslq_f32(vceqq_f32(dots_f32x4, dots_f32x4), angular_f32x4, dots_f32x4);
}

/** Euclidean from_dot: computes √(q + t − 2 × dot) for 4 pairs in f32, where q is @p query_sumsq
 *  and t each target's sum of squares. */
NUMKONG_INLINE void nk_euclidean_through_f32_from_dot_neon_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
                                                            nk_b128_vec_t const *target_sumsqs_vec,
                                                            nk_b128_vec_t *result_vec) {
    float32x4_t dots_f32x4 = dots_vec->f32x4;
    float32x4_t query_sumsq_f32x4 = vdupq_n_f32(query_sumsq);
    float32x4_t sum_sq_f32x4 = vaddq_f32(query_sumsq_f32x4, target_sumsqs_vec->f32x4);
    // dist_sq = sum_sq − 2 × dot
    float32x4_t dist_sq_f32x4 = vfmsq_f32(sum_sq_f32x4, vdupq_n_f32(2.0f), dots_f32x4);
    // Clamp and sqrt
    dist_sq_f32x4 = vmaxq_f32(dist_sq_f32x4, vdupq_n_f32(0.0f));
    result_vec->f32x4 = vsqrtq_f32(dist_sq_f32x4);
}

/** Angular from_dot for i32 dots d against u32 norms a, b, with the rules of the f32 variant.
 *  With s = √(ab), a positive dot takes 1 − d / s as (ab − d²) / (ab + d · s), whose
 *  numerator is an exact integer by Cauchy–Schwarz, and any other dot as (ab − d · s) / ab,
 *  so neither cancels and equal vectors give exactly 0. UMULL and SMLSL form ab − d² exactly
 *  in U64 lanes, cheaper than F32 TwoProduct behind a range guard or than F64 lanes. */
NUMKONG_INLINE void nk_angular_through_i32_from_dot_neon_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                          nk_b128_vec_t const *target_sumsqs_vec,
                                                          nk_b128_vec_t *result_vec) {
    int32x4_t const dots_i32x4 = dots_vec->i32x4;
    uint32x4_t const target_sumsqs_u32x4 = target_sumsqs_vec->u32x4;
    uint64x2_t const products_ab_u64x2 = vmull_n_u32(vget_low_u32(target_sumsqs_u32x4), query_sumsq);
    uint64x2_t const products_cd_u64x2 = vmull_high_n_u32(target_sumsqs_u32x4, query_sumsq);
    // ab − d² wraps through I64 lanes into the exact U64 gap
    int64x2_t const gaps_ab_i64x2 = vmlsl_s32(vreinterpretq_s64_u64(products_ab_u64x2), vget_low_s32(dots_i32x4),
                                              vget_low_s32(dots_i32x4));
    int64x2_t const gaps_cd_i64x2 = vmlsl_high_s32(vreinterpretq_s64_u64(products_cd_u64x2), dots_i32x4, dots_i32x4);
    float32x4_t const gaps_f32x4 = vcvt_high_f32_f64(vcvt_f32_f64(vcvtq_f64_u64(vreinterpretq_u64_s64(gaps_ab_i64x2))),
                                                     vcvtq_f64_u64(vreinterpretq_u64_s64(gaps_cd_i64x2)));
    float32x4_t const products_f32x4 = vcvt_high_f32_f64(vcvt_f32_f64(vcvtq_f64_u64(products_ab_u64x2)),
                                                         vcvtq_f64_u64(products_cd_u64x2));
    float32x4_t const dots_f32x4 = vcvtq_f32_s32(dots_i32x4);
    float32x4_t const roots_f32x4 = vsqrtq_f32(products_f32x4);
    uint32x4_t const positive_u32x4 = vcgtzq_s32(dots_i32x4);
    float32x4_t const numerators_f32x4 = vbslq_f32(positive_u32x4, gaps_f32x4,
                                                   vfmsq_f32(products_f32x4, dots_f32x4, roots_f32x4));
    float32x4_t const denominators_f32x4 = vfmaq_f32(products_f32x4, vmaxq_f32(dots_f32x4, vdupq_n_f32(0)),
                                                     roots_f32x4);
    // A zero norm makes 0 / 0, and FMAXNM turns that NaN into 1 for one zero norm and 0 for two
    float32x4_t const zero_norms_f32x4 = vbslq_f32(vceqzq_u32(target_sumsqs_u32x4), vdupq_n_f32(query_sumsq != 0),
                                                   vdupq_n_f32(query_sumsq == 0));
    result_vec->f32x4 = vmaxnmq_f32(vdivq_f32(numerators_f32x4, denominators_f32x4), zero_norms_f32x4);
}

/** Euclidean from_dot for i32 dots d against u32 norms a, b: a + b − 2d is the exact
 *  squared distance in I64 lanes, below 2³⁵ and so exact in F64, and rounds once into F32
 *  for the root. */
NUMKONG_INLINE void nk_euclidean_through_i32_from_dot_neon_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                            nk_b128_vec_t const *target_sumsqs_vec,
                                                            nk_b128_vec_t *result_vec) {
    int32x4_t const dots_i32x4 = dots_vec->i32x4;
    uint32x4_t const target_sumsqs_u32x4 = target_sumsqs_vec->u32x4;
    uint64x2_t const query_sumsq_u64x2 = vdupq_n_u64(query_sumsq);
    int64x2_t const distances_sq_ab_i64x2 = vmlsl_n_s32(
        vreinterpretq_s64_u64(vaddw_u32(query_sumsq_u64x2, vget_low_u32(target_sumsqs_u32x4))),
        vget_low_s32(dots_i32x4), 2);
    int64x2_t const distances_sq_cd_i64x2 = vmlsl_high_n_s32(
        vreinterpretq_s64_u64(vaddw_high_u32(query_sumsq_u64x2, target_sumsqs_u32x4)), dots_i32x4, 2);
    result_vec->f32x4 = vsqrtq_f32(
        vcvt_high_f32_f64(vcvt_f32_f64(vcvtq_f64_s64(distances_sq_ab_i64x2)), vcvtq_f64_s64(distances_sq_cd_i64x2)));
}

/** Angular from_dot for u32 dots d and norms a, b: (ab − d²) / (ab + d · √(ab)) as in the i32
 *  variant, where an unsigned dot is never negative and a zero dot gives exactly 1. */
NUMKONG_INLINE void nk_angular_through_u32_from_dot_neon_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                          nk_b128_vec_t const *target_sumsqs_vec,
                                                          nk_b128_vec_t *result_vec) {
    uint32x4_t const dots_u32x4 = dots_vec->u32x4, target_sumsqs_u32x4 = target_sumsqs_vec->u32x4;
    uint64x2_t const products_ab_u64x2 = vmull_n_u32(vget_low_u32(target_sumsqs_u32x4), query_sumsq);
    uint64x2_t const products_cd_u64x2 = vmull_high_n_u32(target_sumsqs_u32x4, query_sumsq);
    uint64x2_t const gaps_ab_u64x2 = vmlsl_u32(products_ab_u64x2, vget_low_u32(dots_u32x4), vget_low_u32(dots_u32x4));
    uint64x2_t const gaps_cd_u64x2 = vmlsl_high_u32(products_cd_u64x2, dots_u32x4, dots_u32x4);
    float32x4_t const gaps_f32x4 = vcvt_high_f32_f64(vcvt_f32_f64(vcvtq_f64_u64(gaps_ab_u64x2)),
                                                     vcvtq_f64_u64(gaps_cd_u64x2));
    float32x4_t const products_f32x4 = vcvt_high_f32_f64(vcvt_f32_f64(vcvtq_f64_u64(products_ab_u64x2)),
                                                         vcvtq_f64_u64(products_cd_u64x2));
    float32x4_t const denominators_f32x4 = vfmaq_f32(products_f32x4, vcvtq_f32_u32(dots_u32x4),
                                                     vsqrtq_f32(products_f32x4));
    // A zero norm makes 0 / 0, and FMAXNM turns that NaN into 1 for one zero norm and 0 for two
    float32x4_t const zero_norms_f32x4 = vbslq_f32(vceqzq_u32(target_sumsqs_u32x4), vdupq_n_f32(query_sumsq != 0),
                                                   vdupq_n_f32(query_sumsq == 0));
    result_vec->f32x4 = vmaxnmq_f32(vdivq_f32(gaps_f32x4, denominators_f32x4), zero_norms_f32x4);
}

/** Euclidean from_dot for u32 dots d and norms a, b: a + b − 2d is exact in U64 lanes. */
NUMKONG_INLINE void nk_euclidean_through_u32_from_dot_neon_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                            nk_b128_vec_t const *target_sumsqs_vec,
                                                            nk_b128_vec_t *result_vec) {
    uint32x4_t const dots_u32x4 = dots_vec->u32x4, target_sumsqs_u32x4 = target_sumsqs_vec->u32x4;
    uint64x2_t const query_sumsq_u64x2 = vdupq_n_u64(query_sumsq);
    uint64x2_t const distances_sq_ab_u64x2 = vmlsl_n_u32(
        vaddw_u32(query_sumsq_u64x2, vget_low_u32(target_sumsqs_u32x4)), vget_low_u32(dots_u32x4), 2);
    uint64x2_t const distances_sq_cd_u64x2 = vmlsl_high_n_u32(vaddw_high_u32(query_sumsq_u64x2, target_sumsqs_u32x4),
                                                              dots_u32x4, 2);
    result_vec->f32x4 = vsqrtq_f32(
        vcvt_high_f32_f64(vcvt_f32_f64(vcvtq_f64_u64(distances_sq_ab_u64x2)), vcvtq_f64_u64(distances_sq_cd_u64x2)));
}

#pragma endregion F16 and BF16 Floats

#if NUMKONG_TARGET_NEON

#pragma region F32 and F64 Floats

NUMKONG_API nk_status_t nk_sqeuclidean_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f32_neon_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                              void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f32_neon_(a, b, n, result);
    *result = vget_lane_f64(vsqrt_f64(vdup_n_f64(*result)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                            void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
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
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f64_neon_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                              void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f64_neon_(a, b, n, result);
    *result = vget_lane_f64(vsqrt_f64(vdup_n_f64(*result)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                            void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    float64x2_t ab_sum_f64x2 = vdupq_n_f64(0), ab_compensation_f64x2 = vdupq_n_f64(0);
    float64x2_t a2_sum_f64x2 = vdupq_n_f64(0), a2_compensation_f64x2 = vdupq_n_f64(0);
    float64x2_t b2_sum_f64x2 = vdupq_n_f64(0), b2_compensation_f64x2 = vdupq_n_f64(0);
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
    nk_dot2_f64x2_neon_(&ab_sum_f64x2, &ab_compensation_f64x2, a_f64x2, b_f64x2);
    nk_dot2_f64x2_neon_(&a2_sum_f64x2, &a2_compensation_f64x2, a_f64x2, a_f64x2);
    nk_dot2_f64x2_neon_(&b2_sum_f64x2, &b2_compensation_f64x2, b_f64x2, b_f64x2);
    if (n) goto nk_angular_f64_neon_cycle;

    *result = nk_angular_normalize_f64_neon_( //
        nk_dot_stable_sum_f64x2_neon_(ab_sum_f64x2, ab_compensation_f64x2),
        nk_dot_stable_sum_f64x2_neon_(a2_sum_f64x2, a2_compensation_f64x2),
        nk_dot_stable_sum_f64x2_neon_(b2_sum_f64x2, b2_compensation_f64x2));
    return nk_success_k;
}

#pragma endregion F32 and F64 Floats
#pragma region F16 and BF16 Floats

NUMKONG_API nk_status_t nk_sqeuclidean_bf16_neon(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_bf16_neon_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_bf16_neon(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                               void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_bf16_neon_(a, b, n, result);
    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32(*result)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_bf16_neon(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                             void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
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
    float32x4_t a_high_f32x4 = vreinterpretq_f32_u32(vshll_high_n_u16(a_u16x8, 16));
    float32x4_t b_low_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b_u16x8), 16));
    float32x4_t b_high_f32x4 = vreinterpretq_f32_u32(vshll_high_n_u16(b_u16x8, 16));
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
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f16_neon_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                              void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f16_neon_(a, b, n, result);
    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32(*result)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                            void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    uint16x8_t a_u16x8, b_u16x8;
    float32x4_t ab_f32x4 = vdupq_n_f32(0);
    float32x4_t a2_f32x4 = vdupq_n_f32(0);
    float32x4_t b2_f32x4 = vdupq_n_f32(0);
nk_angular_f16_neon_cycle:
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
    float16x8_t a_f16x8 = vreinterpretq_f16_u16(a_u16x8);
    float16x8_t b_f16x8 = vreinterpretq_f16_u16(b_u16x8);
    float32x4_t a_low_f32x4 = vcvt_f32_f16(vget_low_f16(a_f16x8));
    float32x4_t a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
    ab_f32x4 = vfmaq_f32(ab_f32x4, a_low_f32x4, b_low_f32x4);
    ab_f32x4 = vfmaq_f32(ab_f32x4, a_high_f32x4, b_high_f32x4);
    a2_f32x4 = vfmaq_f32(a2_f32x4, a_low_f32x4, a_low_f32x4);
    a2_f32x4 = vfmaq_f32(a2_f32x4, a_high_f32x4, a_high_f32x4);
    b2_f32x4 = vfmaq_f32(b2_f32x4, b_low_f32x4, b_low_f32x4);
    b2_f32x4 = vfmaq_f32(b2_f32x4, b_high_f32x4, b_high_f32x4);
    if (n) goto nk_angular_f16_neon_cycle;
    nk_f32_t ab = vaddvq_f32(ab_f32x4);
    nk_f32_t a2 = vaddvq_f32(a2_f32x4);
    nk_f32_t b2 = vaddvq_f32(b2_f32x4);
    *result = nk_angular_normalize_f32_neon_(ab, a2, b2);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_e2m3_neon(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_e2m3_neon_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_e2m3_neon(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                               void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_e2m3_neon_(a, b, n, result);
    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32(*result)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_e2m3_neon(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result,
                                             void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
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
    float32x4_t a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
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
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_e3m2_neon(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_e3m2_neon_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_e3m2_neon(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                               void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_e3m2_neon_(a, b, n, result);
    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32(*result)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_e3m2_neon(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result,
                                             void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
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
    float32x4_t a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
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
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                                 void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_e4m3_neon_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                               void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_e4m3_neon_(a, b, n, result);
    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32(*result)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_e4m3_neon(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result,
                                             void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
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
    float32x4_t a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
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
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                                 void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_e5m2_neon_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                               void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_e5m2_neon_(a, b, n, result);
    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32(*result)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_e5m2_neon(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result,
                                             void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
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
    float32x4_t a_high_f32x4 = vcvt_high_f32_f16(a_f16x8);
    float32x4_t b_low_f32x4 = vcvt_f32_f16(vget_low_f16(b_f16x8));
    float32x4_t b_high_f32x4 = vcvt_high_f32_f16(b_f16x8);
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
    return nk_success_k;
}

#pragma endregion F16 and BF16 Floats

#endif // NUMKONG_TARGET_NEON

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ARM64_NEON_
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_SPATIAL_NEON_H
