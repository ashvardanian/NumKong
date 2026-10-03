/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for RISC-V.
 *  @file include/numkong/spatial/rvv.h
 *  @author Ash Vardanian
 *  @date January 5, 2026
 *
 *  @sa include/numkong/spatial.h
 *
 *  RVV uses vector length agnostic programming where:
 *  - `vsetvl_e*m*(n)` sets VL = min(n, VLMAX) and returns actual VL
 *  - Loads/stores with VL automatically handle partial vectors (tail elements)
 *  - No explicit masking needed for simple reductions
 *
 *  This file contains base RVV 1.0 operations (i8, u8, f32, f64).
 *  For f16 (Zvfh) see rvvhalf.h, for bf16 (Zvfbfwma) see rvvbf16.h.
 *
 *  Precision strategies matching Skylake:
 *  - i8 L2: diff (i8-i8 → i16), square (i16 × i16 → i32), reduce to i32
 *  - u8 L2: |diff| via widening, square → u32, reduce to u32
 *  - f32: Widen to f64 for accumulation, downcast result to f32
 *  - f64: Direct f64 accumulation
 */
#ifndef NK_SPATIAL_RVV_H
#define NK_SPATIAL_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/scalar/rvv.h" // `nk_f32_rsqrt_rvv`
#include "numkong/cast/rvv.h"   // `nk_e4m3m1_to_f32m4_rvv_`
#include "numkong/dot/rvv.h"    // `nk_dot_stable_sum_f64m1_rvv_`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief  Vectorized `1/√x` for f32 m1 register group using `vfrsqrt7` + 2 Newton-Raphson steps.
 *
 *  Achieves ~28 bits of precision, sufficient for f32's 23-bit mantissa.
 *  Formula per iteration: y' = y × (3 − x × y²) × 0.5
 */
NK_INTERNAL vfloat32m1_t nk_rsqrt_f32m1_rvv_(vfloat32m1_t values_f32m1, size_t vector_length) {
    vfloat32m1_t rsqrt_f32m1 = __riscv_vfrsqrt7_v_f32m1(values_f32m1, vector_length);
    for (int step = 0; step < 2; step++) {
        vfloat32m1_t rsqrt_sq_f32m1 = __riscv_vfmul_vv_f32m1(rsqrt_f32m1, rsqrt_f32m1, vector_length);
        vfloat32m1_t residual_f32m1 = __riscv_vfrsub_vf_f32m1(
            __riscv_vfmul_vv_f32m1(values_f32m1, rsqrt_sq_f32m1, vector_length), 3.0f, vector_length);
        rsqrt_f32m1 = __riscv_vfmul_vf_f32m1(__riscv_vfmul_vv_f32m1(rsqrt_f32m1, residual_f32m1, vector_length), 0.5f,
                                             vector_length);
    }
    return rsqrt_f32m1;
}

/**
 *  @brief  Vectorized `1/√x` for f64 m1 register group using `vfrsqrt7` + 3 Newton-Raphson steps.
 *
 *  Achieves ~56 bits of precision, sufficient for f64's 52-bit mantissa.
 *  Formula per iteration: y' = y × (3 − x × y²) × 0.5
 */
NK_INTERNAL vfloat64m1_t nk_rsqrt_f64m1_rvv_(vfloat64m1_t values_f64m1, size_t vector_length) {
    vfloat64m1_t rsqrt_f64m1 = __riscv_vfrsqrt7_v_f64m1(values_f64m1, vector_length);
    for (int step = 0; step < 3; step++) {
        vfloat64m1_t rsqrt_sq_f64m1 = __riscv_vfmul_vv_f64m1(rsqrt_f64m1, rsqrt_f64m1, vector_length);
        vfloat64m1_t residual_f64m1 = __riscv_vfrsub_vf_f64m1(
            __riscv_vfmul_vv_f64m1(values_f64m1, rsqrt_sq_f64m1, vector_length), 3.0, vector_length);
        rsqrt_f64m1 = __riscv_vfmul_vf_f64m1(__riscv_vfmul_vv_f64m1(rsqrt_f64m1, residual_f64m1, vector_length), 0.5,
                                             vector_length);
    }
    return rsqrt_f64m1;
}

/**
 *  @brief Approximate reciprocal of f32 vector (m4) using vfrec7 + 2 Newton-Raphson steps.
 *  Achieves ~28-bit precision, sufficient for f32 (24-bit mantissa).
 */
NK_INTERNAL vfloat32m4_t nk_f32m4_reciprocal_rvv_(vfloat32m4_t x_f32m4, nk_size_t vector_length) {
    vfloat32m4_t est_f32m4 = __riscv_vfrec7_v_f32m4(x_f32m4, vector_length);
    vfloat32m4_t two_f32m4 = __riscv_vfmv_v_f_f32m4(2.0f, vector_length);
    // NR step 1: est = est * (2 - x * est)
    est_f32m4 = __riscv_vfmul_vv_f32m4(
        est_f32m4, __riscv_vfnmsac_vv_f32m4(two_f32m4, x_f32m4, est_f32m4, vector_length), vector_length);
    // NR step 2: est = est * (2 - x * est)
    two_f32m4 = __riscv_vfmv_v_f_f32m4(2.0f, vector_length);
    est_f32m4 = __riscv_vfmul_vv_f32m4(
        est_f32m4, __riscv_vfnmsac_vv_f32m4(two_f32m4, x_f32m4, est_f32m4, vector_length), vector_length);
    return est_f32m4;
}

/**
 *  @brief Approximate reciprocal of f32 vector (m2) using vfrec7 + 2 Newton-Raphson steps.
 *  Achieves ~28-bit precision, sufficient for f32 (24-bit mantissa).
 */
NK_INTERNAL vfloat32m2_t nk_f32m2_reciprocal_rvv_(vfloat32m2_t x_f32m2, nk_size_t vector_length) {
    vfloat32m2_t est_f32m2 = __riscv_vfrec7_v_f32m2(x_f32m2, vector_length);
    vfloat32m2_t two_f32m2 = __riscv_vfmv_v_f_f32m2(2.0f, vector_length);
    // NR step 1: est = est * (2 - x * est)
    est_f32m2 = __riscv_vfmul_vv_f32m2(
        est_f32m2, __riscv_vfnmsac_vv_f32m2(two_f32m2, x_f32m2, est_f32m2, vector_length), vector_length);
    // NR step 2: est = est * (2 - x * est)
    two_f32m2 = __riscv_vfmv_v_f_f32m2(2.0f, vector_length);
    est_f32m2 = __riscv_vfmul_vv_f32m2(
        est_f32m2, __riscv_vfnmsac_vv_f32m2(two_f32m2, x_f32m2, est_f32m2, vector_length), vector_length);
    return est_f32m2;
}

/**
 *  @brief Approximate reciprocal of f64 vector (m4) using vfrec7 + 3 Newton-Raphson steps.
 *  Achieves ~56-bit precision, sufficient for f64 (52-bit mantissa).
 */
NK_INTERNAL vfloat64m4_t nk_f64m4_reciprocal_rvv_(vfloat64m4_t x_f64m4, nk_size_t vector_length) {
    vfloat64m4_t est_f64m4 = __riscv_vfrec7_v_f64m4(x_f64m4, vector_length);
    vfloat64m4_t two_f64m4 = __riscv_vfmv_v_f_f64m4(2.0, vector_length);
    // NR step 1
    est_f64m4 = __riscv_vfmul_vv_f64m4(
        est_f64m4, __riscv_vfnmsac_vv_f64m4(two_f64m4, x_f64m4, est_f64m4, vector_length), vector_length);
    // NR step 2
    two_f64m4 = __riscv_vfmv_v_f_f64m4(2.0, vector_length);
    est_f64m4 = __riscv_vfmul_vv_f64m4(
        est_f64m4, __riscv_vfnmsac_vv_f64m4(two_f64m4, x_f64m4, est_f64m4, vector_length), vector_length);
    // NR step 3
    two_f64m4 = __riscv_vfmv_v_f_f64m4(2.0, vector_length);
    est_f64m4 = __riscv_vfmul_vv_f64m4(
        est_f64m4, __riscv_vfnmsac_vv_f64m4(two_f64m4, x_f64m4, est_f64m4, vector_length), vector_length);
    return est_f64m4;
}

#pragma region - Small Integers

NK_PUBLIC void nk_sqeuclidean_i8_rvv(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                     nk_u32_t *result) {
    vint32m1_t sum_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vint8m1_t a_i8m1 = __riscv_vle8_v_i8m1(a_scalars, vector_length);
        vint8m1_t b_i8m1 = __riscv_vle8_v_i8m1(b_scalars, vector_length);
        // Widening subtract: i8 - i8 → i16
        vint16m2_t diff_i16m2 = __riscv_vwsub_vv_i16m2(a_i8m1, b_i8m1, vector_length);
        // Widening square: i16 × i16 → i32
        vint32m4_t sq_i32m4 = __riscv_vwmul_vv_i32m4(diff_i16m2, diff_i16m2, vector_length);
        // Reduce to scalar
        sum_i32m1 = __riscv_vredsum_vs_i32m4_i32m1(sq_i32m4, sum_i32m1, vector_length);
    }
    *result = (nk_u32_t)__riscv_vmv_x_s_i32m1_i32(sum_i32m1);
}

NK_PUBLIC void nk_euclidean_i8_rvv(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_u32_t d2;
    nk_sqeuclidean_i8_rvv(a_scalars, b_scalars, count_scalars, &d2);
    *result = nk_f32_sqrt_rvv((nk_f32_t)d2);
}

NK_PUBLIC void nk_sqeuclidean_u8_rvv(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                     nk_u32_t *result) {
    vuint32m1_t sum_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1(a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1(b_scalars, vector_length);
        // Compute |a - b| using saturating subtraction: max(a-b, b-a) = (a -sat b) | (b -sat a)
        vuint8m1_t diff_ab_u8m1 = __riscv_vssubu_vv_u8m1(a_u8m1, b_u8m1, vector_length);
        vuint8m1_t diff_ba_u8m1 = __riscv_vssubu_vv_u8m1(b_u8m1, a_u8m1, vector_length);
        vuint8m1_t abs_diff_u8m1 = __riscv_vor_vv_u8m1(diff_ab_u8m1, diff_ba_u8m1, vector_length);
        // Widening multiply: u8 × u8 → u16
        vuint16m2_t sq_u16m2 = __riscv_vwmulu_vv_u16m2(abs_diff_u8m1, abs_diff_u8m1, vector_length);
        // Widening reduce: u16 → u32
        sum_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(sq_u16m2, sum_u32m1, vector_length);
    }
    *result = __riscv_vmv_x_s_u32m1_u32(sum_u32m1);
}

NK_PUBLIC void nk_euclidean_u8_rvv(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_u32_t d2;
    nk_sqeuclidean_u8_rvv(a_scalars, b_scalars, count_scalars, &d2);
    *result = nk_f32_sqrt_rvv((nk_f32_t)d2);
}

#pragma endregion - Small Integers
#pragma region - Traditional Floats

NK_PUBLIC void nk_sqeuclidean_f32_rvv(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                      nk_f64_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count_scalars);
        vfloat32m1_t a_f32m1 = __riscv_vle32_v_f32m1(a_scalars, vector_length);
        vfloat32m1_t b_f32m1 = __riscv_vle32_v_f32m1(b_scalars, vector_length);
        vfloat64m2_t a_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(a_f32m1, vector_length);
        vfloat64m2_t b_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(b_f32m1, vector_length);
        vfloat64m2_t diff_f64m2 = __riscv_vfsub_vv_f64m2(a_f64m2, b_f64m2, vector_length);
        sum_f64m2 = __riscv_vfmacc_vv_f64m2_tu(sum_f64m2, diff_f64m2, diff_f64m2, vector_length);
    }
    // Single horizontal reduction at the end
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *result = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_f64m2, zero_f64m1, vlmax));
}

NK_PUBLIC void nk_euclidean_f32_rvv(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                    nk_f64_t *result) {
    nk_sqeuclidean_f32_rvv(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f64_sqrt_rvv(*result);
}

NK_PUBLIC void nk_sqeuclidean_f64_rvv(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                      nk_f64_t *result) {
    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vector_length_max);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(count_scalars);
        vfloat64m1_t a_f64m1 = __riscv_vle64_v_f64m1(a_scalars, vector_length);
        vfloat64m1_t b_f64m1 = __riscv_vle64_v_f64m1(b_scalars, vector_length);
        // Compute difference and accumulate diff² into vector lanes
        vfloat64m1_t diff_f64m1 = __riscv_vfsub_vv_f64m1(a_f64m1, b_f64m1, vector_length);
        sum_f64m1 = __riscv_vfmacc_vv_f64m1_tu(sum_f64m1, diff_f64m1, diff_f64m1, vector_length);
    }
    // Single horizontal reduction at the end
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vector_length_max);
    *result = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(sum_f64m1, zero_f64m1, vector_length_max));
}

NK_PUBLIC void nk_euclidean_f64_rvv(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                    nk_f64_t *result) {
    nk_sqeuclidean_f64_rvv(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f64_sqrt_rvv(*result);
}

#pragma endregion - Traditional Floats
#pragma region - Small Integers

NK_PUBLIC void nk_angular_i8_rvv(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_f32_t *result) {
    vint32m1_t dot_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t a_norm_sq_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t b_norm_sq_i32m1 = __riscv_vmv_v_x_i32m1(0, 1);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vint8m1_t a_i8m1 = __riscv_vle8_v_i8m1(a_scalars, vector_length);
        vint8m1_t b_i8m1 = __riscv_vle8_v_i8m1(b_scalars, vector_length);

        // dot += a × b (widened to i32)
        vint16m2_t ab_i16m2 = __riscv_vwmul_vv_i16m2(a_i8m1, b_i8m1, vector_length);
        dot_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(ab_i16m2, dot_i32m1, vector_length);

        // a_norm_sq += a × a
        vint16m2_t aa_i16m2 = __riscv_vwmul_vv_i16m2(a_i8m1, a_i8m1, vector_length);
        a_norm_sq_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(aa_i16m2, a_norm_sq_i32m1, vector_length);

        // b_norm_sq += b × b
        vint16m2_t bb_i16m2 = __riscv_vwmul_vv_i16m2(b_i8m1, b_i8m1, vector_length);
        b_norm_sq_i32m1 = __riscv_vwredsum_vs_i16m2_i32m1(bb_i16m2, b_norm_sq_i32m1, vector_length);
    }

    nk_i32_t dot_i32 = __riscv_vmv_x_s_i32m1_i32(dot_i32m1);
    nk_i32_t a_norm_sq_i32 = __riscv_vmv_x_s_i32m1_i32(a_norm_sq_i32m1);
    nk_i32_t b_norm_sq_i32 = __riscv_vmv_x_s_i32m1_i32(b_norm_sq_i32m1);

    // Normalize: 1 − dot / √(‖a‖² × ‖b‖²)
    if (a_norm_sq_i32 == 0 && b_norm_sq_i32 == 0) { *result = 0.0f; }
    else if (dot_i32 == 0) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - (nk_f32_t)dot_i32 * nk_f32_rsqrt_rvv((nk_f32_t)a_norm_sq_i32) *
                                        nk_f32_rsqrt_rvv((nk_f32_t)b_norm_sq_i32);
        *result = unclipped > 0 ? unclipped : 0;
    }
}

NK_PUBLIC void nk_angular_u8_rvv(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_f32_t *result) {
    vuint32m1_t dot_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    vuint32m1_t a_norm_sq_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    vuint32m1_t b_norm_sq_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1(a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1(b_scalars, vector_length);

        // dot += a × b (widened to u32)
        vuint16m2_t ab_u16m2 = __riscv_vwmulu_vv_u16m2(a_u8m1, b_u8m1, vector_length);
        dot_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(ab_u16m2, dot_u32m1, vector_length);

        // a_norm_sq += a × a
        vuint16m2_t aa_u16m2 = __riscv_vwmulu_vv_u16m2(a_u8m1, a_u8m1, vector_length);
        a_norm_sq_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(aa_u16m2, a_norm_sq_u32m1, vector_length);

        // b_norm_sq += b × b
        vuint16m2_t bb_u16m2 = __riscv_vwmulu_vv_u16m2(b_u8m1, b_u8m1, vector_length);
        b_norm_sq_u32m1 = __riscv_vwredsumu_vs_u16m2_u32m1(bb_u16m2, b_norm_sq_u32m1, vector_length);
    }

    nk_u32_t dot_u32 = __riscv_vmv_x_s_u32m1_u32(dot_u32m1);
    nk_u32_t a_norm_sq_u32 = __riscv_vmv_x_s_u32m1_u32(a_norm_sq_u32m1);
    nk_u32_t b_norm_sq_u32 = __riscv_vmv_x_s_u32m1_u32(b_norm_sq_u32m1);

    // Normalize: 1 − dot / √(‖a‖² × ‖b‖²)
    if (a_norm_sq_u32 == 0 && b_norm_sq_u32 == 0) { *result = 0.0f; }
    else if (dot_u32 == 0) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - (nk_f32_t)dot_u32 * nk_f32_rsqrt_rvv((nk_f32_t)a_norm_sq_u32) *
                                        nk_f32_rsqrt_rvv((nk_f32_t)b_norm_sq_u32);
        *result = unclipped > 0 ? unclipped : 0;
    }
}

#pragma endregion - Small Integers
#pragma region - Traditional Floats

NK_PUBLIC void nk_angular_f32_rvv(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f64_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t dot_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t a_norm_sq_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t b_norm_sq_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count_scalars);
        vfloat32m1_t a_f32m1 = __riscv_vle32_v_f32m1(a_scalars, vector_length);
        vfloat32m1_t b_f32m1 = __riscv_vle32_v_f32m1(b_scalars, vector_length);

        // Widening multiply-accumulate into f64 vector lanes
        dot_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(dot_f64m2, a_f32m1, b_f32m1, vector_length);
        a_norm_sq_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(a_norm_sq_f64m2, a_f32m1, a_f32m1, vector_length);
        b_norm_sq_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(b_norm_sq_f64m2, b_f32m1, b_f32m1, vector_length);
    }

    // Single horizontal reduction at the end for all three accumulators
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f64_t dot_f64 = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(dot_f64m2, zero_f64m1, vlmax));
    nk_f64_t a_norm_sq_f64 = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m2_f64m1(a_norm_sq_f64m2, zero_f64m1, vlmax));
    nk_f64_t b_norm_sq_f64 = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m2_f64m1(b_norm_sq_f64m2, zero_f64m1, vlmax));

    // Normalize: 1 − dot / √(‖a‖² × ‖b‖²)
    if (a_norm_sq_f64 == 0.0 && b_norm_sq_f64 == 0.0) { *result = 0.0; }
    else if (dot_f64 == 0.0) { *result = 1.0; }
    else {
        nk_f64_t unclipped = 1.0 - dot_f64 * nk_f64_rsqrt_rvv(a_norm_sq_f64) * nk_f64_rsqrt_rvv(b_norm_sq_f64);
        *result = unclipped > 0 ? unclipped : 0.0;
    }
}

NK_PUBLIC void nk_angular_f64_rvv(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f64_t *result) {
    // Dot2 (Ogita-Rump-Oishi) for cross-product (may have cancellation),
    // simple FMA for self-products a²/b² (all positive, no cancellation)
    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t dot_sum_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vector_length_max);
    vfloat64m1_t dot_compensation_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vector_length_max);
    vfloat64m1_t a_norm_sq_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vector_length_max);
    vfloat64m1_t b_norm_sq_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vector_length_max);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(count_scalars);
        vfloat64m1_t a_f64m1 = __riscv_vle64_v_f64m1(a_scalars, vector_length);
        vfloat64m1_t b_f64m1 = __riscv_vle64_v_f64m1(b_scalars, vector_length);

        // TwoProd: product = a*b, product_error = fma(a,b,-product)
        vfloat64m1_t product_f64m1 = __riscv_vfmul_vv_f64m1(a_f64m1, b_f64m1, vector_length);
        vfloat64m1_t product_error_f64m1 = __riscv_vfmsac_vv_f64m1(product_f64m1, a_f64m1, b_f64m1, vector_length);
        // TwoSum: tentative_sum = sum + product
        vfloat64m1_t tentative_sum_f64m1 = __riscv_vfadd_vv_f64m1(dot_sum_f64m1, product_f64m1, vector_length);
        vfloat64m1_t virtual_addend_f64m1 = __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, dot_sum_f64m1, vector_length);
        vfloat64m1_t sum_error_f64m1 = __riscv_vfadd_vv_f64m1(
            __riscv_vfsub_vv_f64m1(dot_sum_f64m1,
                                   __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, virtual_addend_f64m1, vector_length),
                                   vector_length),
            __riscv_vfsub_vv_f64m1(product_f64m1, virtual_addend_f64m1, vector_length), vector_length);
        // Tail-undisturbed updates: preserve zero tails across partial iterations
        dot_sum_f64m1 = __riscv_vslideup_vx_f64m1_tu(dot_sum_f64m1, tentative_sum_f64m1, 0, vector_length);
        vfloat64m1_t total_error_f64m1 = __riscv_vfadd_vv_f64m1(sum_error_f64m1, product_error_f64m1, vector_length);
        dot_compensation_f64m1 = __riscv_vfadd_vv_f64m1_tu(dot_compensation_f64m1, dot_compensation_f64m1,
                                                           total_error_f64m1, vector_length);
        // Simple FMA for self-products (no cancellation possible)
        a_norm_sq_f64m1 = __riscv_vfmacc_vv_f64m1_tu(a_norm_sq_f64m1, a_f64m1, a_f64m1, vector_length);
        b_norm_sq_f64m1 = __riscv_vfmacc_vv_f64m1_tu(b_norm_sq_f64m1, b_f64m1, b_f64m1, vector_length);
    }

    // Compensated horizontal reduction for cross-product, simple reduction for self-products
    nk_f64_t dot_f64 = nk_dot_stable_sum_f64m1_rvv_(dot_sum_f64m1, dot_compensation_f64m1);
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vector_length_max);
    nk_f64_t a_norm_sq_f64 = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m1_f64m1(a_norm_sq_f64m1, zero_f64m1, vector_length_max));
    nk_f64_t b_norm_sq_f64 = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m1_f64m1(b_norm_sq_f64m1, zero_f64m1, vector_length_max));

    // Normalize: 1 − dot / √(‖a‖² × ‖b‖²)
    if (a_norm_sq_f64 == 0.0 && b_norm_sq_f64 == 0.0) { *result = 0.0; }
    else if (dot_f64 == 0.0) { *result = 1.0; }
    else {
        nk_f64_t unclipped = 1.0 - dot_f64 * nk_f64_rsqrt_rvv(a_norm_sq_f64) * nk_f64_rsqrt_rvv(b_norm_sq_f64);
        *result = unclipped > 0 ? unclipped : 0;
    }
}

#pragma endregion - Traditional Floats
#pragma region - Smaller Floats

NK_PUBLIC void nk_sqeuclidean_f16_rvv(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                      nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);

        // Load f16 as u16 bits and convert to f32 via helper
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b_scalars, vector_length);
        vfloat32m2_t a_f32m2 = nk_f16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_f16m1_to_f32m2_rvv_(b_u16m1, vector_length);

        // Compute difference in f32, accumulate diff² into vector lanes
        vfloat32m2_t diff_f32m2 = __riscv_vfsub_vv_f32m2(a_f32m2, b_f32m2, vector_length);
        sum_f32m2 = __riscv_vfmacc_vv_f32m2_tu(sum_f32m2, diff_f32m2, diff_f32m2, vector_length);
    }
    // Single horizontal reduction at the end
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(sum_f32m2, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_euclidean_f16_rvv(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                    nk_f32_t *result) {
    nk_sqeuclidean_f16_rvv(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f32_sqrt_rvv(*result);
}

NK_PUBLIC void nk_angular_f16_rvv(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t dot_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t a_norm_sq_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t b_norm_sq_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);

        // Load f16 as u16 bits and convert to f32 via helper
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b_scalars, vector_length);
        vfloat32m2_t a_f32m2 = nk_f16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_f16m1_to_f32m2_rvv_(b_u16m1, vector_length);

        // Multiply-accumulate into f32 vector lanes
        dot_f32m2 = __riscv_vfmacc_vv_f32m2_tu(dot_f32m2, a_f32m2, b_f32m2, vector_length);
        a_norm_sq_f32m2 = __riscv_vfmacc_vv_f32m2_tu(a_norm_sq_f32m2, a_f32m2, a_f32m2, vector_length);
        b_norm_sq_f32m2 = __riscv_vfmacc_vv_f32m2_tu(b_norm_sq_f32m2, b_f32m2, b_f32m2, vector_length);
    }

    // Single horizontal reduction at the end for all three accumulators
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    nk_f32_t dot_f32 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(dot_f32m2, zero_f32m1, vlmax));
    nk_f32_t a_norm_sq_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(a_norm_sq_f32m2, zero_f32m1, vlmax));
    nk_f32_t b_norm_sq_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(b_norm_sq_f32m2, zero_f32m1, vlmax));

    if (a_norm_sq_f32 == 0.0f && b_norm_sq_f32 == 0.0f) { *result = 0.0f; }
    else if (dot_f32 == 0.0f) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - dot_f32 * nk_f32_rsqrt_rvv(a_norm_sq_f32) * nk_f32_rsqrt_rvv(b_norm_sq_f32);
        *result = unclipped > 0.0f ? unclipped : 0.0f;
    }
}

NK_PUBLIC void nk_sqeuclidean_bf16_rvv(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);

        // Load bf16 as u16 and convert to f32 via helper
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b_scalars, vector_length);
        vfloat32m2_t a_f32m2 = nk_bf16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_bf16m1_to_f32m2_rvv_(b_u16m1, vector_length);

        // Compute difference in f32, accumulate diff² into vector lanes
        vfloat32m2_t diff_f32m2 = __riscv_vfsub_vv_f32m2(a_f32m2, b_f32m2, vector_length);
        sum_f32m2 = __riscv_vfmacc_vv_f32m2_tu(sum_f32m2, diff_f32m2, diff_f32m2, vector_length);
    }
    // Single horizontal reduction at the end
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(sum_f32m2, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_euclidean_bf16_rvv(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                     nk_f32_t *result) {
    nk_sqeuclidean_bf16_rvv(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f32_sqrt_rvv(*result);
}

NK_PUBLIC void nk_angular_bf16_rvv(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t dot_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t a_norm_sq_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t b_norm_sq_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);

        // Load bf16 as u16 and convert to f32 via helper
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b_scalars, vector_length);
        vfloat32m2_t a_f32m2 = nk_bf16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_bf16m1_to_f32m2_rvv_(b_u16m1, vector_length);

        // Multiply-accumulate into f32 vector lanes
        dot_f32m2 = __riscv_vfmacc_vv_f32m2_tu(dot_f32m2, a_f32m2, b_f32m2, vector_length);
        a_norm_sq_f32m2 = __riscv_vfmacc_vv_f32m2_tu(a_norm_sq_f32m2, a_f32m2, a_f32m2, vector_length);
        b_norm_sq_f32m2 = __riscv_vfmacc_vv_f32m2_tu(b_norm_sq_f32m2, b_f32m2, b_f32m2, vector_length);
    }

    // Single horizontal reduction at the end for all three accumulators
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    nk_f32_t dot_f32 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(dot_f32m2, zero_f32m1, vlmax));
    nk_f32_t a_norm_sq_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(a_norm_sq_f32m2, zero_f32m1, vlmax));
    nk_f32_t b_norm_sq_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(b_norm_sq_f32m2, zero_f32m1, vlmax));

    if (a_norm_sq_f32 == 0.0f && b_norm_sq_f32 == 0.0f) { *result = 0.0f; }
    else if (dot_f32 == 0.0f) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - dot_f32 * nk_f32_rsqrt_rvv(a_norm_sq_f32) * nk_f32_rsqrt_rvv(b_norm_sq_f32);
        *result = unclipped > 0.0f ? unclipped : 0.0f;
    }
}

NK_PUBLIC void nk_sqeuclidean_e4m3_rvv(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);

        // Load e4m3 as u8 and convert to f32 via helper
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat32m4_t a_f32m4 = nk_e4m3m1_to_f32m4_rvv_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e4m3m1_to_f32m4_rvv_(b_u8m1, vector_length);

        // Compute difference in f32, accumulate diff² into vector lanes
        vfloat32m4_t diff_f32m4 = __riscv_vfsub_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        sum_f32m4 = __riscv_vfmacc_vv_f32m4_tu(sum_f32m4, diff_f32m4, diff_f32m4, vector_length);
    }
    // Single horizontal reduction at the end
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_euclidean_e4m3_rvv(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                     nk_f32_t *result) {
    nk_sqeuclidean_e4m3_rvv(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f32_sqrt_rvv(*result);
}

NK_PUBLIC void nk_angular_e4m3_rvv(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t dot_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t a_norm_sq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t b_norm_sq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);

        // Load e4m3 as u8 and convert to f32 via helper
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat32m4_t a_f32m4 = nk_e4m3m1_to_f32m4_rvv_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e4m3m1_to_f32m4_rvv_(b_u8m1, vector_length);

        // Multiply-accumulate into f32 vector lanes
        dot_f32m4 = __riscv_vfmacc_vv_f32m4_tu(dot_f32m4, a_f32m4, b_f32m4, vector_length);
        a_norm_sq_f32m4 = __riscv_vfmacc_vv_f32m4_tu(a_norm_sq_f32m4, a_f32m4, a_f32m4, vector_length);
        b_norm_sq_f32m4 = __riscv_vfmacc_vv_f32m4_tu(b_norm_sq_f32m4, b_f32m4, b_f32m4, vector_length);
    }

    // Single horizontal reduction at the end for all three accumulators
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    nk_f32_t dot_f32 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(dot_f32m4, zero_f32m1, vlmax));
    nk_f32_t a_norm_sq_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m4_f32m1(a_norm_sq_f32m4, zero_f32m1, vlmax));
    nk_f32_t b_norm_sq_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m4_f32m1(b_norm_sq_f32m4, zero_f32m1, vlmax));

    if (a_norm_sq_f32 == 0.0f && b_norm_sq_f32 == 0.0f) { *result = 0.0f; }
    else if (dot_f32 == 0.0f) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - dot_f32 * nk_f32_rsqrt_rvv(a_norm_sq_f32) * nk_f32_rsqrt_rvv(b_norm_sq_f32);
        *result = unclipped > 0.0f ? unclipped : 0.0f;
    }
}

NK_PUBLIC void nk_sqeuclidean_e5m2_rvv(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);

        // Load e5m2 as u8 and convert to f32 via helper
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat32m4_t a_f32m4 = nk_e5m2m1_to_f32m4_rvv_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e5m2m1_to_f32m4_rvv_(b_u8m1, vector_length);

        // Compute difference in f32, accumulate diff² into vector lanes
        vfloat32m4_t diff_f32m4 = __riscv_vfsub_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        sum_f32m4 = __riscv_vfmacc_vv_f32m4_tu(sum_f32m4, diff_f32m4, diff_f32m4, vector_length);
    }
    // Single horizontal reduction at the end
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_euclidean_e5m2_rvv(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                     nk_f32_t *result) {
    nk_sqeuclidean_e5m2_rvv(a_scalars, b_scalars, count_scalars, result);
    *result = nk_f32_sqrt_rvv(*result);
}

NK_PUBLIC void nk_angular_e5m2_rvv(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t dot_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t a_norm_sq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t b_norm_sq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);

        // Load e5m2 as u8 and convert to f32 via helper
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat32m4_t a_f32m4 = nk_e5m2m1_to_f32m4_rvv_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e5m2m1_to_f32m4_rvv_(b_u8m1, vector_length);

        // Multiply-accumulate into f32 vector lanes
        dot_f32m4 = __riscv_vfmacc_vv_f32m4_tu(dot_f32m4, a_f32m4, b_f32m4, vector_length);
        a_norm_sq_f32m4 = __riscv_vfmacc_vv_f32m4_tu(a_norm_sq_f32m4, a_f32m4, a_f32m4, vector_length);
        b_norm_sq_f32m4 = __riscv_vfmacc_vv_f32m4_tu(b_norm_sq_f32m4, b_f32m4, b_f32m4, vector_length);
    }

    // Single horizontal reduction at the end for all three accumulators
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    nk_f32_t dot_f32 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(dot_f32m4, zero_f32m1, vlmax));
    nk_f32_t a_norm_sq_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m4_f32m1(a_norm_sq_f32m4, zero_f32m1, vlmax));
    nk_f32_t b_norm_sq_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m4_f32m1(b_norm_sq_f32m4, zero_f32m1, vlmax));

    if (a_norm_sq_f32 == 0.0f && b_norm_sq_f32 == 0.0f) { *result = 0.0f; }
    else if (dot_f32 == 0.0f) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - dot_f32 * nk_f32_rsqrt_rvv(a_norm_sq_f32) * nk_f32_rsqrt_rvv(b_norm_sq_f32);
        *result = unclipped > 0.0f ? unclipped : 0.0f;
    }
}

#pragma endregion - Smaller Floats
#pragma region - Small Integers

NK_PUBLIC void nk_sqeuclidean_i4_rvv(nk_i4x2_t const *a_scalars, nk_i4x2_t const *b_scalars, nk_size_t count_scalars,
                                     nk_u32_t *result) {
    static nk_u8_t const nk_i4_sqd_lut_[256] = {
        0,  1,  4,   9,   16,  25,  36,  49,  64,  49,  36,  25,  16,  9,   4,  1,  //
        1,  0,  1,   4,   9,   16,  25,  36,  81,  64,  49,  36,  25,  16,  9,  4,  //
        4,  1,  0,   1,   4,   9,   16,  25,  100, 81,  64,  49,  36,  25,  16, 9,  //
        9,  4,  1,   0,   1,   4,   9,   16,  121, 100, 81,  64,  49,  36,  25, 16, //
        16, 9,  4,   1,   0,   1,   4,   9,   144, 121, 100, 81,  64,  49,  36, 25, //
        25, 16, 9,   4,   1,   0,   1,   4,   169, 144, 121, 100, 81,  64,  49, 36, //
        36, 25, 16,  9,   4,   1,   0,   1,   196, 169, 144, 121, 100, 81,  64, 49, //
        49, 36, 25,  16,  9,   4,   1,   0,   225, 196, 169, 144, 121, 100, 81, 64, //
        64, 81, 100, 121, 144, 169, 196, 225, 0,   1,   4,   9,   16,  25,  36, 49, //
        49, 64, 81,  100, 121, 144, 169, 196, 1,   0,   1,   4,   9,   16,  25, 36, //
        36, 49, 64,  81,  100, 121, 144, 169, 4,   1,   0,   1,   4,   9,   16, 25, //
        25, 36, 49,  64,  81,  100, 121, 144, 9,   4,   1,   0,   1,   4,   9,  16, //
        16, 25, 36,  49,  64,  81,  100, 121, 16,  9,   4,   1,   0,   1,   4,  9,  //
        9,  16, 25,  36,  49,  64,  81,  100, 25,  16,  9,   4,   1,   0,   1,  4,  //
        4,  9,  16,  25,  36,  49,  64,  81,  36,  25,  16,  9,   4,   1,   0,  1,  //
        1,  4,  9,   16,  25,  36,  49,  64,  49,  36,  25,  16,  9,   4,   1,  0,  //
    };
    count_scalars = nk_size_round_up_to_multiple_(count_scalars, 2);
    nk_size_t n_bytes = count_scalars / 2;
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vuint32m4_t sum_u32m4 = __riscv_vmv_v_x_u32m4(0, vlmax);
    for (nk_size_t vector_length; n_bytes > 0;
         n_bytes -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n_bytes);
        vuint8m1_t a_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        // Build LUT indices: high nibble pair = (a_hi << 4) | b_hi
        vuint8m1_t hi_idx_u8m1 = __riscv_vor_vv_u8m1(__riscv_vand_vx_u8m1(a_packed_u8m1, 0xF0, vector_length),
                                                     __riscv_vsrl_vx_u8m1(b_packed_u8m1, 4, vector_length),
                                                     vector_length);
        // Low nibble pair = (a_lo << 4) | b_lo
        vuint8m1_t lo_idx_u8m1 = __riscv_vor_vv_u8m1(
            __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(a_packed_u8m1, 0x0F, vector_length), 4, vector_length),
            __riscv_vand_vx_u8m1(b_packed_u8m1, 0x0F, vector_length), vector_length);
        // Gather squared differences from LUT (0-225, fits u8)
        vuint8m1_t sq_hi_u8m1 = __riscv_vluxei8_v_u8m1(nk_i4_sqd_lut_, hi_idx_u8m1, vector_length);
        vuint8m1_t sq_lo_u8m1 = __riscv_vluxei8_v_u8m1(nk_i4_sqd_lut_, lo_idx_u8m1, vector_length);
        // Combine and per-lane accumulate: u8+u8→u16, then u32+=u16
        vuint16m2_t combined_u16m2 = __riscv_vwaddu_vv_u16m2(sq_hi_u8m1, sq_lo_u8m1, vector_length);
        sum_u32m4 = __riscv_vwaddu_wv_u32m4_tu(sum_u32m4, sum_u32m4, combined_u16m2, vector_length);
    }
    // Single horizontal reduction after loop
    vuint32m1_t zero_u32m1 = __riscv_vmv_v_x_u32m1(0, vlmax);
    *result = __riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(sum_u32m4, zero_u32m1, vlmax));
}

NK_PUBLIC void nk_euclidean_i4_rvv(nk_i4x2_t const *a_scalars, nk_i4x2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_u32_t d2;
    nk_sqeuclidean_i4_rvv(a_scalars, b_scalars, count_scalars, &d2);
    *result = nk_f32_sqrt_rvv((nk_f32_t)d2);
}

NK_PUBLIC void nk_angular_i4_rvv(nk_i4x2_t const *a_scalars, nk_i4x2_t const *b_scalars, nk_size_t count_scalars,
                                 nk_f32_t *result) {
    static nk_i8_t const nk_i4_dot_lut_[256] = {
        0, 0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  //
        0, 1,  2,   3,   4,   5,   6,   7,   -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1, //
        0, 2,  4,   6,   8,   10,  12,  14,  -16, -14, -12, -10, -8,  -6,  -4,  -2, //
        0, 3,  6,   9,   12,  15,  18,  21,  -24, -21, -18, -15, -12, -9,  -6,  -3, //
        0, 4,  8,   12,  16,  20,  24,  28,  -32, -28, -24, -20, -16, -12, -8,  -4, //
        0, 5,  10,  15,  20,  25,  30,  35,  -40, -35, -30, -25, -20, -15, -10, -5, //
        0, 6,  12,  18,  24,  30,  36,  42,  -48, -42, -36, -30, -24, -18, -12, -6, //
        0, 7,  14,  21,  28,  35,  42,  49,  -56, -49, -42, -35, -28, -21, -14, -7, //
        0, -8, -16, -24, -32, -40, -48, -56, 64,  56,  48,  40,  32,  24,  16,  8,  //
        0, -7, -14, -21, -28, -35, -42, -49, 56,  49,  42,  35,  28,  21,  14,  7,  //
        0, -6, -12, -18, -24, -30, -36, -42, 48,  42,  36,  30,  24,  18,  12,  6,  //
        0, -5, -10, -15, -20, -25, -30, -35, 40,  35,  30,  25,  20,  15,  10,  5,  //
        0, -4, -8,  -12, -16, -20, -24, -28, 32,  28,  24,  20,  16,  12,  8,   4,  //
        0, -3, -6,  -9,  -12, -15, -18, -21, 24,  21,  18,  15,  12,  9,   6,   3,  //
        0, -2, -4,  -6,  -8,  -10, -12, -14, 16,  14,  12,  10,  8,   6,   4,   2,  //
        0, -1, -2,  -3,  -4,  -5,  -6,  -7,  8,   7,   6,   5,   4,   3,   2,   1,  //
    };
    static nk_u8_t const nk_i4_sq_lut_[16] = {0, 1, 4, 9, 16, 25, 36, 49, 64, 49, 36, 25, 16, 9, 4, 1};
    count_scalars = nk_size_round_up_to_multiple_(count_scalars, 2);
    nk_size_t n_bytes = count_scalars / 2;
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vint32m4_t dot_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
    vuint32m4_t a_norm_sq_u32m4 = __riscv_vmv_v_x_u32m4(0, vlmax);
    vuint32m4_t b_norm_sq_u32m4 = __riscv_vmv_v_x_u32m4(0, vlmax);

    for (nk_size_t vector_length; n_bytes > 0;
         n_bytes -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n_bytes);
        vuint8m1_t a_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);

        // Extract nibbles for index building
        vuint8m1_t a_hi_u8m1 = __riscv_vsrl_vx_u8m1(a_packed_u8m1, 4, vector_length);
        vuint8m1_t b_hi_u8m1 = __riscv_vsrl_vx_u8m1(b_packed_u8m1, 4, vector_length);
        vuint8m1_t a_lo_u8m1 = __riscv_vand_vx_u8m1(a_packed_u8m1, 0x0F, vector_length);
        vuint8m1_t b_lo_u8m1 = __riscv_vand_vx_u8m1(b_packed_u8m1, 0x0F, vector_length);

        // Dot product via 256-entry LUT: dot_lut[(a<<4)|b] = a_signed * b_signed (i8)
        vuint8m1_t hi_idx_u8m1 = __riscv_vor_vv_u8m1(__riscv_vand_vx_u8m1(a_packed_u8m1, 0xF0, vector_length),
                                                     b_hi_u8m1, vector_length);
        vuint8m1_t lo_idx_u8m1 = __riscv_vor_vv_u8m1(__riscv_vsll_vx_u8m1(a_lo_u8m1, 4, vector_length), b_lo_u8m1,
                                                     vector_length);
        vint8m1_t dot_hi_i8m1 = __riscv_vluxei8_v_i8m1(nk_i4_dot_lut_, hi_idx_u8m1, vector_length);
        vint8m1_t dot_lo_i8m1 = __riscv_vluxei8_v_i8m1(nk_i4_dot_lut_, lo_idx_u8m1, vector_length);
        // Widen i8→i16, add hi+lo, then per-lane accumulate i32+=i16
        vint16m2_t dot_combined_i16m2 = __riscv_vwadd_vv_i16m2(dot_hi_i8m1, dot_lo_i8m1, vector_length);
        dot_i32m4 = __riscv_vwadd_wv_i32m4_tu(dot_i32m4, dot_i32m4, dot_combined_i16m2, vector_length);

        // Norms via 16-entry squaring LUT + vluxei8
        vuint8m1_t a_hi_sq_u8m1 = __riscv_vluxei8_v_u8m1(nk_i4_sq_lut_, a_hi_u8m1, vector_length);
        vuint8m1_t a_lo_sq_u8m1 = __riscv_vluxei8_v_u8m1(nk_i4_sq_lut_, a_lo_u8m1, vector_length);
        vuint16m2_t a_sq_combined_u16m2 = __riscv_vwaddu_vv_u16m2(a_hi_sq_u8m1, a_lo_sq_u8m1, vector_length);
        a_norm_sq_u32m4 = __riscv_vwaddu_wv_u32m4_tu(a_norm_sq_u32m4, a_norm_sq_u32m4, a_sq_combined_u16m2,
                                                     vector_length);

        vuint8m1_t b_hi_sq_u8m1 = __riscv_vluxei8_v_u8m1(nk_i4_sq_lut_, b_hi_u8m1, vector_length);
        vuint8m1_t b_lo_sq_u8m1 = __riscv_vluxei8_v_u8m1(nk_i4_sq_lut_, b_lo_u8m1, vector_length);
        vuint16m2_t b_sq_combined_u16m2 = __riscv_vwaddu_vv_u16m2(b_hi_sq_u8m1, b_lo_sq_u8m1, vector_length);
        b_norm_sq_u32m4 = __riscv_vwaddu_wv_u32m4_tu(b_norm_sq_u32m4, b_norm_sq_u32m4, b_sq_combined_u16m2,
                                                     vector_length);
    }

    // Single horizontal reductions after loop
    vint32m1_t zero_i32m1 = __riscv_vmv_v_x_i32m1(0, vlmax);
    vuint32m1_t zero_u32m1 = __riscv_vmv_v_x_u32m1(0, vlmax);
    nk_i32_t dot_i32 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(dot_i32m4, zero_i32m1, vlmax));
    nk_u32_t a_norm_sq_u32 = __riscv_vmv_x_s_u32m1_u32(
        __riscv_vredsum_vs_u32m4_u32m1(a_norm_sq_u32m4, zero_u32m1, vlmax));
    nk_u32_t b_norm_sq_u32 = __riscv_vmv_x_s_u32m1_u32(
        __riscv_vredsum_vs_u32m4_u32m1(b_norm_sq_u32m4, zero_u32m1, vlmax));

    if (a_norm_sq_u32 == 0 && b_norm_sq_u32 == 0) { *result = 0.0f; }
    else if (dot_i32 == 0) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - (nk_f32_t)dot_i32 * nk_f32_rsqrt_rvv((nk_f32_t)a_norm_sq_u32) *
                                        nk_f32_rsqrt_rvv((nk_f32_t)b_norm_sq_u32);
        *result = unclipped > 0 ? unclipped : 0;
    }
}

NK_PUBLIC void nk_sqeuclidean_u4_rvv(nk_u4x2_t const *a_scalars, nk_u4x2_t const *b_scalars, nk_size_t count_scalars,
                                     nk_u32_t *result) {
    static nk_u8_t const nk_u4_sqd_lut_[256] = {
        0,   1,   4,   9,   16,  25,  36, 49, 64, 81, 100, 121, 144, 169, 196, 225, //
        1,   0,   1,   4,   9,   16,  25, 36, 49, 64, 81,  100, 121, 144, 169, 196, //
        4,   1,   0,   1,   4,   9,   16, 25, 36, 49, 64,  81,  100, 121, 144, 169, //
        9,   4,   1,   0,   1,   4,   9,  16, 25, 36, 49,  64,  81,  100, 121, 144, //
        16,  9,   4,   1,   0,   1,   4,  9,  16, 25, 36,  49,  64,  81,  100, 121, //
        25,  16,  9,   4,   1,   0,   1,  4,  9,  16, 25,  36,  49,  64,  81,  100, //
        36,  25,  16,  9,   4,   1,   0,  1,  4,  9,  16,  25,  36,  49,  64,  81,  //
        49,  36,  25,  16,  9,   4,   1,  0,  1,  4,  9,   16,  25,  36,  49,  64,  //
        64,  49,  36,  25,  16,  9,   4,  1,  0,  1,  4,   9,   16,  25,  36,  49,  //
        81,  64,  49,  36,  25,  16,  9,  4,  1,  0,  1,   4,   9,   16,  25,  36,  //
        100, 81,  64,  49,  36,  25,  16, 9,  4,  1,  0,   1,   4,   9,   16,  25,  //
        121, 100, 81,  64,  49,  36,  25, 16, 9,  4,  1,   0,   1,   4,   9,   16,  //
        144, 121, 100, 81,  64,  49,  36, 25, 16, 9,  4,   1,   0,   1,   4,   9,   //
        169, 144, 121, 100, 81,  64,  49, 36, 25, 16, 9,   4,   1,   0,   1,   4,   //
        196, 169, 144, 121, 100, 81,  64, 49, 36, 25, 16,  9,   4,   1,   0,   1,   //
        225, 196, 169, 144, 121, 100, 81, 64, 49, 36, 25,  16,  9,   4,   1,   0,   //
    };
    count_scalars = nk_size_round_up_to_multiple_(count_scalars, 2);
    nk_size_t n_bytes = count_scalars / 2;
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vuint32m4_t sum_u32m4 = __riscv_vmv_v_x_u32m4(0, vlmax);
    for (nk_size_t vector_length; n_bytes > 0;
         n_bytes -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n_bytes);
        vuint8m1_t a_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        // Build LUT indices: high nibble pair = (a_hi & 0xF0) | (b_hi >> 4)
        vuint8m1_t hi_idx_u8m1 = __riscv_vor_vv_u8m1(__riscv_vand_vx_u8m1(a_packed_u8m1, 0xF0, vector_length),
                                                     __riscv_vsrl_vx_u8m1(b_packed_u8m1, 4, vector_length),
                                                     vector_length);
        // Low nibble pair = (a_lo << 4) | b_lo
        vuint8m1_t lo_idx_u8m1 = __riscv_vor_vv_u8m1(
            __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(a_packed_u8m1, 0x0F, vector_length), 4, vector_length),
            __riscv_vand_vx_u8m1(b_packed_u8m1, 0x0F, vector_length), vector_length);
        // Gather squared differences from LUT (0-225, fits u8)
        vuint8m1_t sq_hi_u8m1 = __riscv_vluxei8_v_u8m1(nk_u4_sqd_lut_, hi_idx_u8m1, vector_length);
        vuint8m1_t sq_lo_u8m1 = __riscv_vluxei8_v_u8m1(nk_u4_sqd_lut_, lo_idx_u8m1, vector_length);
        // Combine and per-lane accumulate: u8+u8→u16, then u32+=u16
        vuint16m2_t combined_u16m2 = __riscv_vwaddu_vv_u16m2(sq_hi_u8m1, sq_lo_u8m1, vector_length);
        sum_u32m4 = __riscv_vwaddu_wv_u32m4_tu(sum_u32m4, sum_u32m4, combined_u16m2, vector_length);
    }
    // Single horizontal reduction after loop
    vuint32m1_t zero_u32m1 = __riscv_vmv_v_x_u32m1(0, vlmax);
    *result = __riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(sum_u32m4, zero_u32m1, vlmax));
}

NK_PUBLIC void nk_euclidean_u4_rvv(nk_u4x2_t const *a_scalars, nk_u4x2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_u32_t d2;
    nk_sqeuclidean_u4_rvv(a_scalars, b_scalars, count_scalars, &d2);
    *result = nk_f32_sqrt_rvv((nk_f32_t)d2);
}

NK_PUBLIC void nk_angular_u4_rvv(nk_u4x2_t const *a_scalars, nk_u4x2_t const *b_scalars, nk_size_t count_scalars,
                                 nk_f32_t *result) {
    static nk_u8_t const nk_u4_dot_lut_[256] = {
        0, 0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   //
        0, 1,  2,  3,  4,  5,  6,  7,   8,   9,   10,  11,  12,  13,  14,  15,  //
        0, 2,  4,  6,  8,  10, 12, 14,  16,  18,  20,  22,  24,  26,  28,  30,  //
        0, 3,  6,  9,  12, 15, 18, 21,  24,  27,  30,  33,  36,  39,  42,  45,  //
        0, 4,  8,  12, 16, 20, 24, 28,  32,  36,  40,  44,  48,  52,  56,  60,  //
        0, 5,  10, 15, 20, 25, 30, 35,  40,  45,  50,  55,  60,  65,  70,  75,  //
        0, 6,  12, 18, 24, 30, 36, 42,  48,  54,  60,  66,  72,  78,  84,  90,  //
        0, 7,  14, 21, 28, 35, 42, 49,  56,  63,  70,  77,  84,  91,  98,  105, //
        0, 8,  16, 24, 32, 40, 48, 56,  64,  72,  80,  88,  96,  104, 112, 120, //
        0, 9,  18, 27, 36, 45, 54, 63,  72,  81,  90,  99,  108, 117, 126, 135, //
        0, 10, 20, 30, 40, 50, 60, 70,  80,  90,  100, 110, 120, 130, 140, 150, //
        0, 11, 22, 33, 44, 55, 66, 77,  88,  99,  110, 121, 132, 143, 154, 165, //
        0, 12, 24, 36, 48, 60, 72, 84,  96,  108, 120, 132, 144, 156, 168, 180, //
        0, 13, 26, 39, 52, 65, 78, 91,  104, 117, 130, 143, 156, 169, 182, 195, //
        0, 14, 28, 42, 56, 70, 84, 98,  112, 126, 140, 154, 168, 182, 196, 210, //
        0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, //
    };
    static nk_u8_t const nk_u4_sq_lut_[16] = {0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225};
    count_scalars = nk_size_round_up_to_multiple_(count_scalars, 2);
    nk_size_t n_bytes = count_scalars / 2;
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vuint32m4_t dot_u32m4 = __riscv_vmv_v_x_u32m4(0, vlmax);
    vuint32m4_t a_norm_sq_u32m4 = __riscv_vmv_v_x_u32m4(0, vlmax);
    vuint32m4_t b_norm_sq_u32m4 = __riscv_vmv_v_x_u32m4(0, vlmax);

    for (nk_size_t vector_length; n_bytes > 0;
         n_bytes -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n_bytes);
        vuint8m1_t a_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);

        // Extract nibbles
        vuint8m1_t a_hi_u8m1 = __riscv_vsrl_vx_u8m1(a_packed_u8m1, 4, vector_length);
        vuint8m1_t b_hi_u8m1 = __riscv_vsrl_vx_u8m1(b_packed_u8m1, 4, vector_length);
        vuint8m1_t a_lo_u8m1 = __riscv_vand_vx_u8m1(a_packed_u8m1, 0x0F, vector_length);
        vuint8m1_t b_lo_u8m1 = __riscv_vand_vx_u8m1(b_packed_u8m1, 0x0F, vector_length);

        // Dot product via 256-entry LUT: dot_lut[(a<<4)|b] = a * b (u8)
        vuint8m1_t hi_idx_u8m1 = __riscv_vor_vv_u8m1(__riscv_vand_vx_u8m1(a_packed_u8m1, 0xF0, vector_length),
                                                     b_hi_u8m1, vector_length);
        vuint8m1_t lo_idx_u8m1 = __riscv_vor_vv_u8m1(__riscv_vsll_vx_u8m1(a_lo_u8m1, 4, vector_length), b_lo_u8m1,
                                                     vector_length);
        vuint8m1_t dot_hi_u8m1 = __riscv_vluxei8_v_u8m1(nk_u4_dot_lut_, hi_idx_u8m1, vector_length);
        vuint8m1_t dot_lo_u8m1 = __riscv_vluxei8_v_u8m1(nk_u4_dot_lut_, lo_idx_u8m1, vector_length);
        // Widen u8→u16, add hi+lo, then per-lane accumulate u32+=u16
        vuint16m2_t dot_combined_u16m2 = __riscv_vwaddu_vv_u16m2(dot_hi_u8m1, dot_lo_u8m1, vector_length);
        dot_u32m4 = __riscv_vwaddu_wv_u32m4_tu(dot_u32m4, dot_u32m4, dot_combined_u16m2, vector_length);

        // Norms via 16-entry squaring LUT + vluxei8
        vuint8m1_t a_hi_sq_u8m1 = __riscv_vluxei8_v_u8m1(nk_u4_sq_lut_, a_hi_u8m1, vector_length);
        vuint8m1_t a_lo_sq_u8m1 = __riscv_vluxei8_v_u8m1(nk_u4_sq_lut_, a_lo_u8m1, vector_length);
        vuint16m2_t a_sq_combined_u16m2 = __riscv_vwaddu_vv_u16m2(a_hi_sq_u8m1, a_lo_sq_u8m1, vector_length);
        a_norm_sq_u32m4 = __riscv_vwaddu_wv_u32m4_tu(a_norm_sq_u32m4, a_norm_sq_u32m4, a_sq_combined_u16m2,
                                                     vector_length);

        vuint8m1_t b_hi_sq_u8m1 = __riscv_vluxei8_v_u8m1(nk_u4_sq_lut_, b_hi_u8m1, vector_length);
        vuint8m1_t b_lo_sq_u8m1 = __riscv_vluxei8_v_u8m1(nk_u4_sq_lut_, b_lo_u8m1, vector_length);
        vuint16m2_t b_sq_combined_u16m2 = __riscv_vwaddu_vv_u16m2(b_hi_sq_u8m1, b_lo_sq_u8m1, vector_length);
        b_norm_sq_u32m4 = __riscv_vwaddu_wv_u32m4_tu(b_norm_sq_u32m4, b_norm_sq_u32m4, b_sq_combined_u16m2,
                                                     vector_length);
    }

    // Single horizontal reductions after loop
    vuint32m1_t zero_u32m1 = __riscv_vmv_v_x_u32m1(0, vlmax);
    nk_u32_t dot_u32 = __riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(dot_u32m4, zero_u32m1, vlmax));
    nk_u32_t a_norm_sq_u32 = __riscv_vmv_x_s_u32m1_u32(
        __riscv_vredsum_vs_u32m4_u32m1(a_norm_sq_u32m4, zero_u32m1, vlmax));
    nk_u32_t b_norm_sq_u32 = __riscv_vmv_x_s_u32m1_u32(
        __riscv_vredsum_vs_u32m4_u32m1(b_norm_sq_u32m4, zero_u32m1, vlmax));

    if (a_norm_sq_u32 == 0 && b_norm_sq_u32 == 0) { *result = 0.0f; }
    else if (dot_u32 == 0) { *result = 1.0f; }
    else {
        nk_f32_t unclipped = 1.0f - (nk_f32_t)dot_u32 * nk_f32_rsqrt_rvv((nk_f32_t)a_norm_sq_u32) *
                                        nk_f32_rsqrt_rvv((nk_f32_t)b_norm_sq_u32);
        *result = unclipped > 0 ? unclipped : 0;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#pragma endregion - Small Integers
#endif // NK_TARGET_RVV
#endif // NK_TARGET_RISCV_
#endif // NK_SPATIAL_RVV_H
