/**
 *  @brief SIMD-accelerated Trigonometric Functions for RISC-V.
 *  @file include/numkong/trigonometry/rvv.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/trigonometry.h
 *  @see https://sleef.org
 *
 *  Implements sin, cos, atan, atan2 for f32 (m4), f64 (m4), and f16 (via f32 m2) using minimax
 *  polynomial approximations with RVV vector intrinsics.
 *
 *  F32 sin/cos use 3-term Horner polynomials with Cody-Waite range reduction modulo pi.
 *  F32 atan uses an 8-term Horner scheme with reciprocal folding for |x| > 1.
 *  F64 sin/cos use 8-term Estrin-style evaluation for better ILP with high-low pi splitting.
 *  F64 atan uses a 19-term Horner polynomial for full double-precision accuracy.
 *
 *  F16 variants convert to f32 (m2) via nk_f16m1_to_f32m2_rvv_, compute in f32, then convert
 *  back via nk_f32m2_to_f16m1_rvv_.
 *
 *  Sign flipping for sin/cos uses XOR with the low bit of the integer quotient shifted to the
 *  sign position, avoiding branch-based selection. Atan uses vmerge for conditional blend.
 *
 *  @section rvv_trig_instructions Key RVV Trigonometry Instructions
 *
 *      Intrinsic                               Purpose
 *      __riscv_vfmadd_vv_f32m4                 FMA: a = a*b + c (Horner step)
 *      __riscv_vfmacc_vv_f32m4                 FMA: a = b*c + a (accumulate form)
 *      __riscv_vfnmsac_vf_f32m4                FNMS: a = a - b*c (range reduction)
 *      __riscv_vfcvt_x_f_v_i32m4              Round-to-nearest float → int
 *      __riscv_vfcvt_f_x_v_f32m4              Int → float conversion
 *      __riscv_vfabs_v_f32m4                   Absolute value
 *      __riscv_vmerge_vvm_f32m4                Conditional select (blend)
 *      __riscv_vfrdiv_vf_f32m4                 Scalar / vector division (reciprocal)
 *      __riscv_vfdiv_vv_f32m4                  Vector / vector division
 */
#ifndef NK_TRIGONOMETRY_RVV_H
#define NK_TRIGONOMETRY_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/cast/rvv.h"
#include "numkong/spatial/rvv.h" /* nk_f32m4_reciprocal_rvv_, nk_f32m2_reciprocal_rvv_, nk_f64m4_reciprocal_rvv_ */

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/*  RVV trigonometry kernels using LMUL=4 for f32 and f64.
 *  Internal helpers return vector register groups for use by geospatial/rvv.h.
 */

NK_INTERNAL vfloat32m4_t nk_f32m4_sin_rvv_(vfloat32m4_t angles, nk_size_t vl) {
    nk_f32_t const pi = 3.14159265358979323846f;
    nk_f32_t const pi_recip = 0.31830988618379067154f;

    // Range reduce: round(angle / pi)
    vfloat32m4_t quotients_f32m4 = __riscv_vfmul_vf_f32m4(angles, pi_recip, vl);
    // vfcvt_x_f rounds to nearest integer by default (RNE)
    vint32m4_t rounded_i32m4 = __riscv_vfcvt_x_f_v_i32m4(quotients_f32m4, vl);
    vfloat32m4_t rounded_f32m4 = __riscv_vfcvt_f_x_v_f32m4(rounded_i32m4, vl);

    // reduced = angle - rounded * pi
    vfloat32m4_t reduced_f32m4 = __riscv_vfnmsac_vf_f32m4(angles, pi, rounded_f32m4, vl);

    // Polynomial: sin(x) ~ x + x^3 * (c1 + x^2 * (c3 + x^2 * c5))
    vfloat32m4_t squared_f32m4 = __riscv_vfmul_vv_f32m4(reduced_f32m4, reduced_f32m4, vl);
    vfloat32m4_t cubed_f32m4 = __riscv_vfmul_vv_f32m4(reduced_f32m4, squared_f32m4, vl);

    vfloat32m4_t poly_f32m4 = __riscv_vfmv_v_f_f32m4(-0.0001881748176f, vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, squared_f32m4, __riscv_vfmv_v_f_f32m4(0.008323502727f, vl), vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, squared_f32m4, __riscv_vfmv_v_f_f32m4(-0.1666651368f, vl), vl);
    vfloat32m4_t result_f32m4 = __riscv_vfmacc_vv_f32m4(reduced_f32m4, cubed_f32m4, poly_f32m4, vl);

    // Sign flip if rounded is odd: XOR bit 0 of rounded_i32m4 shifted to sign position
    vuint32m4_t sign_mask_u32m4 = __riscv_vsll_vx_u32m4(__riscv_vreinterpret_v_i32m4_u32m4(rounded_i32m4), 31, vl);
    vuint32m4_t result_bits_u32m4 = __riscv_vxor_vv_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(result_f32m4),
                                                          sign_mask_u32m4, vl);
    return __riscv_vreinterpret_v_u32m4_f32m4(result_bits_u32m4);
}

NK_INTERNAL vfloat32m4_t nk_f32m4_cos_rvv_(vfloat32m4_t angles, nk_size_t vl) {
    nk_f32_t const pi = 3.14159265358979323846f;
    nk_f32_t const pi_half = 1.57079632679489661923f;
    nk_f32_t const pi_recip = 0.31830988618379067154f;

    // Compute round((angle / pi) - 0.5)
    vfloat32m4_t quotients_f32m4 = __riscv_vfsub_vf_f32m4(__riscv_vfmul_vf_f32m4(angles, pi_recip, vl), 0.5f, vl);
    vint32m4_t rounded_i32m4 = __riscv_vfcvt_x_f_v_i32m4(quotients_f32m4, vl);
    vfloat32m4_t rounded_f32m4 = __riscv_vfcvt_f_x_v_f32m4(rounded_i32m4, vl);

    // Reduce: angle - (rounded * pi + pi/2)
    vfloat32m4_t offset_f32m4 = __riscv_vfmacc_vf_f32m4(__riscv_vfmv_v_f_f32m4(pi_half, vl), pi, rounded_f32m4, vl);
    vfloat32m4_t reduced_f32m4 = __riscv_vfsub_vv_f32m4(angles, offset_f32m4, vl);

    // Polynomial: same 3-term approximation
    vfloat32m4_t squared_f32m4 = __riscv_vfmul_vv_f32m4(reduced_f32m4, reduced_f32m4, vl);
    vfloat32m4_t cubed_f32m4 = __riscv_vfmul_vv_f32m4(reduced_f32m4, squared_f32m4, vl);

    vfloat32m4_t poly_f32m4 = __riscv_vfmv_v_f_f32m4(-0.0001881748176f, vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, squared_f32m4, __riscv_vfmv_v_f_f32m4(0.008323502727f, vl), vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, squared_f32m4, __riscv_vfmv_v_f_f32m4(-0.1666651368f, vl), vl);
    vfloat32m4_t result_f32m4 = __riscv_vfmacc_vv_f32m4(reduced_f32m4, cubed_f32m4, poly_f32m4, vl);

    // If rounded is even, flip the sign
    vuint32m4_t parity_u32m4 = __riscv_vand_vx_u32m4(__riscv_vreinterpret_v_i32m4_u32m4(rounded_i32m4), 1, vl);
    vbool8_t even_mask_b8 = __riscv_vmseq_vx_u32m4_b8(parity_u32m4, 0, vl);
    result_f32m4 = __riscv_vfneg_v_f32m4_mu(even_mask_b8, result_f32m4, result_f32m4, vl);
    return result_f32m4;
}

NK_INTERNAL vfloat32m4_t nk_f32m4_atan_rvv_(vfloat32m4_t inputs, nk_size_t vl) {
    // 8-term polynomial coefficients for atan approximation
    nk_f32_t const c8 = -0.333331018686294555664062f;
    nk_f32_t const c7 = +0.199926957488059997558594f;
    nk_f32_t const c6 = -0.142027363181114196777344f;
    nk_f32_t const c5 = +0.106347933411598205566406f;
    nk_f32_t const c4 = -0.0748900920152664184570312f;
    nk_f32_t const c3 = +0.0425049886107444763183594f;
    nk_f32_t const c2 = -0.0159569028764963150024414f;
    nk_f32_t const c1 = +0.00282363896258175373077393f;

    // Detect negative values
    vbool8_t negative_mask_b8 = __riscv_vmflt_vf_f32m4_b8(inputs, 0.0f, vl);
    vfloat32m4_t values_f32m4 = __riscv_vfabs_v_f32m4(inputs, vl);

    // Check if values > 1 (need reciprocal)
    vbool8_t reciprocal_mask_b8 = __riscv_vmfgt_vf_f32m4_b8(values_f32m4, 1.0f, vl);
    vfloat32m4_t reciprocal_values_f32m4 = nk_f32m4_reciprocal_rvv_(values_f32m4, vl);
    values_f32m4 = __riscv_vmerge_vvm_f32m4(values_f32m4, reciprocal_values_f32m4, reciprocal_mask_b8, vl);

    // Compute powers
    vfloat32m4_t squared_f32m4 = __riscv_vfmul_vv_f32m4(values_f32m4, values_f32m4, vl);
    vfloat32m4_t cubed_f32m4 = __riscv_vfmul_vv_f32m4(values_f32m4, squared_f32m4, vl);

    // Horner evaluation: P(x^2) = c1 + x^2*(c2 + x^2*(c3 + ... ))
    vfloat32m4_t poly_f32m4 = __riscv_vfmv_v_f_f32m4(c1, vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, squared_f32m4, __riscv_vfmv_v_f_f32m4(c2, vl), vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, squared_f32m4, __riscv_vfmv_v_f_f32m4(c3, vl), vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, squared_f32m4, __riscv_vfmv_v_f_f32m4(c4, vl), vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, squared_f32m4, __riscv_vfmv_v_f_f32m4(c5, vl), vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, squared_f32m4, __riscv_vfmv_v_f_f32m4(c6, vl), vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, squared_f32m4, __riscv_vfmv_v_f_f32m4(c7, vl), vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, squared_f32m4, __riscv_vfmv_v_f_f32m4(c8, vl), vl);

    // result = x + x^3 * P(x^2)
    vfloat32m4_t result_f32m4 = __riscv_vfmacc_vv_f32m4(values_f32m4, cubed_f32m4, poly_f32m4, vl);

    // Adjust for reciprocal: result = pi/2 - result
    result_f32m4 = __riscv_vfrsub_vf_f32m4_mu(reciprocal_mask_b8, result_f32m4, result_f32m4, 1.5707963267948966f, vl);

    // Adjust for negative: result = -result
    result_f32m4 = __riscv_vfneg_v_f32m4_mu(negative_mask_b8, result_f32m4, result_f32m4, vl);
    return result_f32m4;
}

NK_INTERNAL vfloat32m4_t nk_f32m4_atan2_rvv_(vfloat32m4_t ys_inputs, vfloat32m4_t xs_inputs, nk_size_t vl) {
    // 8-term polynomial coefficients (same as atan)
    nk_f32_t const c8 = -0.333331018686294555664062f;
    nk_f32_t const c7 = +0.199926957488059997558594f;
    nk_f32_t const c6 = -0.142027363181114196777344f;
    nk_f32_t const c5 = +0.106347933411598205566406f;
    nk_f32_t const c4 = -0.0748900920152664184570312f;
    nk_f32_t const c3 = +0.0425049886107444763183594f;
    nk_f32_t const c2 = -0.0159569028764963150024414f;
    nk_f32_t const c1 = +0.00282363896258175373077393f;

    // Quadrant adjustments - take absolute values
    vbool8_t xs_negative_mask_b8 = __riscv_vmflt_vf_f32m4_b8(xs_inputs, 0.0f, vl);
    vfloat32m4_t xs_f32m4 = __riscv_vfabs_v_f32m4(xs_inputs, vl);
    vfloat32m4_t ys_f32m4 = __riscv_vfabs_v_f32m4(ys_inputs, vl);

    // Ensure proper fraction where numerator < denominator
    vbool8_t swap_mask_b8 = __riscv_vmfgt_vv_f32m4_b8(ys_f32m4, xs_f32m4, vl);
    vfloat32m4_t saved_xs_f32m4 = xs_f32m4;
    xs_f32m4 = __riscv_vmerge_vvm_f32m4(xs_f32m4, ys_f32m4, swap_mask_b8, vl);
    ys_f32m4 = __riscv_vfneg_v_f32m4_mu(swap_mask_b8, ys_f32m4, saved_xs_f32m4, vl);

    // Compute ratio and powers
    vfloat32m4_t ratio_f32m4 = __riscv_vfmul_vv_f32m4(ys_f32m4, nk_f32m4_reciprocal_rvv_(xs_f32m4, vl), vl);
    vfloat32m4_t ratio_squared_f32m4 = __riscv_vfmul_vv_f32m4(ratio_f32m4, ratio_f32m4, vl);
    vfloat32m4_t ratio_cubed_f32m4 = __riscv_vfmul_vv_f32m4(ratio_f32m4, ratio_squared_f32m4, vl);

    // Horner evaluation
    vfloat32m4_t poly_f32m4 = __riscv_vfmv_v_f_f32m4(c1, vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, ratio_squared_f32m4, __riscv_vfmv_v_f_f32m4(c2, vl), vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, ratio_squared_f32m4, __riscv_vfmv_v_f_f32m4(c3, vl), vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, ratio_squared_f32m4, __riscv_vfmv_v_f_f32m4(c4, vl), vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, ratio_squared_f32m4, __riscv_vfmv_v_f_f32m4(c5, vl), vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, ratio_squared_f32m4, __riscv_vfmv_v_f_f32m4(c6, vl), vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, ratio_squared_f32m4, __riscv_vfmv_v_f_f32m4(c7, vl), vl);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, ratio_squared_f32m4, __riscv_vfmv_v_f_f32m4(c8, vl), vl);

    // result = ratio + ratio^3 * P(ratio^2)
    vfloat32m4_t results_f32m4 = __riscv_vfmacc_vv_f32m4(ratio_f32m4, ratio_cubed_f32m4, poly_f32m4, vl);

    // Compute quadrant value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    vfloat32m4_t quadrant_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    quadrant_f32m4 = __riscv_vmerge_vvm_f32m4(quadrant_f32m4, __riscv_vfmv_v_f_f32m4(-2.0f, vl), xs_negative_mask_b8,
                                              vl);
    vfloat32m4_t quadrant_incremented_f32m4 = __riscv_vfadd_vf_f32m4(quadrant_f32m4, 1.0f, vl);
    quadrant_f32m4 = __riscv_vmerge_vvm_f32m4(quadrant_f32m4, quadrant_incremented_f32m4, swap_mask_b8, vl);

    // Adjust for quadrant: result += quadrant * pi/2
    results_f32m4 = __riscv_vfmacc_vf_f32m4(results_f32m4, 1.5707963267948966f, quadrant_f32m4, vl);

    // Transfer sign from x (XOR with sign bit of xs_inputs)
    vuint32m4_t sign_mask_u32m4 = __riscv_vreinterpret_v_f32m4_u32m4(__riscv_vfmv_v_f_f32m4(-0.0f, vl));
    vuint32m4_t xs_sign_bits_u32m4 = __riscv_vand_vv_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(xs_inputs),
                                                           sign_mask_u32m4, vl);
    vuint32m4_t result_bits_u32m4 = __riscv_vxor_vv_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(results_f32m4),
                                                          xs_sign_bits_u32m4, vl);

    // Transfer sign from y (XOR with sign bit of ys_inputs)
    vuint32m4_t ys_sign_bits_u32m4 = __riscv_vand_vv_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(ys_inputs),
                                                           sign_mask_u32m4, vl);
    result_bits_u32m4 = __riscv_vxor_vv_u32m4(result_bits_u32m4, ys_sign_bits_u32m4, vl);

    return __riscv_vreinterpret_v_u32m4_f32m4(result_bits_u32m4);
}

NK_INTERNAL vfloat64m4_t nk_f64m4_sin_rvv_(vfloat64m4_t angles_radians, nk_size_t vl) {
    // Constants for two-step Cody-Waite range reduction
    nk_f64_t const pi_high = 3.141592653589793116;
    nk_f64_t const pi_low = 1.2246467991473532072e-16;
    nk_f64_t const pi_recip = 0.31830988618379067154;

    // Polynomial coefficients for sine approximation (8 terms + linear)
    nk_f64_t const c0 = +0.00833333333333332974823815;
    nk_f64_t const c1 = -0.000198412698412696162806809;
    nk_f64_t const c2 = +2.75573192239198747630416e-06;
    nk_f64_t const c3 = -2.50521083763502045810755e-08;
    nk_f64_t const c4 = +1.60590430605664501629054e-10;
    nk_f64_t const c5 = -7.64712219118158833288484e-13;
    nk_f64_t const c6 = +2.81009972710863200091251e-15;
    nk_f64_t const c7 = -7.97255955009037868891952e-18;
    nk_f64_t const c8 = -0.166666666666666657414808;

    // Compute round(angle / pi)
    vfloat64m4_t quotients_f64m4 = __riscv_vfmul_vf_f64m4(angles_radians, pi_recip, vl);
    // Round to nearest: vfcvt_x_f rounds to nearest (RNE), then convert back
    vint64m4_t rounded_i64m4 = __riscv_vfcvt_x_f_v_i64m4(quotients_f64m4, vl);
    vfloat64m4_t rounded_f64m4 = __riscv_vfcvt_f_x_v_f64m4(rounded_i64m4, vl);

    // Two-step Cody-Waite reduction: angle - rounded * pi_high - rounded * pi_low
    vfloat64m4_t angles_f64m4 = __riscv_vfnmsac_vf_f64m4(angles_radians, pi_high, rounded_f64m4, vl);
    angles_f64m4 = __riscv_vfnmsac_vf_f64m4(angles_f64m4, pi_low, rounded_f64m4, vl);

    // If rounded is odd, negate the angle
    vuint64m4_t parity_u64m4 = __riscv_vand_vx_u64m4(__riscv_vreinterpret_v_i64m4_u64m4(rounded_i64m4), 1, vl);
    vbool16_t odd_mask_b16 = __riscv_vmsne_vx_u64m4_b16(parity_u64m4, 0, vl);
    angles_f64m4 = __riscv_vfneg_v_f64m4_mu(odd_mask_b16, angles_f64m4, angles_f64m4, vl);

    vfloat64m4_t squared_f64m4 = __riscv_vfmul_vv_f64m4(angles_f64m4, angles_f64m4, vl);
    vfloat64m4_t cubed_f64m4 = __riscv_vfmul_vv_f64m4(angles_f64m4, squared_f64m4, vl);
    vfloat64m4_t fourth_f64m4 = __riscv_vfmul_vv_f64m4(squared_f64m4, squared_f64m4, vl);
    vfloat64m4_t eighth_f64m4 = __riscv_vfmul_vv_f64m4(fourth_f64m4, fourth_f64m4, vl);

    // Estrin-style evaluation for better ILP
    // poly_67 = c6 + a2 * c7
    vfloat64m4_t poly_67_f64m4 = __riscv_vfmacc_vf_f64m4(__riscv_vfmv_v_f_f64m4(c6, vl), c7, squared_f64m4, vl);
    // poly_45 = c4 + a2 * c5
    vfloat64m4_t poly_45_f64m4 = __riscv_vfmacc_vf_f64m4(__riscv_vfmv_v_f_f64m4(c4, vl), c5, squared_f64m4, vl);
    // poly_4567 = poly_45 + a4 * poly_67
    vfloat64m4_t poly_4567_f64m4 = __riscv_vfmacc_vv_f64m4(poly_45_f64m4, fourth_f64m4, poly_67_f64m4, vl);

    // poly_23 = c2 + a2 * c3
    vfloat64m4_t poly_23_f64m4 = __riscv_vfmacc_vf_f64m4(__riscv_vfmv_v_f_f64m4(c2, vl), c3, squared_f64m4, vl);
    // poly_01 = c0 + a2 * c1
    vfloat64m4_t poly_01_f64m4 = __riscv_vfmacc_vf_f64m4(__riscv_vfmv_v_f_f64m4(c0, vl), c1, squared_f64m4, vl);
    // poly_0123 = poly_01 + a4 * poly_23
    vfloat64m4_t poly_0123_f64m4 = __riscv_vfmacc_vv_f64m4(poly_01_f64m4, fourth_f64m4, poly_23_f64m4, vl);

    // Combine: results = poly_0123 + a8 * poly_4567
    vfloat64m4_t results_f64m4 = __riscv_vfmacc_vv_f64m4(poly_0123_f64m4, eighth_f64m4, poly_4567_f64m4, vl);
    // results = c8 + a2 * results
    results_f64m4 = __riscv_vfmadd_vv_f64m4(results_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c8, vl), vl);
    // results = angles + a3 * results
    results_f64m4 = __riscv_vfmacc_vv_f64m4(angles_f64m4, cubed_f64m4, results_f64m4, vl);

    // Handle zero input (preserve sign of zero)
    vbool16_t non_zero_mask_b16 = __riscv_vmfne_vf_f64m4_b16(angles_radians, 0.0, vl);
    vfloat64m4_t zeros_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vl);
    results_f64m4 = __riscv_vmerge_vvm_f64m4(zeros_f64m4, results_f64m4, non_zero_mask_b16, vl);
    return results_f64m4;
}

NK_INTERNAL vfloat64m4_t nk_f64m4_cos_rvv_(vfloat64m4_t angles_radians, nk_size_t vl) {
    // Constants for two-step Cody-Waite range reduction
    nk_f64_t const pi_high_half = 3.141592653589793116 * 0.5;
    nk_f64_t const pi_low_half = 1.2246467991473532072e-16 * 0.5;
    nk_f64_t const pi_recip = 0.31830988618379067154;

    // Polynomial coefficients (same as sin f64)
    nk_f64_t const c0 = +0.00833333333333332974823815;
    nk_f64_t const c1 = -0.000198412698412696162806809;
    nk_f64_t const c2 = +2.75573192239198747630416e-06;
    nk_f64_t const c3 = -2.50521083763502045810755e-08;
    nk_f64_t const c4 = +1.60590430605664501629054e-10;
    nk_f64_t const c5 = -7.64712219118158833288484e-13;
    nk_f64_t const c6 = +2.81009972710863200091251e-15;
    nk_f64_t const c7 = -7.97255955009037868891952e-18;
    nk_f64_t const c8 = -0.166666666666666657414808;

    // Compute 2 * round(angle / pi - 0.5) + 1
    vfloat64m4_t quotients_f64m4 = __riscv_vfsub_vf_f64m4(__riscv_vfmul_vf_f64m4(angles_radians, pi_recip, vl), 0.5,
                                                          vl);
    vint64m4_t rounded_i64m4 = __riscv_vfcvt_x_f_v_i64m4(quotients_f64m4, vl);
    vfloat64m4_t rounded_f64m4 = __riscv_vfcvt_f_x_v_f64m4(rounded_i64m4, vl);
    // rounded_quotients = 2 * rounded + 1
    vfloat64m4_t rounded_quotients_f64m4 = __riscv_vfmadd_vf_f64m4(rounded_f64m4, 2.0, __riscv_vfmv_v_f_f64m4(1.0, vl),
                                                                   vl);

    // Two-step Cody-Waite reduction: angle - rounded_quotients * pi_high_half - rounded_quotients * pi_low_half
    vfloat64m4_t angles_f64m4 = __riscv_vfnmsac_vf_f64m4(angles_radians, pi_high_half, rounded_quotients_f64m4, vl);
    angles_f64m4 = __riscv_vfnmsac_vf_f64m4(angles_f64m4, pi_low_half, rounded_quotients_f64m4, vl);

    // If (rounded_quotients & 2) == 0, negate the angle
    // We need the integer version of rounded_quotients for the bit check
    vint64m4_t quotients_i64m4 = __riscv_vfcvt_x_f_v_i64m4(rounded_quotients_f64m4, vl);
    vuint64m4_t bit2_u64m4 = __riscv_vand_vx_u64m4(__riscv_vreinterpret_v_i64m4_u64m4(quotients_i64m4), 2, vl);
    vbool16_t flip_mask_b16 = __riscv_vmseq_vx_u64m4_b16(bit2_u64m4, 0, vl);
    angles_f64m4 = __riscv_vfneg_v_f64m4_mu(flip_mask_b16, angles_f64m4, angles_f64m4, vl);

    vfloat64m4_t squared_f64m4 = __riscv_vfmul_vv_f64m4(angles_f64m4, angles_f64m4, vl);
    vfloat64m4_t cubed_f64m4 = __riscv_vfmul_vv_f64m4(angles_f64m4, squared_f64m4, vl);
    vfloat64m4_t fourth_f64m4 = __riscv_vfmul_vv_f64m4(squared_f64m4, squared_f64m4, vl);
    vfloat64m4_t eighth_f64m4 = __riscv_vfmul_vv_f64m4(fourth_f64m4, fourth_f64m4, vl);

    // Estrin-style evaluation
    vfloat64m4_t poly_67_f64m4 = __riscv_vfmacc_vf_f64m4(__riscv_vfmv_v_f_f64m4(c6, vl), c7, squared_f64m4, vl);
    vfloat64m4_t poly_45_f64m4 = __riscv_vfmacc_vf_f64m4(__riscv_vfmv_v_f_f64m4(c4, vl), c5, squared_f64m4, vl);
    vfloat64m4_t poly_4567_f64m4 = __riscv_vfmacc_vv_f64m4(poly_45_f64m4, fourth_f64m4, poly_67_f64m4, vl);

    vfloat64m4_t poly_23_f64m4 = __riscv_vfmacc_vf_f64m4(__riscv_vfmv_v_f_f64m4(c2, vl), c3, squared_f64m4, vl);
    vfloat64m4_t poly_01_f64m4 = __riscv_vfmacc_vf_f64m4(__riscv_vfmv_v_f_f64m4(c0, vl), c1, squared_f64m4, vl);
    vfloat64m4_t poly_0123_f64m4 = __riscv_vfmacc_vv_f64m4(poly_01_f64m4, fourth_f64m4, poly_23_f64m4, vl);

    vfloat64m4_t results_f64m4 = __riscv_vfmacc_vv_f64m4(poly_0123_f64m4, eighth_f64m4, poly_4567_f64m4, vl);
    results_f64m4 = __riscv_vfmadd_vv_f64m4(results_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c8, vl), vl);
    results_f64m4 = __riscv_vfmacc_vv_f64m4(angles_f64m4, cubed_f64m4, results_f64m4, vl);
    return results_f64m4;
}

NK_INTERNAL vfloat64m4_t nk_f64m4_atan_rvv_(vfloat64m4_t inputs, nk_size_t vl) {
    // 19-term polynomial coefficients
    nk_f64_t const c19 = -1.88796008463073496563746e-05;
    nk_f64_t const c18 = +0.000209850076645816976906797;
    nk_f64_t const c17 = -0.00110611831486672482563471;
    nk_f64_t const c16 = +0.00370026744188713119232403;
    nk_f64_t const c15 = -0.00889896195887655491740809;
    nk_f64_t const c14 = +0.016599329773529201970117;
    nk_f64_t const c13 = -0.0254517624932312641616861;
    nk_f64_t const c12 = +0.0337852580001353069993897;
    nk_f64_t const c11 = -0.0407629191276836500001934;
    nk_f64_t const c10 = +0.0466667150077840625632675;
    nk_f64_t const c9 = -0.0523674852303482457616113;
    nk_f64_t const c8 = +0.0587666392926673580854313;
    nk_f64_t const c7 = -0.0666573579361080525984562;
    nk_f64_t const c6 = +0.0769219538311769618355029;
    nk_f64_t const c5 = -0.090908995008245008229153;
    nk_f64_t const c4 = +0.111111105648261418443745;
    nk_f64_t const c3 = -0.14285714266771329383765;
    nk_f64_t const c2 = +0.199999999996591265594148;
    nk_f64_t const c1 = -0.333333333333311110369124;

    // Detect negative values
    vbool16_t negative_mask_b16 = __riscv_vmflt_vf_f64m4_b16(inputs, 0.0, vl);
    vfloat64m4_t values_f64m4 = __riscv_vfabs_v_f64m4(inputs, vl);

    // Check if values > 1 (need reciprocal)
    vbool16_t reciprocal_mask_b16 = __riscv_vmfgt_vf_f64m4_b16(values_f64m4, 1.0, vl);
    vfloat64m4_t reciprocal_values_f64m4 = nk_f64m4_reciprocal_rvv_(values_f64m4, vl);
    values_f64m4 = __riscv_vmerge_vvm_f64m4(values_f64m4, reciprocal_values_f64m4, reciprocal_mask_b16, vl);

    // Compute powers
    vfloat64m4_t squared_f64m4 = __riscv_vfmul_vv_f64m4(values_f64m4, values_f64m4, vl);
    vfloat64m4_t cubed_f64m4 = __riscv_vfmul_vv_f64m4(values_f64m4, squared_f64m4, vl);

    // Horner evaluation: 19 terms
    vfloat64m4_t poly_f64m4 = __riscv_vfmv_v_f_f64m4(c19, vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c18, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c17, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c16, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c15, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c14, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c13, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c12, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c11, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c10, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c9, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c8, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c7, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c6, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c5, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c4, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c3, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c2, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, squared_f64m4, __riscv_vfmv_v_f_f64m4(c1, vl), vl);

    // result = x + x^3 * P(x^2)
    vfloat64m4_t result_f64m4 = __riscv_vfmacc_vv_f64m4(values_f64m4, cubed_f64m4, poly_f64m4, vl);

    // Adjust for reciprocal: result = pi/2 - result
    result_f64m4 = __riscv_vfrsub_vf_f64m4_mu(reciprocal_mask_b16, result_f64m4, result_f64m4, 1.5707963267948966, vl);

    // Adjust for negative: result = -result
    result_f64m4 = __riscv_vfneg_v_f64m4_mu(negative_mask_b16, result_f64m4, result_f64m4, vl);
    return result_f64m4;
}

NK_INTERNAL vfloat64m4_t nk_f64m4_atan2_rvv_(vfloat64m4_t ys_inputs, vfloat64m4_t xs_inputs, nk_size_t vl) {
    // 19-term polynomial coefficients (same as atan)
    nk_f64_t const c19 = -1.88796008463073496563746e-05;
    nk_f64_t const c18 = +0.000209850076645816976906797;
    nk_f64_t const c17 = -0.00110611831486672482563471;
    nk_f64_t const c16 = +0.00370026744188713119232403;
    nk_f64_t const c15 = -0.00889896195887655491740809;
    nk_f64_t const c14 = +0.016599329773529201970117;
    nk_f64_t const c13 = -0.0254517624932312641616861;
    nk_f64_t const c12 = +0.0337852580001353069993897;
    nk_f64_t const c11 = -0.0407629191276836500001934;
    nk_f64_t const c10 = +0.0466667150077840625632675;
    nk_f64_t const c9 = -0.0523674852303482457616113;
    nk_f64_t const c8 = +0.0587666392926673580854313;
    nk_f64_t const c7 = -0.0666573579361080525984562;
    nk_f64_t const c6 = +0.0769219538311769618355029;
    nk_f64_t const c5 = -0.090908995008245008229153;
    nk_f64_t const c4 = +0.111111105648261418443745;
    nk_f64_t const c3 = -0.14285714266771329383765;
    nk_f64_t const c2 = +0.199999999996591265594148;
    nk_f64_t const c1 = -0.333333333333311110369124;

    // Quadrant adjustments - take absolute values
    vbool16_t xs_negative_mask_b16 = __riscv_vmflt_vf_f64m4_b16(xs_inputs, 0.0, vl);
    vfloat64m4_t xs_f64m4 = __riscv_vfabs_v_f64m4(xs_inputs, vl);
    vfloat64m4_t ys_f64m4 = __riscv_vfabs_v_f64m4(ys_inputs, vl);

    // Ensure proper fraction where numerator < denominator
    vbool16_t swap_mask_b16 = __riscv_vmfgt_vv_f64m4_b16(ys_f64m4, xs_f64m4, vl);
    vfloat64m4_t saved_xs_f64m4 = xs_f64m4;
    xs_f64m4 = __riscv_vmerge_vvm_f64m4(xs_f64m4, ys_f64m4, swap_mask_b16, vl);
    ys_f64m4 = __riscv_vfneg_v_f64m4_mu(swap_mask_b16, ys_f64m4, saved_xs_f64m4, vl);

    // Compute ratio and powers
    vfloat64m4_t ratio_f64m4 = __riscv_vfmul_vv_f64m4(ys_f64m4, nk_f64m4_reciprocal_rvv_(xs_f64m4, vl), vl);
    vfloat64m4_t ratio_squared_f64m4 = __riscv_vfmul_vv_f64m4(ratio_f64m4, ratio_f64m4, vl);
    vfloat64m4_t ratio_cubed_f64m4 = __riscv_vfmul_vv_f64m4(ratio_f64m4, ratio_squared_f64m4, vl);

    // Horner evaluation: 19 terms
    vfloat64m4_t poly_f64m4 = __riscv_vfmv_v_f_f64m4(c19, vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c18, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c17, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c16, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c15, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c14, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c13, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c12, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c11, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c10, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c9, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c8, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c7, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c6, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c5, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c4, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c3, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c2, vl), vl);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(poly_f64m4, ratio_squared_f64m4, __riscv_vfmv_v_f_f64m4(c1, vl), vl);

    // result = ratio + ratio^3 * P(ratio^2)
    vfloat64m4_t results_f64m4 = __riscv_vfmacc_vv_f64m4(ratio_f64m4, ratio_cubed_f64m4, poly_f64m4, vl);

    // Compute quadrant value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    vfloat64m4_t quadrant_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vl);
    quadrant_f64m4 = __riscv_vmerge_vvm_f64m4(quadrant_f64m4, __riscv_vfmv_v_f_f64m4(-2.0, vl), xs_negative_mask_b16,
                                              vl);
    vfloat64m4_t quadrant_incremented_f64m4 = __riscv_vfadd_vf_f64m4(quadrant_f64m4, 1.0, vl);
    quadrant_f64m4 = __riscv_vmerge_vvm_f64m4(quadrant_f64m4, quadrant_incremented_f64m4, swap_mask_b16, vl);

    // Adjust for quadrant: result += quadrant * pi/2
    results_f64m4 = __riscv_vfmacc_vf_f64m4(results_f64m4, 1.5707963267948966, quadrant_f64m4, vl);

    // Transfer sign from x (XOR with sign bit of xs_inputs)
    vuint64m4_t sign_mask_u64m4 = __riscv_vreinterpret_v_f64m4_u64m4(__riscv_vfmv_v_f_f64m4(-0.0, vl));
    vuint64m4_t xs_sign_bits_u64m4 = __riscv_vand_vv_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(xs_inputs),
                                                           sign_mask_u64m4, vl);
    vuint64m4_t result_bits_u64m4 = __riscv_vxor_vv_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(results_f64m4),
                                                          xs_sign_bits_u64m4, vl);

    // Transfer sign from y (XOR with sign bit of ys_inputs)
    vuint64m4_t ys_sign_bits_u64m4 = __riscv_vand_vv_u64m4(__riscv_vreinterpret_v_f64m4_u64m4(ys_inputs),
                                                           sign_mask_u64m4, vl);
    result_bits_u64m4 = __riscv_vxor_vv_u64m4(result_bits_u64m4, ys_sign_bits_u64m4, vl);

    return __riscv_vreinterpret_v_u64m4_f64m4(result_bits_u64m4);
}

/*  m2-width versions of sin/cos/atan for the f16 conversion path.
 *  f16 data is loaded as m1 (16-bit), widened to f32 m2, computed, then narrowed back.
 */

NK_INTERNAL vfloat32m2_t nk_f32m2_sin_rvv_(vfloat32m2_t angles, nk_size_t vl) {
    nk_f32_t const pi = 3.14159265358979323846f;
    nk_f32_t const pi_recip = 0.31830988618379067154f;

    vfloat32m2_t quotients_f32m2 = __riscv_vfmul_vf_f32m2(angles, pi_recip, vl);
    vint32m2_t rounded_i32m2 = __riscv_vfcvt_x_f_v_i32m2(quotients_f32m2, vl);
    vfloat32m2_t rounded_f32m2 = __riscv_vfcvt_f_x_v_f32m2(rounded_i32m2, vl);

    vfloat32m2_t reduced_f32m2 = __riscv_vfnmsac_vf_f32m2(angles, pi, rounded_f32m2, vl);
    vfloat32m2_t squared_f32m2 = __riscv_vfmul_vv_f32m2(reduced_f32m2, reduced_f32m2, vl);
    vfloat32m2_t cubed_f32m2 = __riscv_vfmul_vv_f32m2(reduced_f32m2, squared_f32m2, vl);

    vfloat32m2_t poly_f32m2 = __riscv_vfmv_v_f_f32m2(-0.0001881748176f, vl);
    poly_f32m2 = __riscv_vfmadd_vv_f32m2(poly_f32m2, squared_f32m2, __riscv_vfmv_v_f_f32m2(0.008323502727f, vl), vl);
    poly_f32m2 = __riscv_vfmadd_vv_f32m2(poly_f32m2, squared_f32m2, __riscv_vfmv_v_f_f32m2(-0.1666651368f, vl), vl);
    vfloat32m2_t result_f32m2 = __riscv_vfmacc_vv_f32m2(reduced_f32m2, cubed_f32m2, poly_f32m2, vl);

    vuint32m2_t sign_mask_u32m2 = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_i32m2_u32m2(rounded_i32m2), 31, vl);
    vuint32m2_t result_bits_u32m2 = __riscv_vxor_vv_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(result_f32m2),
                                                          sign_mask_u32m2, vl);
    return __riscv_vreinterpret_v_u32m2_f32m2(result_bits_u32m2);
}

NK_INTERNAL vfloat32m2_t nk_f32m2_cos_rvv_(vfloat32m2_t angles, nk_size_t vl) {
    nk_f32_t const pi = 3.14159265358979323846f;
    nk_f32_t const pi_half = 1.57079632679489661923f;
    nk_f32_t const pi_recip = 0.31830988618379067154f;

    vfloat32m2_t quotients_f32m2 = __riscv_vfsub_vf_f32m2(__riscv_vfmul_vf_f32m2(angles, pi_recip, vl), 0.5f, vl);
    vint32m2_t rounded_i32m2 = __riscv_vfcvt_x_f_v_i32m2(quotients_f32m2, vl);
    vfloat32m2_t rounded_f32m2 = __riscv_vfcvt_f_x_v_f32m2(rounded_i32m2, vl);

    vfloat32m2_t offset_f32m2 = __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(pi_half, vl), pi, rounded_f32m2, vl);
    vfloat32m2_t reduced_f32m2 = __riscv_vfsub_vv_f32m2(angles, offset_f32m2, vl);

    vfloat32m2_t squared_f32m2 = __riscv_vfmul_vv_f32m2(reduced_f32m2, reduced_f32m2, vl);
    vfloat32m2_t cubed_f32m2 = __riscv_vfmul_vv_f32m2(reduced_f32m2, squared_f32m2, vl);

    vfloat32m2_t poly_f32m2 = __riscv_vfmv_v_f_f32m2(-0.0001881748176f, vl);
    poly_f32m2 = __riscv_vfmadd_vv_f32m2(poly_f32m2, squared_f32m2, __riscv_vfmv_v_f_f32m2(0.008323502727f, vl), vl);
    poly_f32m2 = __riscv_vfmadd_vv_f32m2(poly_f32m2, squared_f32m2, __riscv_vfmv_v_f_f32m2(-0.1666651368f, vl), vl);
    vfloat32m2_t result_f32m2 = __riscv_vfmacc_vv_f32m2(reduced_f32m2, cubed_f32m2, poly_f32m2, vl);

    vuint32m2_t parity_u32m2 = __riscv_vand_vx_u32m2(__riscv_vreinterpret_v_i32m2_u32m2(rounded_i32m2), 1, vl);
    vbool16_t even_mask_b16 = __riscv_vmseq_vx_u32m2_b16(parity_u32m2, 0, vl);
    result_f32m2 = __riscv_vfneg_v_f32m2_mu(even_mask_b16, result_f32m2, result_f32m2, vl);
    return result_f32m2;
}

NK_INTERNAL vfloat32m2_t nk_f32m2_atan_rvv_(vfloat32m2_t inputs, nk_size_t vl) {
    nk_f32_t const c8 = -0.333331018686294555664062f;
    nk_f32_t const c7 = +0.199926957488059997558594f;
    nk_f32_t const c6 = -0.142027363181114196777344f;
    nk_f32_t const c5 = +0.106347933411598205566406f;
    nk_f32_t const c4 = -0.0748900920152664184570312f;
    nk_f32_t const c3 = +0.0425049886107444763183594f;
    nk_f32_t const c2 = -0.0159569028764963150024414f;
    nk_f32_t const c1 = +0.00282363896258175373077393f;

    vbool16_t negative_mask_b16 = __riscv_vmflt_vf_f32m2_b16(inputs, 0.0f, vl);
    vfloat32m2_t values_f32m2 = __riscv_vfabs_v_f32m2(inputs, vl);

    vbool16_t reciprocal_mask_b16 = __riscv_vmfgt_vf_f32m2_b16(values_f32m2, 1.0f, vl);
    vfloat32m2_t reciprocal_values_f32m2 = nk_f32m2_reciprocal_rvv_(values_f32m2, vl);
    values_f32m2 = __riscv_vmerge_vvm_f32m2(values_f32m2, reciprocal_values_f32m2, reciprocal_mask_b16, vl);

    vfloat32m2_t squared_f32m2 = __riscv_vfmul_vv_f32m2(values_f32m2, values_f32m2, vl);
    vfloat32m2_t cubed_f32m2 = __riscv_vfmul_vv_f32m2(values_f32m2, squared_f32m2, vl);

    vfloat32m2_t poly_f32m2 = __riscv_vfmv_v_f_f32m2(c1, vl);
    poly_f32m2 = __riscv_vfmadd_vv_f32m2(poly_f32m2, squared_f32m2, __riscv_vfmv_v_f_f32m2(c2, vl), vl);
    poly_f32m2 = __riscv_vfmadd_vv_f32m2(poly_f32m2, squared_f32m2, __riscv_vfmv_v_f_f32m2(c3, vl), vl);
    poly_f32m2 = __riscv_vfmadd_vv_f32m2(poly_f32m2, squared_f32m2, __riscv_vfmv_v_f_f32m2(c4, vl), vl);
    poly_f32m2 = __riscv_vfmadd_vv_f32m2(poly_f32m2, squared_f32m2, __riscv_vfmv_v_f_f32m2(c5, vl), vl);
    poly_f32m2 = __riscv_vfmadd_vv_f32m2(poly_f32m2, squared_f32m2, __riscv_vfmv_v_f_f32m2(c6, vl), vl);
    poly_f32m2 = __riscv_vfmadd_vv_f32m2(poly_f32m2, squared_f32m2, __riscv_vfmv_v_f_f32m2(c7, vl), vl);
    poly_f32m2 = __riscv_vfmadd_vv_f32m2(poly_f32m2, squared_f32m2, __riscv_vfmv_v_f_f32m2(c8, vl), vl);

    vfloat32m2_t result_f32m2 = __riscv_vfmacc_vv_f32m2(values_f32m2, cubed_f32m2, poly_f32m2, vl);

    result_f32m2 = __riscv_vfrsub_vf_f32m2_mu(reciprocal_mask_b16, result_f32m2, result_f32m2, 1.5707963267948966f, vl);

    result_f32m2 = __riscv_vfneg_v_f32m2_mu(negative_mask_b16, result_f32m2, result_f32m2, vl);
    return result_f32m2;
}

NK_PUBLIC void nk_each_sin_f32_rvv(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    for (nk_size_t vector_length; n > 0; n -= vector_length, ins += vector_length, outs += vector_length) {
        vector_length = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t angles_f32m4 = __riscv_vle32_v_f32m4(ins, vector_length);
        vfloat32m4_t results_f32m4 = nk_f32m4_sin_rvv_(angles_f32m4, vector_length);
        __riscv_vse32_v_f32m4(outs, results_f32m4, vector_length);
    }
}

NK_PUBLIC void nk_each_cos_f32_rvv(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    for (nk_size_t vector_length; n > 0; n -= vector_length, ins += vector_length, outs += vector_length) {
        vector_length = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t angles_f32m4 = __riscv_vle32_v_f32m4(ins, vector_length);
        vfloat32m4_t results_f32m4 = nk_f32m4_cos_rvv_(angles_f32m4, vector_length);
        __riscv_vse32_v_f32m4(outs, results_f32m4, vector_length);
    }
}

NK_PUBLIC void nk_each_atan_f32_rvv(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    for (nk_size_t vector_length; n > 0; n -= vector_length, ins += vector_length, outs += vector_length) {
        vector_length = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t values_f32m4 = __riscv_vle32_v_f32m4(ins, vector_length);
        vfloat32m4_t results_f32m4 = nk_f32m4_atan_rvv_(values_f32m4, vector_length);
        __riscv_vse32_v_f32m4(outs, results_f32m4, vector_length);
    }
}

NK_PUBLIC void nk_each_sin_f64_rvv(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    for (nk_size_t vector_length; n > 0; n -= vector_length, ins += vector_length, outs += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        vfloat64m4_t angles_f64m4 = __riscv_vle64_v_f64m4(ins, vector_length);
        vfloat64m4_t results_f64m4 = nk_f64m4_sin_rvv_(angles_f64m4, vector_length);
        __riscv_vse64_v_f64m4(outs, results_f64m4, vector_length);
    }
}

NK_PUBLIC void nk_each_cos_f64_rvv(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    for (nk_size_t vector_length; n > 0; n -= vector_length, ins += vector_length, outs += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        vfloat64m4_t angles_f64m4 = __riscv_vle64_v_f64m4(ins, vector_length);
        vfloat64m4_t results_f64m4 = nk_f64m4_cos_rvv_(angles_f64m4, vector_length);
        __riscv_vse64_v_f64m4(outs, results_f64m4, vector_length);
    }
}

NK_PUBLIC void nk_each_atan_f64_rvv(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    for (nk_size_t vector_length; n > 0; n -= vector_length, ins += vector_length, outs += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        vfloat64m4_t values_f64m4 = __riscv_vle64_v_f64m4(ins, vector_length);
        vfloat64m4_t results_f64m4 = nk_f64m4_atan_rvv_(values_f64m4, vector_length);
        __riscv_vse64_v_f64m4(outs, results_f64m4, vector_length);
    }
}

NK_PUBLIC void nk_each_sin_f16_rvv(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
    for (nk_size_t vector_length; n > 0; n -= vector_length, ins += vector_length, outs += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        vuint16m1_t f16_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)ins, vector_length);
        vfloat32m2_t values_f32m2 = nk_f16m1_to_f32m2_rvv_(f16_u16m1, vector_length);
        vfloat32m2_t results_f32m2 = nk_f32m2_sin_rvv_(values_f32m2, vector_length);
        vuint16m1_t f16_results = nk_f32m2_to_f16m1_rvv_(results_f32m2, vector_length);
        __riscv_vse16_v_u16m1((nk_u16_t *)outs, f16_results, vector_length);
    }
}

NK_PUBLIC void nk_each_cos_f16_rvv(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
    for (nk_size_t vector_length; n > 0; n -= vector_length, ins += vector_length, outs += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        vuint16m1_t f16_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)ins, vector_length);
        vfloat32m2_t values_f32m2 = nk_f16m1_to_f32m2_rvv_(f16_u16m1, vector_length);
        vfloat32m2_t results_f32m2 = nk_f32m2_cos_rvv_(values_f32m2, vector_length);
        vuint16m1_t f16_results = nk_f32m2_to_f16m1_rvv_(results_f32m2, vector_length);
        __riscv_vse16_v_u16m1((nk_u16_t *)outs, f16_results, vector_length);
    }
}

NK_PUBLIC void nk_each_atan_f16_rvv(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
    for (nk_size_t vector_length; n > 0; n -= vector_length, ins += vector_length, outs += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        vuint16m1_t f16_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)ins, vector_length);
        vfloat32m2_t values_f32m2 = nk_f16m1_to_f32m2_rvv_(f16_u16m1, vector_length);
        vfloat32m2_t results_f32m2 = nk_f32m2_atan_rvv_(values_f32m2, vector_length);
        vuint16m1_t f16_results = nk_f32m2_to_f16m1_rvv_(results_f32m2, vector_length);
        __riscv_vse16_v_u16m1((nk_u16_t *)outs, f16_results, vector_length);
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

#endif // NK_TARGET_RVV
#endif // NK_TARGET_RISCV_
#endif // NK_TRIGONOMETRY_RVV_H
