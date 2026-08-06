/**
 *  @brief SIMD-accelerated Trigonometric Functions for WASM.
 *  @file include/numkong/trigonometry/v128relaxed.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/trigonometry.h
 *  @see https://sleef.org
 *
 *  Implements sin, cos, atan, atan2 for f32x4 and f64x2 using minimax polynomial approximations.
 *  F32 sin/cos use 3-term polynomials with Cody-Waite range reduction; f64 uses 8-term Estrin.
 *  F32 atan uses an 8-term Horner scheme; f64 atan uses 19 terms. Atan2 adds quadrant correction.
 *
 *  Polynomial chains rely on relaxed FMA (`relaxed_madd`/`relaxed_nmadd`) for throughput.
 *  Range reduction uses `nearest` rounding (avoiding int-float roundtrips); `trunc_sat` only
 *  for integer parity checks. F32 atan uses division for reciprocal (no approximate reciprocal
 *  available in WASM). All type punning is free since `v128_t` is untyped.
 *
 *  @section wasm_trig_instructions Key WASM SIMD Instructions
 *
 *      Intrinsic                               Operation
 *      wasm_f32x4_relaxed_madd(a, b, c)        a*b + c  (relaxed FMA)
 *      wasm_f32x4_relaxed_nmadd(a, b, c)       -(a*b) + c  (relaxed FNMA)
 *      wasm_f64x2_relaxed_madd(a, b, c)        a*b + c  (relaxed FMA)
 *      wasm_f64x2_relaxed_nmadd(a, b, c)       -(a*b) + c  (relaxed FNMA)
 *      wasm_f32x4_nearest(a)                   Round to nearest integer
 *      wasm_f64x2_nearest(a)                   Round to nearest integer
 *      wasm_v128_bitselect(true, false, mask)  Bitwise select
 *      wasm_i32x4_relaxed_laneselect(a, b, m)  Lane select (1 instr vs 3 on x86)
 *      wasm_i64x2_relaxed_laneselect(a, b, m)  Lane select for f64 masks
 *      wasm_i32x4_relaxed_trunc_f32x4(a)       Truncate without sat fixup (1 vs 7 on x86)
 *      wasm_i32x4_relaxed_trunc_f64x2_zero(a)  Truncate f64→i32 without sat fixup (1 vs 7 on x86)
 */
#ifndef NK_TRIGONOMETRY_V128RELAXED_H
#define NK_TRIGONOMETRY_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_partial_load_b32x4_serial_`, `nk_partial_load_b64x2_serial_`, `nk_partial_store_b32x4_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/*  WASM Relaxed SIMD trigonometry kernels (4-way f32, 2-way f64)
 *  These implement polynomial approximations using 128-bit WASM SIMD vectors.
 */

NK_INTERNAL v128_t nk_f32x4_sin_v128relaxed_(v128_t const angles_radians) {
    // Constants for argument reduction
    v128_t const pi_f32x4 = wasm_f32x4_splat(3.14159265358979323846f);
    v128_t const pi_reciprocal_f32x4 = wasm_f32x4_splat(0.31830988618379067154f);
    v128_t const coeff_5_f32x4 = wasm_f32x4_splat(-0.0001881748176f);
    v128_t const coeff_3_f32x4 = wasm_f32x4_splat(+0.008323502727f);
    v128_t const coeff_1_f32x4 = wasm_f32x4_splat(-0.1666651368f);

    // Compute (multiples_of_pi_f32x4) = round(angle / pi_f32x4) using nearest rounding
    v128_t quotients_f32x4 = wasm_f32x4_mul(angles_radians, pi_reciprocal_f32x4);
    v128_t rounded_quotients_f32x4 = wasm_f32x4_nearest(quotients_f32x4);
    // relaxed_trunc: 1 instruction (cvttps2dq) vs 7 (with NaN/overflow fixup) on x86.
    // Safe because rounded_quotients_f32x4 are small integers from nearest(), never NaN or out of i32 range.
    v128_t multiples_of_pi_f32x4 = wasm_i32x4_relaxed_trunc_f32x4(rounded_quotients_f32x4);

    // Reduce the angle: angle - rounded_quotients_f32x4 * pi_f32x4
    // vfmsq_f32(acc, a, b) = acc - a*b -> wasm_f32x4_relaxed_nmadd(a, b, acc)
    v128_t const angles_f32x4 = wasm_f32x4_relaxed_nmadd(rounded_quotients_f32x4, pi_f32x4, angles_radians);
    v128_t const angles_squared_f32x4 = wasm_f32x4_mul(angles_f32x4, angles_f32x4);
    v128_t const angles_cubed_f32x4 = wasm_f32x4_mul(angles_f32x4, angles_squared_f32x4);

    // Compute the polynomial approximation
    // vfmaq_f32(acc, a, b) = acc + a*b -> wasm_f32x4_relaxed_madd(a, b, acc)
    v128_t polynomials_f32x4 = coeff_5_f32x4;
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, angles_squared_f32x4, coeff_3_f32x4);
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, angles_squared_f32x4, coeff_1_f32x4);
    v128_t results_f32x4 = wasm_f32x4_relaxed_madd(angles_cubed_f32x4, polynomials_f32x4, angles_f32x4);

    // If multiples_of_pi_f32x4 is odd, flip the sign
    v128_t parity_i32x4 = wasm_v128_and(multiples_of_pi_f32x4, wasm_i32x4_splat(1));
    v128_t odd_mask_i32x4 = wasm_i32x4_eq(parity_i32x4, wasm_i32x4_splat(1));
    v128_t negated_f32x4 = wasm_f32x4_neg(results_f32x4);
    // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    results_f32x4 = wasm_i32x4_relaxed_laneselect(negated_f32x4, results_f32x4, odd_mask_i32x4);
    return results_f32x4;
}

NK_INTERNAL v128_t nk_f32x4_cos_v128relaxed_(v128_t const angles_radians) {
    // Constants for argument reduction
    v128_t const pi_f32x4 = wasm_f32x4_splat(3.14159265358979323846f);
    v128_t const pi_half_f32x4 = wasm_f32x4_splat(1.57079632679489661923f);
    v128_t const pi_reciprocal_f32x4 = wasm_f32x4_splat(0.31830988618379067154f);
    v128_t const coeff_5_f32x4 = wasm_f32x4_splat(-0.0001881748176f);
    v128_t const coeff_3_f32x4 = wasm_f32x4_splat(+0.008323502727f);
    v128_t const coeff_1_f32x4 = wasm_f32x4_splat(-0.1666651368f);

    // Compute round((angle / pi_f32x4) - 0.5)
    v128_t const neg_half_f32x4 = wasm_f32x4_splat(-0.5f);
    v128_t quotients_f32x4 = wasm_f32x4_relaxed_madd(angles_radians, pi_reciprocal_f32x4, neg_half_f32x4);
    v128_t rounded_quotients_f32x4 = wasm_f32x4_nearest(quotients_f32x4);
    // relaxed_trunc: 1 instruction (cvttps2dq) vs 7 (with NaN/overflow fixup) on x86.
    // Safe because rounded_quotients_f32x4 are small integers from nearest(), never NaN or out of i32 range.
    v128_t multiples_of_pi_f32x4 = wasm_i32x4_relaxed_trunc_f32x4(rounded_quotients_f32x4);

    // Reduce the angle: (angle - pi_f32x4/2) - rounded_quotients_f32x4 * pi_f32x4
    v128_t shifted_f32x4 = wasm_f32x4_sub(angles_radians, pi_half_f32x4);
    v128_t const angles_f32x4 = wasm_f32x4_relaxed_nmadd(rounded_quotients_f32x4, pi_f32x4, shifted_f32x4);
    v128_t const angles_squared_f32x4 = wasm_f32x4_mul(angles_f32x4, angles_f32x4);
    v128_t const angles_cubed_f32x4 = wasm_f32x4_mul(angles_f32x4, angles_squared_f32x4);

    // Compute the polynomial approximation
    v128_t polynomials_f32x4 = coeff_5_f32x4;
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, angles_squared_f32x4, coeff_3_f32x4);
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, angles_squared_f32x4, coeff_1_f32x4);
    v128_t results_f32x4 = wasm_f32x4_relaxed_madd(angles_cubed_f32x4, polynomials_f32x4, angles_f32x4);

    // If multiples_of_pi_f32x4 is even, flip the sign
    v128_t parity_i32x4 = wasm_v128_and(multiples_of_pi_f32x4, wasm_i32x4_splat(1));
    v128_t even_mask_i32x4 = wasm_i32x4_eq(parity_i32x4, wasm_i32x4_splat(0));
    v128_t negated_f32x4 = wasm_f32x4_neg(results_f32x4);
    // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    results_f32x4 = wasm_i32x4_relaxed_laneselect(negated_f32x4, results_f32x4, even_mask_i32x4);
    return results_f32x4;
}

NK_INTERNAL v128_t nk_f32x4_atan_v128relaxed_(v128_t const inputs) {
    // Polynomial coefficients for atan approximation (8 terms)
    v128_t const coeff_8_f32x4 = wasm_f32x4_splat(-0.333331018686294555664062f);
    v128_t const coeff_7_f32x4 = wasm_f32x4_splat(+0.199926957488059997558594f);
    v128_t const coeff_6_f32x4 = wasm_f32x4_splat(-0.142027363181114196777344f);
    v128_t const coeff_5_f32x4 = wasm_f32x4_splat(+0.106347933411598205566406f);
    v128_t const coeff_4_f32x4 = wasm_f32x4_splat(-0.0748900920152664184570312f);
    v128_t const coeff_3_f32x4 = wasm_f32x4_splat(+0.0425049886107444763183594f);
    v128_t const coeff_2_f32x4 = wasm_f32x4_splat(-0.0159569028764963150024414f);
    v128_t const coeff_1_f32x4 = wasm_f32x4_splat(+0.00282363896258175373077393f);
    v128_t const half_pi_f32x4 = wasm_f32x4_splat(1.5707963267948966f);

    // Detect negative values_f32x4 and take absolute value
    v128_t const zeros_f32x4 = wasm_f32x4_splat(0);
    v128_t negative_mask_f32x4 = wasm_f32x4_lt(inputs, zeros_f32x4);
    v128_t values_f32x4 = wasm_f32x4_abs(inputs);

    // Check if values_f32x4 > 1 (need reciprocal)
    v128_t reciprocal_mask_f32x4 = wasm_f32x4_gt(values_f32x4, wasm_f32x4_splat(1.0f));

    // No fast reciprocal in WASM — use division
    v128_t recip_f32x4 = wasm_f32x4_div(wasm_f32x4_splat(1.0f), values_f32x4);
    // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros_f32x4 per lane).
    values_f32x4 = wasm_i32x4_relaxed_laneselect(recip_f32x4, values_f32x4, reciprocal_mask_f32x4);

    // Compute powers
    v128_t const values_squared_f32x4 = wasm_f32x4_mul(values_f32x4, values_f32x4);
    v128_t const values_cubed_f32x4 = wasm_f32x4_mul(values_f32x4, values_squared_f32x4);

    // Polynomial evaluation using Horner's method
    v128_t polynomials_f32x4 = coeff_1_f32x4;
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, values_squared_f32x4, coeff_2_f32x4);
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, values_squared_f32x4, coeff_3_f32x4);
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, values_squared_f32x4, coeff_4_f32x4);
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, values_squared_f32x4, coeff_5_f32x4);
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, values_squared_f32x4, coeff_6_f32x4);
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, values_squared_f32x4, coeff_7_f32x4);
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, values_squared_f32x4, coeff_8_f32x4);

    // Compute result_f32x4: atan(x) ~ x + x^3 * P(x^2)
    v128_t result_f32x4 = wasm_f32x4_relaxed_madd(values_cubed_f32x4, polynomials_f32x4, values_f32x4);

    // Adjust for reciprocal: result_f32x4 = pi/2 - result_f32x4
    v128_t adjusted_f32x4 = wasm_f32x4_sub(half_pi_f32x4, result_f32x4);
    // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros_f32x4 per lane).
    result_f32x4 = wasm_i32x4_relaxed_laneselect(adjusted_f32x4, result_f32x4, reciprocal_mask_f32x4);

    // Adjust for negative: result_f32x4 = -result_f32x4
    v128_t negated_f32x4 = wasm_f32x4_neg(result_f32x4);
    // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros_f32x4 per lane).
    result_f32x4 = wasm_i32x4_relaxed_laneselect(negated_f32x4, result_f32x4, negative_mask_f32x4);
    return result_f32x4;
}

NK_INTERNAL v128_t nk_f32x4_atan2_v128relaxed_(v128_t const ys_inputs, v128_t const xs_inputs) {
    // Polynomial coefficients (same as atan)
    v128_t const coeff_8_f32x4 = wasm_f32x4_splat(-0.333331018686294555664062f);
    v128_t const coeff_7_f32x4 = wasm_f32x4_splat(+0.199926957488059997558594f);
    v128_t const coeff_6_f32x4 = wasm_f32x4_splat(-0.142027363181114196777344f);
    v128_t const coeff_5_f32x4 = wasm_f32x4_splat(+0.106347933411598205566406f);
    v128_t const coeff_4_f32x4 = wasm_f32x4_splat(-0.0748900920152664184570312f);
    v128_t const coeff_3_f32x4 = wasm_f32x4_splat(+0.0425049886107444763183594f);
    v128_t const coeff_2_f32x4 = wasm_f32x4_splat(-0.0159569028764963150024414f);
    v128_t const coeff_1_f32x4 = wasm_f32x4_splat(+0.00282363896258175373077393f);
    v128_t const pi_f32x4 = wasm_f32x4_splat(3.14159265358979323846f);
    v128_t const half_pi_f32x4 = wasm_f32x4_splat(1.5707963267948966f);
    v128_t const zeros_f32x4 = wasm_f32x4_splat(0);

    // Quadrant adjustments - take absolute values
    v128_t xs_negative_mask_f32x4 = wasm_f32x4_lt(xs_inputs, zeros_f32x4);
    v128_t xs_f32x4 = wasm_f32x4_abs(xs_inputs);
    v128_t ys_f32x4 = wasm_f32x4_abs(ys_inputs);

    // Ensure proper fraction where numerator < denominator
    v128_t swap_mask_f32x4 = wasm_f32x4_gt(ys_f32x4, xs_f32x4);
    v128_t temps_f32x4 = xs_f32x4;
    // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros_f32x4 per lane).
    xs_f32x4 = wasm_i32x4_relaxed_laneselect(ys_f32x4, xs_f32x4, swap_mask_f32x4);
    ys_f32x4 = wasm_i32x4_relaxed_laneselect(wasm_f32x4_neg(temps_f32x4), ys_f32x4, swap_mask_f32x4);

    // Division for ratio_f32x4: ratio_f32x4 = ys_f32x4 / xs_f32x4
    v128_t const ratio_f32x4 = wasm_f32x4_div(ys_f32x4, xs_f32x4);
    v128_t const ratio_squared_f32x4 = wasm_f32x4_mul(ratio_f32x4, ratio_f32x4);
    v128_t const ratio_cubed_f32x4 = wasm_f32x4_mul(ratio_f32x4, ratio_squared_f32x4);

    // Polynomial evaluation using Horner's method
    v128_t polynomials_f32x4 = coeff_1_f32x4;
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, ratio_squared_f32x4, coeff_2_f32x4);
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, ratio_squared_f32x4, coeff_3_f32x4);
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, ratio_squared_f32x4, coeff_4_f32x4);
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, ratio_squared_f32x4, coeff_5_f32x4);
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, ratio_squared_f32x4, coeff_6_f32x4);
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, ratio_squared_f32x4, coeff_7_f32x4);
    polynomials_f32x4 = wasm_f32x4_relaxed_madd(polynomials_f32x4, ratio_squared_f32x4, coeff_8_f32x4);

    // Compute the result
    v128_t results_f32x4 = wasm_f32x4_relaxed_madd(ratio_cubed_f32x4, polynomials_f32x4, ratio_f32x4);

    // Compute quadrant_f32x4 value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    v128_t quadrant_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t neg_two_f32x4 = wasm_f32x4_splat(-2.0f);
    // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros_f32x4 per lane).
    quadrant_f32x4 = wasm_i32x4_relaxed_laneselect(neg_two_f32x4, quadrant_f32x4, xs_negative_mask_f32x4);
    v128_t quadrant_incremented_f32x4 = wasm_f32x4_add(quadrant_f32x4, wasm_f32x4_splat(1.0f));
    quadrant_f32x4 = wasm_i32x4_relaxed_laneselect(quadrant_incremented_f32x4, quadrant_f32x4, swap_mask_f32x4);

    // Adjust for quadrant_f32x4: result += quadrant_f32x4 * pi_f32x4/2
    results_f32x4 = wasm_f32x4_relaxed_madd(quadrant_f32x4, half_pi_f32x4, results_f32x4);

    // Transfer sign from x and y by XOR with sign bits
    v128_t sign_mask_f32x4 = wasm_f32x4_splat(-0.0f);
    v128_t xs_sign_f32x4 = wasm_v128_and(xs_inputs, sign_mask_f32x4);
    v128_t ys_sign_f32x4 = wasm_v128_and(ys_inputs, sign_mask_f32x4);
    results_f32x4 = wasm_v128_xor(results_f32x4, xs_sign_f32x4);
    results_f32x4 = wasm_v128_xor(results_f32x4, ys_sign_f32x4);

    return results_f32x4;
}

NK_INTERNAL v128_t nk_f64x2_sin_v128relaxed_(v128_t const angles_radians) {
    // Constants for argument reduction
    v128_t const pi_high_f64x2 = wasm_f64x2_splat(3.141592653589793116);
    v128_t const pi_low_f64x2 = wasm_f64x2_splat(1.2246467991473532072e-16);
    v128_t const pi_reciprocal_f64x2 = wasm_f64x2_splat(0.31830988618379067154);

    // Polynomial coefficients for sine approximation
    v128_t const coeff_0_f64x2 = wasm_f64x2_splat(+0.00833333333333332974823815);
    v128_t const coeff_1_f64x2 = wasm_f64x2_splat(-0.000198412698412696162806809);
    v128_t const coeff_2_f64x2 = wasm_f64x2_splat(+2.75573192239198747630416e-06);
    v128_t const coeff_3_f64x2 = wasm_f64x2_splat(-2.50521083763502045810755e-08);
    v128_t const coeff_4_f64x2 = wasm_f64x2_splat(+1.60590430605664501629054e-10);
    v128_t const coeff_5_f64x2 = wasm_f64x2_splat(-7.64712219118158833288484e-13);
    v128_t const coeff_6_f64x2 = wasm_f64x2_splat(+2.81009972710863200091251e-15);
    v128_t const coeff_7_f64x2 = wasm_f64x2_splat(-7.97255955009037868891952e-18);
    v128_t const coeff_8_f64x2 = wasm_f64x2_splat(-0.166666666666666657414808);

    // Compute round(angle / pi)
    v128_t const quotients_f64x2 = wasm_f64x2_mul(angles_radians, pi_reciprocal_f64x2);
    v128_t rounded_quotients_f64x2 = wasm_f64x2_nearest(quotients_f64x2);
    // relaxed_trunc: 1 instruction (cvttpd2dq) vs 7 (with NaN/overflow fixup) on x86.
    // Safe because rounded_quotients_f64x2 are small integers from nearest(), never NaN or out of i32 range.
    v128_t multiples_i32_f64x2 = wasm_i32x4_relaxed_trunc_f64x2_zero(rounded_quotients_f64x2);

    // Two-step Cody-Waite reduction: angle - rounded * pi_high_f64x2 - rounded * pi_low_f64x2
    v128_t angles_f64x2 = angles_radians;
    angles_f64x2 = wasm_f64x2_relaxed_nmadd(rounded_quotients_f64x2, pi_high_f64x2, angles_f64x2);
    angles_f64x2 = wasm_f64x2_relaxed_nmadd(rounded_quotients_f64x2, pi_low_f64x2, angles_f64x2);

    // Check parity in i32, then widen to i64 mask for laneselect
    v128_t parity_i32_i32x4 = wasm_v128_and(multiples_i32_f64x2, wasm_i32x4_splat(1));
    v128_t odd_i32_i32x4 = wasm_i32x4_eq(parity_i32_i32x4, wasm_i32x4_splat(1));
    // Widen: lane0 of i32 -> lanes 0-1 of i64, lane1 -> lanes 2-3
    // Shuffle i32 lanes [0,0,1,1] to broadcast each i32 parity into both halves of each i64
    v128_t odd_mask_i32x4 = wasm_i32x4_shuffle(odd_i32_i32x4, odd_i32_i32x4, 0, 0, 1, 1);
    v128_t negated_angles_f64x2 = wasm_f64x2_neg(angles_f64x2);
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is lane-granular at i64 width (all-ones or all-zeros per 64-bit lane).
    angles_f64x2 = wasm_i64x2_relaxed_laneselect(negated_angles_f64x2, angles_f64x2, odd_mask_i32x4);

    v128_t const angles_squared_f64x2 = wasm_f64x2_mul(angles_f64x2, angles_f64x2);
    v128_t const angles_cubed_f64x2 = wasm_f64x2_mul(angles_f64x2, angles_squared_f64x2);
    v128_t const angles_quadratic_f64x2 = wasm_f64x2_mul(angles_squared_f64x2, angles_squared_f64x2);
    v128_t const angles_octic_f64x2 = wasm_f64x2_mul(angles_quadratic_f64x2, angles_quadratic_f64x2);

    // Compute polynomial terms using Estrin's scheme for better ILP
    v128_t const poly_67_f64x2 = wasm_f64x2_relaxed_madd(angles_squared_f64x2, coeff_7_f64x2, coeff_6_f64x2);
    v128_t const poly_45_f64x2 = wasm_f64x2_relaxed_madd(angles_squared_f64x2, coeff_5_f64x2, coeff_4_f64x2);
    v128_t const poly_4567_f64x2 = wasm_f64x2_relaxed_madd(angles_quadratic_f64x2, poly_67_f64x2, poly_45_f64x2);

    v128_t const poly_23_f64x2 = wasm_f64x2_relaxed_madd(angles_squared_f64x2, coeff_3_f64x2, coeff_2_f64x2);
    v128_t const poly_01_f64x2 = wasm_f64x2_relaxed_madd(angles_squared_f64x2, coeff_1_f64x2, coeff_0_f64x2);
    v128_t const poly_0123_f64x2 = wasm_f64x2_relaxed_madd(angles_quadratic_f64x2, poly_23_f64x2, poly_01_f64x2);

    // Combine polynomial terms
    v128_t results_f64x2 = wasm_f64x2_relaxed_madd(angles_octic_f64x2, poly_4567_f64x2, poly_0123_f64x2);
    results_f64x2 = wasm_f64x2_relaxed_madd(results_f64x2, angles_squared_f64x2, coeff_8_f64x2);
    results_f64x2 = wasm_f64x2_relaxed_madd(results_f64x2, angles_cubed_f64x2, angles_f64x2);

    // Handle zero input (preserve sign of zero)
    v128_t const non_zero_mask_f64x2 = wasm_f64x2_eq(angles_radians, wasm_f64x2_splat(0));
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    results_f64x2 = wasm_i64x2_relaxed_laneselect(angles_radians, results_f64x2, non_zero_mask_f64x2);
    return results_f64x2;
}

NK_INTERNAL v128_t nk_f64x2_cos_v128relaxed_(v128_t const angles_radians) {
    // Constants for argument reduction
    v128_t const pi_high_half_f64x2 = wasm_f64x2_splat(3.141592653589793116 * 0.5);
    v128_t const pi_low_half_f64x2 = wasm_f64x2_splat(1.2246467991473532072e-16 * 0.5);
    v128_t const pi_reciprocal_f64x2 = wasm_f64x2_splat(0.31830988618379067154);

    // Polynomial coefficients for cosine approximation
    v128_t const coeff_0_f64x2 = wasm_f64x2_splat(+0.00833333333333332974823815);
    v128_t const coeff_1_f64x2 = wasm_f64x2_splat(-0.000198412698412696162806809);
    v128_t const coeff_2_f64x2 = wasm_f64x2_splat(+2.75573192239198747630416e-06);
    v128_t const coeff_3_f64x2 = wasm_f64x2_splat(-2.50521083763502045810755e-08);
    v128_t const coeff_4_f64x2 = wasm_f64x2_splat(+1.60590430605664501629054e-10);
    v128_t const coeff_5_f64x2 = wasm_f64x2_splat(-7.64712219118158833288484e-13);
    v128_t const coeff_6_f64x2 = wasm_f64x2_splat(+2.81009972710863200091251e-15);
    v128_t const coeff_7_f64x2 = wasm_f64x2_splat(-7.97255955009037868891952e-18);
    v128_t const coeff_8_f64x2 = wasm_f64x2_splat(-0.166666666666666657414808);

    // Compute 2 * round(angle / pi - 0.5) + 1
    v128_t const neg_half_f64x2 = wasm_f64x2_splat(-0.5);
    v128_t const quotients_f64x2 = wasm_f64x2_relaxed_madd(angles_radians, pi_reciprocal_f64x2, neg_half_f64x2);
    v128_t const rounded_f64x2 = wasm_f64x2_nearest(quotients_f64x2);
    v128_t const rounded_quotients_f64x2 = wasm_f64x2_relaxed_madd(wasm_f64x2_splat(2.0), rounded_f64x2,
                                                                   wasm_f64x2_splat(1.0));
    // relaxed_trunc: 1 instruction (cvttpd2dq) vs 7 (with NaN/overflow fixup) on x86.
    // Safe because rounded_quotients_f64x2 are small integers from nearest(), never NaN or out of i32 range.
    v128_t quotients_i32_f64x2 = wasm_i32x4_relaxed_trunc_f64x2_zero(rounded_quotients_f64x2);

    // Two-step Cody-Waite reduction
    v128_t angles_f64x2 = angles_radians;
    angles_f64x2 = wasm_f64x2_relaxed_nmadd(rounded_quotients_f64x2, pi_high_half_f64x2, angles_f64x2);
    angles_f64x2 = wasm_f64x2_relaxed_nmadd(rounded_quotients_f64x2, pi_low_half_f64x2, angles_f64x2);

    // Check bit 1 in i32, then widen to i64 mask for laneselect
    v128_t bit2_i32_i32x4 = wasm_v128_and(quotients_i32_f64x2, wasm_i32x4_splat(2));
    v128_t flip_i32_i32x4 = wasm_i32x4_eq(bit2_i32_i32x4, wasm_i32x4_splat(0));
    v128_t flip_mask_i32x4 = wasm_i32x4_shuffle(flip_i32_i32x4, flip_i32_i32x4, 0, 0, 1, 1);
    v128_t negated_angles_f64x2 = wasm_f64x2_neg(angles_f64x2);
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is lane-granular at i64 width (all-ones or all-zeros per 64-bit lane).
    angles_f64x2 = wasm_i64x2_relaxed_laneselect(negated_angles_f64x2, angles_f64x2, flip_mask_i32x4);

    v128_t const angles_squared_f64x2 = wasm_f64x2_mul(angles_f64x2, angles_f64x2);
    v128_t const angles_cubed_f64x2 = wasm_f64x2_mul(angles_f64x2, angles_squared_f64x2);
    v128_t const angles_quadratic_f64x2 = wasm_f64x2_mul(angles_squared_f64x2, angles_squared_f64x2);
    v128_t const angles_octic_f64x2 = wasm_f64x2_mul(angles_quadratic_f64x2, angles_quadratic_f64x2);

    // Compute polynomial terms using Estrin's scheme
    v128_t const poly_67_f64x2 = wasm_f64x2_relaxed_madd(angles_squared_f64x2, coeff_7_f64x2, coeff_6_f64x2);
    v128_t const poly_45_f64x2 = wasm_f64x2_relaxed_madd(angles_squared_f64x2, coeff_5_f64x2, coeff_4_f64x2);
    v128_t const poly_4567_f64x2 = wasm_f64x2_relaxed_madd(angles_quadratic_f64x2, poly_67_f64x2, poly_45_f64x2);

    v128_t const poly_23_f64x2 = wasm_f64x2_relaxed_madd(angles_squared_f64x2, coeff_3_f64x2, coeff_2_f64x2);
    v128_t const poly_01_f64x2 = wasm_f64x2_relaxed_madd(angles_squared_f64x2, coeff_1_f64x2, coeff_0_f64x2);
    v128_t const poly_0123_f64x2 = wasm_f64x2_relaxed_madd(angles_quadratic_f64x2, poly_23_f64x2, poly_01_f64x2);

    // Combine polynomial terms
    v128_t results_f64x2 = wasm_f64x2_relaxed_madd(angles_octic_f64x2, poly_4567_f64x2, poly_0123_f64x2);
    results_f64x2 = wasm_f64x2_relaxed_madd(results_f64x2, angles_squared_f64x2, coeff_8_f64x2);
    results_f64x2 = wasm_f64x2_relaxed_madd(results_f64x2, angles_cubed_f64x2, angles_f64x2);
    return results_f64x2;
}

NK_INTERNAL v128_t nk_f64x2_atan_v128relaxed_(v128_t const inputs) {
    // Polynomial coefficients for atan approximation (19 terms)
    v128_t const coeff_19_f64x2 = wasm_f64x2_splat(-1.88796008463073496563746e-05);
    v128_t const coeff_18_f64x2 = wasm_f64x2_splat(+0.000209850076645816976906797);
    v128_t const coeff_17_f64x2 = wasm_f64x2_splat(-0.00110611831486672482563471);
    v128_t const coeff_16_f64x2 = wasm_f64x2_splat(+0.00370026744188713119232403);
    v128_t const coeff_15_f64x2 = wasm_f64x2_splat(-0.00889896195887655491740809);
    v128_t const coeff_14_f64x2 = wasm_f64x2_splat(+0.016599329773529201970117);
    v128_t const coeff_13_f64x2 = wasm_f64x2_splat(-0.0254517624932312641616861);
    v128_t const coeff_12_f64x2 = wasm_f64x2_splat(+0.0337852580001353069993897);
    v128_t const coeff_11_f64x2 = wasm_f64x2_splat(-0.0407629191276836500001934);
    v128_t const coeff_10_f64x2 = wasm_f64x2_splat(+0.0466667150077840625632675);
    v128_t const coeff_9_f64x2 = wasm_f64x2_splat(-0.0523674852303482457616113);
    v128_t const coeff_8_f64x2 = wasm_f64x2_splat(+0.0587666392926673580854313);
    v128_t const coeff_7_f64x2 = wasm_f64x2_splat(-0.0666573579361080525984562);
    v128_t const coeff_6_f64x2 = wasm_f64x2_splat(+0.0769219538311769618355029);
    v128_t const coeff_5_f64x2 = wasm_f64x2_splat(-0.090908995008245008229153);
    v128_t const coeff_4_f64x2 = wasm_f64x2_splat(+0.111111105648261418443745);
    v128_t const coeff_3_f64x2 = wasm_f64x2_splat(-0.14285714266771329383765);
    v128_t const coeff_2_f64x2 = wasm_f64x2_splat(+0.199999999996591265594148);
    v128_t const coeff_1_f64x2 = wasm_f64x2_splat(-0.333333333333311110369124);
    v128_t const half_pi_f64x2 = wasm_f64x2_splat(1.5707963267948966);
    v128_t const zeros_f64x2 = wasm_f64x2_splat(0);

    // Detect negative and take absolute value
    v128_t negative_mask_f64x2 = wasm_f64x2_lt(inputs, zeros_f64x2);
    v128_t values_f64x2 = wasm_f64x2_abs(inputs);

    // Check if values_f64x2 > 1 (need reciprocal) - use division for f64 precision
    v128_t reciprocal_mask_f64x2 = wasm_f64x2_gt(values_f64x2, wasm_f64x2_splat(1.0));
    v128_t reciprocal_values_f64x2 = wasm_f64x2_div(wasm_f64x2_splat(1.0), values_f64x2);
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros_f64x2 per lane).
    values_f64x2 = wasm_i64x2_relaxed_laneselect(reciprocal_values_f64x2, values_f64x2, reciprocal_mask_f64x2);

    // Compute powers
    v128_t const values_squared_f64x2 = wasm_f64x2_mul(values_f64x2, values_f64x2);
    v128_t const values_cubed_f64x2 = wasm_f64x2_mul(values_f64x2, values_squared_f64x2);

    // Polynomial evaluation using Horner's method
    v128_t polynomials_f64x2 = coeff_19_f64x2;
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_18_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_17_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_16_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_15_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_14_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_13_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_12_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_11_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_10_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_9_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_8_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_7_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_6_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_5_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_4_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_3_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_2_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, values_squared_f64x2, coeff_1_f64x2);

    // Compute result_f64x2
    v128_t result_f64x2 = wasm_f64x2_relaxed_madd(values_cubed_f64x2, polynomials_f64x2, values_f64x2);

    // Adjust for reciprocal: result_f64x2 = pi/2 - result_f64x2
    v128_t adjusted_f64x2 = wasm_f64x2_sub(half_pi_f64x2, result_f64x2);
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros_f64x2 per lane).
    result_f64x2 = wasm_i64x2_relaxed_laneselect(adjusted_f64x2, result_f64x2, reciprocal_mask_f64x2);

    // Adjust for negative: result_f64x2 = -result_f64x2
    v128_t negated_f64x2 = wasm_f64x2_neg(result_f64x2);
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros_f64x2 per lane).
    result_f64x2 = wasm_i64x2_relaxed_laneselect(negated_f64x2, result_f64x2, negative_mask_f64x2);
    return result_f64x2;
}

NK_INTERNAL v128_t nk_f64x2_atan2_v128relaxed_(v128_t const ys_inputs, v128_t const xs_inputs) {
    // Polynomial coefficients (same as atan)
    v128_t const coeff_19_f64x2 = wasm_f64x2_splat(-1.88796008463073496563746e-05);
    v128_t const coeff_18_f64x2 = wasm_f64x2_splat(+0.000209850076645816976906797);
    v128_t const coeff_17_f64x2 = wasm_f64x2_splat(-0.00110611831486672482563471);
    v128_t const coeff_16_f64x2 = wasm_f64x2_splat(+0.00370026744188713119232403);
    v128_t const coeff_15_f64x2 = wasm_f64x2_splat(-0.00889896195887655491740809);
    v128_t const coeff_14_f64x2 = wasm_f64x2_splat(+0.016599329773529201970117);
    v128_t const coeff_13_f64x2 = wasm_f64x2_splat(-0.0254517624932312641616861);
    v128_t const coeff_12_f64x2 = wasm_f64x2_splat(+0.0337852580001353069993897);
    v128_t const coeff_11_f64x2 = wasm_f64x2_splat(-0.0407629191276836500001934);
    v128_t const coeff_10_f64x2 = wasm_f64x2_splat(+0.0466667150077840625632675);
    v128_t const coeff_9_f64x2 = wasm_f64x2_splat(-0.0523674852303482457616113);
    v128_t const coeff_8_f64x2 = wasm_f64x2_splat(+0.0587666392926673580854313);
    v128_t const coeff_7_f64x2 = wasm_f64x2_splat(-0.0666573579361080525984562);
    v128_t const coeff_6_f64x2 = wasm_f64x2_splat(+0.0769219538311769618355029);
    v128_t const coeff_5_f64x2 = wasm_f64x2_splat(-0.090908995008245008229153);
    v128_t const coeff_4_f64x2 = wasm_f64x2_splat(+0.111111105648261418443745);
    v128_t const coeff_3_f64x2 = wasm_f64x2_splat(-0.14285714266771329383765);
    v128_t const coeff_2_f64x2 = wasm_f64x2_splat(+0.199999999996591265594148);
    v128_t const coeff_1_f64x2 = wasm_f64x2_splat(-0.333333333333311110369124);
    v128_t const pi_f64x2 = wasm_f64x2_splat(3.14159265358979323846);
    v128_t const half_pi_f64x2 = wasm_f64x2_splat(1.5707963267948966);
    v128_t const zeros_f64x2 = wasm_f64x2_splat(0);

    // Quadrant adjustments - take absolute values
    v128_t xs_negative_mask_f64x2 = wasm_f64x2_lt(xs_inputs, zeros_f64x2);
    v128_t xs_f64x2 = wasm_f64x2_abs(xs_inputs);
    v128_t ys_f64x2 = wasm_f64x2_abs(ys_inputs);

    // Ensure proper fraction where numerator < denominator
    v128_t swap_mask_f64x2 = wasm_f64x2_gt(ys_f64x2, xs_f64x2);
    v128_t temps_f64x2 = xs_f64x2;
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros_f64x2 per lane).
    xs_f64x2 = wasm_i64x2_relaxed_laneselect(ys_f64x2, xs_f64x2, swap_mask_f64x2);
    ys_f64x2 = wasm_i64x2_relaxed_laneselect(wasm_f64x2_neg(temps_f64x2), ys_f64x2, swap_mask_f64x2);

    // Division for f64 precision
    v128_t const ratio_f64x2 = wasm_f64x2_div(ys_f64x2, xs_f64x2);
    v128_t const ratio_squared_f64x2 = wasm_f64x2_mul(ratio_f64x2, ratio_f64x2);
    v128_t const ratio_cubed_f64x2 = wasm_f64x2_mul(ratio_f64x2, ratio_squared_f64x2);

    // Polynomial evaluation using Horner's method
    v128_t polynomials_f64x2 = coeff_19_f64x2;
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_18_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_17_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_16_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_15_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_14_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_13_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_12_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_11_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_10_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_9_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_8_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_7_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_6_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_5_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_4_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_3_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_2_f64x2);
    polynomials_f64x2 = wasm_f64x2_relaxed_madd(polynomials_f64x2, ratio_squared_f64x2, coeff_1_f64x2);

    // Compute the result
    v128_t results_f64x2 = wasm_f64x2_relaxed_madd(ratio_cubed_f64x2, polynomials_f64x2, ratio_f64x2);

    // Compute quadrant_f64x2 value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    v128_t quadrant_f64x2 = wasm_f64x2_splat(0.0);
    v128_t neg_two_f64x2 = wasm_f64x2_splat(-2.0);
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros_f64x2 per lane).
    quadrant_f64x2 = wasm_i64x2_relaxed_laneselect(neg_two_f64x2, quadrant_f64x2, xs_negative_mask_f64x2);
    v128_t quadrant_incremented_f64x2 = wasm_f64x2_add(quadrant_f64x2, wasm_f64x2_splat(1.0));
    quadrant_f64x2 = wasm_i64x2_relaxed_laneselect(quadrant_incremented_f64x2, quadrant_f64x2, swap_mask_f64x2);

    // Adjust for quadrant_f64x2: result += quadrant_f64x2 * pi_f64x2/2
    results_f64x2 = wasm_f64x2_relaxed_madd(quadrant_f64x2, half_pi_f64x2, results_f64x2);

    // Transfer sign from x and y by XOR with sign bits
    v128_t sign_mask_f64x2 = wasm_f64x2_splat(-0.0);
    v128_t xs_sign_f64x2 = wasm_v128_and(xs_inputs, sign_mask_f64x2);
    v128_t ys_sign_f64x2 = wasm_v128_and(ys_inputs, sign_mask_f64x2);
    results_f64x2 = wasm_v128_xor(results_f64x2, xs_sign_f64x2);
    results_f64x2 = wasm_v128_xor(results_f64x2, ys_sign_f64x2);

    return results_f64x2;
}

/*  NK_PUBLIC wrappers — same loop+tail pattern as neon.h.
 *  Full loads use wasm_v128_load/wasm_v128_store.
 *  Tails use nk_partial_load_b32x4_serial_/nk_partial_store_b32x4_serial_ via .v128 union member.
 */

NK_PUBLIC void nk_each_sin_f32_v128relaxed(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        v128_t angles_f32x4 = wasm_v128_load(ins + i);
        v128_t results_f32x4 = nk_f32x4_sin_v128relaxed_(angles_f32x4);
        wasm_v128_store(outs + i, results_f32x4);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b128_vec_t angles_vec;
        nk_partial_load_b32x4_serial_(ins + i, &angles_vec, remaining);
        nk_b128_vec_t results_vec;
        results_vec.v128 = nk_f32x4_sin_v128relaxed_(angles_vec.v128);
        nk_partial_store_b32x4_serial_(&results_vec, outs + i, remaining);
    }
}

NK_PUBLIC void nk_each_cos_f32_v128relaxed(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        v128_t angles_f32x4 = wasm_v128_load(ins + i);
        v128_t results_f32x4 = nk_f32x4_cos_v128relaxed_(angles_f32x4);
        wasm_v128_store(outs + i, results_f32x4);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b128_vec_t angles_vec;
        nk_partial_load_b32x4_serial_(ins + i, &angles_vec, remaining);
        nk_b128_vec_t results_vec;
        results_vec.v128 = nk_f32x4_cos_v128relaxed_(angles_vec.v128);
        nk_partial_store_b32x4_serial_(&results_vec, outs + i, remaining);
    }
}

NK_PUBLIC void nk_each_atan_f32_v128relaxed(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        v128_t values_f32x4 = wasm_v128_load(ins + i);
        v128_t results_f32x4 = nk_f32x4_atan_v128relaxed_(values_f32x4);
        wasm_v128_store(outs + i, results_f32x4);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b128_vec_t values_vec;
        nk_partial_load_b32x4_serial_(ins + i, &values_vec, remaining);
        nk_b128_vec_t results_vec;
        results_vec.v128 = nk_f32x4_atan_v128relaxed_(values_vec.v128);
        nk_partial_store_b32x4_serial_(&results_vec, outs + i, remaining);
    }
}

NK_PUBLIC void nk_each_sin_f64_v128relaxed(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        v128_t angles_f64x2 = wasm_v128_load(ins + i);
        v128_t results_f64x2 = nk_f64x2_sin_v128relaxed_(angles_f64x2);
        wasm_v128_store(outs + i, results_f64x2);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b128_vec_t angles_vec;
        nk_partial_load_b64x2_serial_(ins + i, &angles_vec, remaining);
        nk_b128_vec_t results_vec;
        results_vec.v128 = nk_f64x2_sin_v128relaxed_(angles_vec.v128);
        nk_partial_store_b64x2_serial_(&results_vec, outs + i, remaining);
    }
}

NK_PUBLIC void nk_each_cos_f64_v128relaxed(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        v128_t angles_f64x2 = wasm_v128_load(ins + i);
        v128_t results_f64x2 = nk_f64x2_cos_v128relaxed_(angles_f64x2);
        wasm_v128_store(outs + i, results_f64x2);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b128_vec_t angles_vec;
        nk_partial_load_b64x2_serial_(ins + i, &angles_vec, remaining);
        nk_b128_vec_t results_vec;
        results_vec.v128 = nk_f64x2_cos_v128relaxed_(angles_vec.v128);
        nk_partial_store_b64x2_serial_(&results_vec, outs + i, remaining);
    }
}

NK_PUBLIC void nk_each_atan_f64_v128relaxed(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        v128_t values_f64x2 = wasm_v128_load(ins + i);
        v128_t results_f64x2 = nk_f64x2_atan_v128relaxed_(values_f64x2);
        wasm_v128_store(outs + i, results_f64x2);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b128_vec_t values_vec;
        nk_partial_load_b64x2_serial_(ins + i, &values_vec, remaining);
        nk_b128_vec_t results_vec;
        results_vec.v128 = nk_f64x2_atan_v128relaxed_(values_vec.v128);
        nk_partial_store_b64x2_serial_(&results_vec, outs + i, remaining);
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_TRIGONOMETRY_V128RELAXED_H
