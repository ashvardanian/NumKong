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
    v128_t const pi = wasm_f32x4_splat(3.14159265358979323846f);
    v128_t const pi_reciprocal = wasm_f32x4_splat(0.31830988618379067154f);
    v128_t const coeff_5 = wasm_f32x4_splat(-0.0001881748176f);
    v128_t const coeff_3 = wasm_f32x4_splat(+0.008323502727f);
    v128_t const coeff_1 = wasm_f32x4_splat(-0.1666651368f);

    // Compute (multiples_of_pi) = round(angle / pi) using nearest rounding
    v128_t quotients = wasm_f32x4_mul(angles_radians, pi_reciprocal);
    v128_t rounded_quotients = wasm_f32x4_nearest(quotients);
    // relaxed_trunc: 1 instruction (cvttps2dq) vs 7 (with NaN/overflow fixup) on x86.
    // Safe because rounded_quotients are small integers from nearest(), never NaN or out of i32 range.
    v128_t multiples_of_pi = wasm_i32x4_relaxed_trunc_f32x4(rounded_quotients);

    // Reduce the angle: angle - rounded_quotients * pi
    // vfmsq_f32(acc, a, b) = acc - a*b -> wasm_f32x4_relaxed_nmadd(a, b, acc)
    v128_t const angles = wasm_f32x4_relaxed_nmadd(rounded_quotients, pi, angles_radians);
    v128_t const angles_squared = wasm_f32x4_mul(angles, angles);
    v128_t const angles_cubed = wasm_f32x4_mul(angles, angles_squared);

    // Compute the polynomial approximation
    // vfmaq_f32(acc, a, b) = acc + a*b -> wasm_f32x4_relaxed_madd(a, b, acc)
    v128_t polynomials = coeff_5;
    polynomials = wasm_f32x4_relaxed_madd(polynomials, angles_squared, coeff_3);
    polynomials = wasm_f32x4_relaxed_madd(polynomials, angles_squared, coeff_1);
    v128_t results = wasm_f32x4_relaxed_madd(angles_cubed, polynomials, angles);

    // If multiples_of_pi is odd, flip the sign
    v128_t parity = wasm_v128_and(multiples_of_pi, wasm_i32x4_splat(1));
    v128_t odd_mask = wasm_i32x4_eq(parity, wasm_i32x4_splat(1));
    v128_t negated = wasm_f32x4_neg(results);
    // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    results = wasm_i32x4_relaxed_laneselect(negated, results, odd_mask);
    return results;
}

NK_INTERNAL v128_t nk_f32x4_cos_v128relaxed_(v128_t const angles_radians) {
    // Constants for argument reduction
    v128_t const pi = wasm_f32x4_splat(3.14159265358979323846f);
    v128_t const pi_half = wasm_f32x4_splat(1.57079632679489661923f);
    v128_t const pi_reciprocal = wasm_f32x4_splat(0.31830988618379067154f);
    v128_t const coeff_5 = wasm_f32x4_splat(-0.0001881748176f);
    v128_t const coeff_3 = wasm_f32x4_splat(+0.008323502727f);
    v128_t const coeff_1 = wasm_f32x4_splat(-0.1666651368f);

    // Compute round((angle / pi) - 0.5)
    v128_t const neg_half = wasm_f32x4_splat(-0.5f);
    v128_t quotients = wasm_f32x4_relaxed_madd(angles_radians, pi_reciprocal, neg_half);
    v128_t rounded_quotients = wasm_f32x4_nearest(quotients);
    // relaxed_trunc: 1 instruction (cvttps2dq) vs 7 (with NaN/overflow fixup) on x86.
    // Safe because rounded_quotients are small integers from nearest(), never NaN or out of i32 range.
    v128_t multiples_of_pi = wasm_i32x4_relaxed_trunc_f32x4(rounded_quotients);

    // Reduce the angle: (angle - pi/2) - rounded_quotients * pi
    v128_t shifted = wasm_f32x4_sub(angles_radians, pi_half);
    v128_t const angles = wasm_f32x4_relaxed_nmadd(rounded_quotients, pi, shifted);
    v128_t const angles_squared = wasm_f32x4_mul(angles, angles);
    v128_t const angles_cubed = wasm_f32x4_mul(angles, angles_squared);

    // Compute the polynomial approximation
    v128_t polynomials = coeff_5;
    polynomials = wasm_f32x4_relaxed_madd(polynomials, angles_squared, coeff_3);
    polynomials = wasm_f32x4_relaxed_madd(polynomials, angles_squared, coeff_1);
    v128_t results = wasm_f32x4_relaxed_madd(angles_cubed, polynomials, angles);

    // If multiples_of_pi is even, flip the sign
    v128_t parity = wasm_v128_and(multiples_of_pi, wasm_i32x4_splat(1));
    v128_t even_mask = wasm_i32x4_eq(parity, wasm_i32x4_splat(0));
    v128_t negated = wasm_f32x4_neg(results);
    // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    results = wasm_i32x4_relaxed_laneselect(negated, results, even_mask);
    return results;
}

NK_INTERNAL v128_t nk_f32x4_atan_v128relaxed_(v128_t const inputs) {
    // Polynomial coefficients for atan approximation (8 terms)
    v128_t const coeff_8 = wasm_f32x4_splat(-0.333331018686294555664062f);
    v128_t const coeff_7 = wasm_f32x4_splat(+0.199926957488059997558594f);
    v128_t const coeff_6 = wasm_f32x4_splat(-0.142027363181114196777344f);
    v128_t const coeff_5 = wasm_f32x4_splat(+0.106347933411598205566406f);
    v128_t const coeff_4 = wasm_f32x4_splat(-0.0748900920152664184570312f);
    v128_t const coeff_3 = wasm_f32x4_splat(+0.0425049886107444763183594f);
    v128_t const coeff_2 = wasm_f32x4_splat(-0.0159569028764963150024414f);
    v128_t const coeff_1 = wasm_f32x4_splat(+0.00282363896258175373077393f);
    v128_t const half_pi = wasm_f32x4_splat(1.5707963267948966f);

    // Detect negative values and take absolute value
    v128_t const zeros = wasm_f32x4_splat(0);
    v128_t negative_mask = wasm_f32x4_lt(inputs, zeros);
    v128_t values = wasm_f32x4_abs(inputs);

    // Check if values > 1 (need reciprocal)
    v128_t reciprocal_mask = wasm_f32x4_gt(values, wasm_f32x4_splat(1.0f));

    // No fast reciprocal in WASM — use division
    v128_t recip = wasm_f32x4_div(wasm_f32x4_splat(1.0f), values);
    // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    values = wasm_i32x4_relaxed_laneselect(recip, values, reciprocal_mask);

    // Compute powers
    v128_t const values_squared = wasm_f32x4_mul(values, values);
    v128_t const values_cubed = wasm_f32x4_mul(values, values_squared);

    // Polynomial evaluation using Horner's method
    v128_t polynomials = coeff_1;
    polynomials = wasm_f32x4_relaxed_madd(polynomials, values_squared, coeff_2);
    polynomials = wasm_f32x4_relaxed_madd(polynomials, values_squared, coeff_3);
    polynomials = wasm_f32x4_relaxed_madd(polynomials, values_squared, coeff_4);
    polynomials = wasm_f32x4_relaxed_madd(polynomials, values_squared, coeff_5);
    polynomials = wasm_f32x4_relaxed_madd(polynomials, values_squared, coeff_6);
    polynomials = wasm_f32x4_relaxed_madd(polynomials, values_squared, coeff_7);
    polynomials = wasm_f32x4_relaxed_madd(polynomials, values_squared, coeff_8);

    // Compute result: atan(x) ~ x + x^3 * P(x^2)
    v128_t result = wasm_f32x4_relaxed_madd(values_cubed, polynomials, values);

    // Adjust for reciprocal: result = pi/2 - result
    v128_t adjusted = wasm_f32x4_sub(half_pi, result);
    // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    result = wasm_i32x4_relaxed_laneselect(adjusted, result, reciprocal_mask);

    // Adjust for negative: result = -result
    v128_t negated = wasm_f32x4_neg(result);
    // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    result = wasm_i32x4_relaxed_laneselect(negated, result, negative_mask);
    return result;
}

NK_INTERNAL v128_t nk_f32x4_atan2_v128relaxed_(v128_t const ys_inputs, v128_t const xs_inputs) {
    // Polynomial coefficients (same as atan)
    v128_t const coeff_8 = wasm_f32x4_splat(-0.333331018686294555664062f);
    v128_t const coeff_7 = wasm_f32x4_splat(+0.199926957488059997558594f);
    v128_t const coeff_6 = wasm_f32x4_splat(-0.142027363181114196777344f);
    v128_t const coeff_5 = wasm_f32x4_splat(+0.106347933411598205566406f);
    v128_t const coeff_4 = wasm_f32x4_splat(-0.0748900920152664184570312f);
    v128_t const coeff_3 = wasm_f32x4_splat(+0.0425049886107444763183594f);
    v128_t const coeff_2 = wasm_f32x4_splat(-0.0159569028764963150024414f);
    v128_t const coeff_1 = wasm_f32x4_splat(+0.00282363896258175373077393f);
    v128_t const pi = wasm_f32x4_splat(3.14159265358979323846f);
    v128_t const half_pi = wasm_f32x4_splat(1.5707963267948966f);
    v128_t const zeros = wasm_f32x4_splat(0);

    // Quadrant adjustments - take absolute values
    v128_t xs_negative_mask = wasm_f32x4_lt(xs_inputs, zeros);
    v128_t xs = wasm_f32x4_abs(xs_inputs);
    v128_t ys = wasm_f32x4_abs(ys_inputs);

    // Ensure proper fraction where numerator < denominator
    v128_t swap_mask = wasm_f32x4_gt(ys, xs);
    v128_t temps = xs;
    // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    xs = wasm_i32x4_relaxed_laneselect(ys, xs, swap_mask);
    ys = wasm_i32x4_relaxed_laneselect(wasm_f32x4_neg(temps), ys, swap_mask);

    // Division for ratio: ratio = ys / xs
    v128_t const ratio = wasm_f32x4_div(ys, xs);
    v128_t const ratio_squared = wasm_f32x4_mul(ratio, ratio);
    v128_t const ratio_cubed = wasm_f32x4_mul(ratio, ratio_squared);

    // Polynomial evaluation using Horner's method
    v128_t polynomials = coeff_1;
    polynomials = wasm_f32x4_relaxed_madd(polynomials, ratio_squared, coeff_2);
    polynomials = wasm_f32x4_relaxed_madd(polynomials, ratio_squared, coeff_3);
    polynomials = wasm_f32x4_relaxed_madd(polynomials, ratio_squared, coeff_4);
    polynomials = wasm_f32x4_relaxed_madd(polynomials, ratio_squared, coeff_5);
    polynomials = wasm_f32x4_relaxed_madd(polynomials, ratio_squared, coeff_6);
    polynomials = wasm_f32x4_relaxed_madd(polynomials, ratio_squared, coeff_7);
    polynomials = wasm_f32x4_relaxed_madd(polynomials, ratio_squared, coeff_8);

    // Compute the result
    v128_t results = wasm_f32x4_relaxed_madd(ratio_cubed, polynomials, ratio);

    // Compute quadrant value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    v128_t quadrant = wasm_f32x4_splat(0.0f);
    v128_t neg_two = wasm_f32x4_splat(-2.0f);
    // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    quadrant = wasm_i32x4_relaxed_laneselect(neg_two, quadrant, xs_negative_mask);
    v128_t quadrant_incremented = wasm_f32x4_add(quadrant, wasm_f32x4_splat(1.0f));
    quadrant = wasm_i32x4_relaxed_laneselect(quadrant_incremented, quadrant, swap_mask);

    // Adjust for quadrant: result += quadrant * pi/2
    results = wasm_f32x4_relaxed_madd(quadrant, half_pi, results);

    // Transfer sign from x and y by XOR with sign bits
    v128_t sign_mask = wasm_f32x4_splat(-0.0f);
    v128_t xs_sign = wasm_v128_and(xs_inputs, sign_mask);
    v128_t ys_sign = wasm_v128_and(ys_inputs, sign_mask);
    results = wasm_v128_xor(results, xs_sign);
    results = wasm_v128_xor(results, ys_sign);

    return results;
}

NK_INTERNAL v128_t nk_f64x2_sin_v128relaxed_(v128_t const angles_radians) {
    // Constants for argument reduction
    v128_t const pi_high = wasm_f64x2_splat(3.141592653589793116);
    v128_t const pi_low = wasm_f64x2_splat(1.2246467991473532072e-16);
    v128_t const pi_reciprocal = wasm_f64x2_splat(0.31830988618379067154);

    // Polynomial coefficients for sine approximation
    v128_t const coeff_0 = wasm_f64x2_splat(+0.00833333333333332974823815);
    v128_t const coeff_1 = wasm_f64x2_splat(-0.000198412698412696162806809);
    v128_t const coeff_2 = wasm_f64x2_splat(+2.75573192239198747630416e-06);
    v128_t const coeff_3 = wasm_f64x2_splat(-2.50521083763502045810755e-08);
    v128_t const coeff_4 = wasm_f64x2_splat(+1.60590430605664501629054e-10);
    v128_t const coeff_5 = wasm_f64x2_splat(-7.64712219118158833288484e-13);
    v128_t const coeff_6 = wasm_f64x2_splat(+2.81009972710863200091251e-15);
    v128_t const coeff_7 = wasm_f64x2_splat(-7.97255955009037868891952e-18);
    v128_t const coeff_8 = wasm_f64x2_splat(-0.166666666666666657414808);

    // Compute round(angle / pi)
    v128_t const quotients = wasm_f64x2_mul(angles_radians, pi_reciprocal);
    v128_t rounded_quotients = wasm_f64x2_nearest(quotients);
    // relaxed_trunc: 1 instruction (cvttpd2dq) vs 7 (with NaN/overflow fixup) on x86.
    // Safe because rounded_quotients are small integers from nearest(), never NaN or out of i32 range.
    v128_t multiples_i32 = wasm_i32x4_relaxed_trunc_f64x2_zero(rounded_quotients);

    // Two-step Cody-Waite reduction: angle - rounded * pi_high - rounded * pi_low
    v128_t angles = angles_radians;
    angles = wasm_f64x2_relaxed_nmadd(rounded_quotients, pi_high, angles);
    angles = wasm_f64x2_relaxed_nmadd(rounded_quotients, pi_low, angles);

    // Check parity in i32, then widen to i64 mask for laneselect
    v128_t parity_i32 = wasm_v128_and(multiples_i32, wasm_i32x4_splat(1));
    v128_t odd_i32 = wasm_i32x4_eq(parity_i32, wasm_i32x4_splat(1));
    // Widen: lane0 of i32 -> lanes 0-1 of i64, lane1 -> lanes 2-3
    // Shuffle i32 lanes [0,0,1,1] to broadcast each i32 parity into both halves of each i64
    v128_t odd_mask = wasm_i32x4_shuffle(odd_i32, odd_i32, 0, 0, 1, 1);
    v128_t negated_angles = wasm_f64x2_neg(angles);
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is lane-granular at i64 width (all-ones or all-zeros per 64-bit lane).
    angles = wasm_i64x2_relaxed_laneselect(negated_angles, angles, odd_mask);

    v128_t const angles_squared = wasm_f64x2_mul(angles, angles);
    v128_t const angles_cubed = wasm_f64x2_mul(angles, angles_squared);
    v128_t const angles_quadratic = wasm_f64x2_mul(angles_squared, angles_squared);
    v128_t const angles_octic = wasm_f64x2_mul(angles_quadratic, angles_quadratic);

    // Compute polynomial terms using Estrin's scheme for better ILP
    v128_t const poly_67 = wasm_f64x2_relaxed_madd(angles_squared, coeff_7, coeff_6);
    v128_t const poly_45 = wasm_f64x2_relaxed_madd(angles_squared, coeff_5, coeff_4);
    v128_t const poly_4567 = wasm_f64x2_relaxed_madd(angles_quadratic, poly_67, poly_45);

    v128_t const poly_23 = wasm_f64x2_relaxed_madd(angles_squared, coeff_3, coeff_2);
    v128_t const poly_01 = wasm_f64x2_relaxed_madd(angles_squared, coeff_1, coeff_0);
    v128_t const poly_0123 = wasm_f64x2_relaxed_madd(angles_quadratic, poly_23, poly_01);

    // Combine polynomial terms
    v128_t results = wasm_f64x2_relaxed_madd(angles_octic, poly_4567, poly_0123);
    results = wasm_f64x2_relaxed_madd(results, angles_squared, coeff_8);
    results = wasm_f64x2_relaxed_madd(results, angles_cubed, angles);

    // Handle zero input (preserve sign of zero)
    v128_t const non_zero_mask = wasm_f64x2_eq(angles_radians, wasm_f64x2_splat(0));
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    results = wasm_i64x2_relaxed_laneselect(angles_radians, results, non_zero_mask);
    return results;
}

NK_INTERNAL v128_t nk_f64x2_cos_v128relaxed_(v128_t const angles_radians) {
    // Constants for argument reduction
    v128_t const pi_high_half = wasm_f64x2_splat(3.141592653589793116 * 0.5);
    v128_t const pi_low_half = wasm_f64x2_splat(1.2246467991473532072e-16 * 0.5);
    v128_t const pi_reciprocal = wasm_f64x2_splat(0.31830988618379067154);

    // Polynomial coefficients for cosine approximation
    v128_t const coeff_0 = wasm_f64x2_splat(+0.00833333333333332974823815);
    v128_t const coeff_1 = wasm_f64x2_splat(-0.000198412698412696162806809);
    v128_t const coeff_2 = wasm_f64x2_splat(+2.75573192239198747630416e-06);
    v128_t const coeff_3 = wasm_f64x2_splat(-2.50521083763502045810755e-08);
    v128_t const coeff_4 = wasm_f64x2_splat(+1.60590430605664501629054e-10);
    v128_t const coeff_5 = wasm_f64x2_splat(-7.64712219118158833288484e-13);
    v128_t const coeff_6 = wasm_f64x2_splat(+2.81009972710863200091251e-15);
    v128_t const coeff_7 = wasm_f64x2_splat(-7.97255955009037868891952e-18);
    v128_t const coeff_8 = wasm_f64x2_splat(-0.166666666666666657414808);

    // Compute 2 * round(angle / pi - 0.5) + 1
    v128_t const neg_half = wasm_f64x2_splat(-0.5);
    v128_t const quotients = wasm_f64x2_relaxed_madd(angles_radians, pi_reciprocal, neg_half);
    v128_t const rounded = wasm_f64x2_nearest(quotients);
    v128_t const rounded_quotients = wasm_f64x2_relaxed_madd(wasm_f64x2_splat(2.0), rounded, wasm_f64x2_splat(1.0));
    // relaxed_trunc: 1 instruction (cvttpd2dq) vs 7 (with NaN/overflow fixup) on x86.
    // Safe because rounded_quotients are small integers from nearest(), never NaN or out of i32 range.
    v128_t quotients_i32 = wasm_i32x4_relaxed_trunc_f64x2_zero(rounded_quotients);

    // Two-step Cody-Waite reduction
    v128_t angles = angles_radians;
    angles = wasm_f64x2_relaxed_nmadd(rounded_quotients, pi_high_half, angles);
    angles = wasm_f64x2_relaxed_nmadd(rounded_quotients, pi_low_half, angles);

    // Check bit 1 in i32, then widen to i64 mask for laneselect
    v128_t bit2_i32 = wasm_v128_and(quotients_i32, wasm_i32x4_splat(2));
    v128_t flip_i32 = wasm_i32x4_eq(bit2_i32, wasm_i32x4_splat(0));
    v128_t flip_mask = wasm_i32x4_shuffle(flip_i32, flip_i32, 0, 0, 1, 1);
    v128_t negated_angles = wasm_f64x2_neg(angles);
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is lane-granular at i64 width (all-ones or all-zeros per 64-bit lane).
    angles = wasm_i64x2_relaxed_laneselect(negated_angles, angles, flip_mask);

    v128_t const angles_squared = wasm_f64x2_mul(angles, angles);
    v128_t const angles_cubed = wasm_f64x2_mul(angles, angles_squared);
    v128_t const angles_quadratic = wasm_f64x2_mul(angles_squared, angles_squared);
    v128_t const angles_octic = wasm_f64x2_mul(angles_quadratic, angles_quadratic);

    // Compute polynomial terms using Estrin's scheme
    v128_t const poly_67 = wasm_f64x2_relaxed_madd(angles_squared, coeff_7, coeff_6);
    v128_t const poly_45 = wasm_f64x2_relaxed_madd(angles_squared, coeff_5, coeff_4);
    v128_t const poly_4567 = wasm_f64x2_relaxed_madd(angles_quadratic, poly_67, poly_45);

    v128_t const poly_23 = wasm_f64x2_relaxed_madd(angles_squared, coeff_3, coeff_2);
    v128_t const poly_01 = wasm_f64x2_relaxed_madd(angles_squared, coeff_1, coeff_0);
    v128_t const poly_0123 = wasm_f64x2_relaxed_madd(angles_quadratic, poly_23, poly_01);

    // Combine polynomial terms
    v128_t results = wasm_f64x2_relaxed_madd(angles_octic, poly_4567, poly_0123);
    results = wasm_f64x2_relaxed_madd(results, angles_squared, coeff_8);
    results = wasm_f64x2_relaxed_madd(results, angles_cubed, angles);
    return results;
}

NK_INTERNAL v128_t nk_f64x2_atan_v128relaxed_(v128_t const inputs) {
    // Polynomial coefficients for atan approximation (19 terms)
    v128_t const coeff_19 = wasm_f64x2_splat(-1.88796008463073496563746e-05);
    v128_t const coeff_18 = wasm_f64x2_splat(+0.000209850076645816976906797);
    v128_t const coeff_17 = wasm_f64x2_splat(-0.00110611831486672482563471);
    v128_t const coeff_16 = wasm_f64x2_splat(+0.00370026744188713119232403);
    v128_t const coeff_15 = wasm_f64x2_splat(-0.00889896195887655491740809);
    v128_t const coeff_14 = wasm_f64x2_splat(+0.016599329773529201970117);
    v128_t const coeff_13 = wasm_f64x2_splat(-0.0254517624932312641616861);
    v128_t const coeff_12 = wasm_f64x2_splat(+0.0337852580001353069993897);
    v128_t const coeff_11 = wasm_f64x2_splat(-0.0407629191276836500001934);
    v128_t const coeff_10 = wasm_f64x2_splat(+0.0466667150077840625632675);
    v128_t const coeff_9 = wasm_f64x2_splat(-0.0523674852303482457616113);
    v128_t const coeff_8 = wasm_f64x2_splat(+0.0587666392926673580854313);
    v128_t const coeff_7 = wasm_f64x2_splat(-0.0666573579361080525984562);
    v128_t const coeff_6 = wasm_f64x2_splat(+0.0769219538311769618355029);
    v128_t const coeff_5 = wasm_f64x2_splat(-0.090908995008245008229153);
    v128_t const coeff_4 = wasm_f64x2_splat(+0.111111105648261418443745);
    v128_t const coeff_3 = wasm_f64x2_splat(-0.14285714266771329383765);
    v128_t const coeff_2 = wasm_f64x2_splat(+0.199999999996591265594148);
    v128_t const coeff_1 = wasm_f64x2_splat(-0.333333333333311110369124);
    v128_t const half_pi = wasm_f64x2_splat(1.5707963267948966);
    v128_t const zeros = wasm_f64x2_splat(0);

    // Detect negative and take absolute value
    v128_t negative_mask = wasm_f64x2_lt(inputs, zeros);
    v128_t values = wasm_f64x2_abs(inputs);

    // Check if values > 1 (need reciprocal) - use division for f64 precision
    v128_t reciprocal_mask = wasm_f64x2_gt(values, wasm_f64x2_splat(1.0));
    v128_t reciprocal_values = wasm_f64x2_div(wasm_f64x2_splat(1.0), values);
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    values = wasm_i64x2_relaxed_laneselect(reciprocal_values, values, reciprocal_mask);

    // Compute powers
    v128_t const values_squared = wasm_f64x2_mul(values, values);
    v128_t const values_cubed = wasm_f64x2_mul(values, values_squared);

    // Polynomial evaluation using Horner's method
    v128_t polynomials = coeff_19;
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_18);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_17);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_16);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_15);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_14);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_13);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_12);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_11);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_10);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_9);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_8);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_7);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_6);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_5);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_4);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_3);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_2);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, values_squared, coeff_1);

    // Compute result
    v128_t result = wasm_f64x2_relaxed_madd(values_cubed, polynomials, values);

    // Adjust for reciprocal: result = pi/2 - result
    v128_t adjusted = wasm_f64x2_sub(half_pi, result);
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    result = wasm_i64x2_relaxed_laneselect(adjusted, result, reciprocal_mask);

    // Adjust for negative: result = -result
    v128_t negated = wasm_f64x2_neg(result);
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    result = wasm_i64x2_relaxed_laneselect(negated, result, negative_mask);
    return result;
}

NK_INTERNAL v128_t nk_f64x2_atan2_v128relaxed_(v128_t const ys_inputs, v128_t const xs_inputs) {
    // Polynomial coefficients (same as atan)
    v128_t const coeff_19 = wasm_f64x2_splat(-1.88796008463073496563746e-05);
    v128_t const coeff_18 = wasm_f64x2_splat(+0.000209850076645816976906797);
    v128_t const coeff_17 = wasm_f64x2_splat(-0.00110611831486672482563471);
    v128_t const coeff_16 = wasm_f64x2_splat(+0.00370026744188713119232403);
    v128_t const coeff_15 = wasm_f64x2_splat(-0.00889896195887655491740809);
    v128_t const coeff_14 = wasm_f64x2_splat(+0.016599329773529201970117);
    v128_t const coeff_13 = wasm_f64x2_splat(-0.0254517624932312641616861);
    v128_t const coeff_12 = wasm_f64x2_splat(+0.0337852580001353069993897);
    v128_t const coeff_11 = wasm_f64x2_splat(-0.0407629191276836500001934);
    v128_t const coeff_10 = wasm_f64x2_splat(+0.0466667150077840625632675);
    v128_t const coeff_9 = wasm_f64x2_splat(-0.0523674852303482457616113);
    v128_t const coeff_8 = wasm_f64x2_splat(+0.0587666392926673580854313);
    v128_t const coeff_7 = wasm_f64x2_splat(-0.0666573579361080525984562);
    v128_t const coeff_6 = wasm_f64x2_splat(+0.0769219538311769618355029);
    v128_t const coeff_5 = wasm_f64x2_splat(-0.090908995008245008229153);
    v128_t const coeff_4 = wasm_f64x2_splat(+0.111111105648261418443745);
    v128_t const coeff_3 = wasm_f64x2_splat(-0.14285714266771329383765);
    v128_t const coeff_2 = wasm_f64x2_splat(+0.199999999996591265594148);
    v128_t const coeff_1 = wasm_f64x2_splat(-0.333333333333311110369124);
    v128_t const pi = wasm_f64x2_splat(3.14159265358979323846);
    v128_t const half_pi = wasm_f64x2_splat(1.5707963267948966);
    v128_t const zeros = wasm_f64x2_splat(0);

    // Quadrant adjustments - take absolute values
    v128_t xs_negative_mask = wasm_f64x2_lt(xs_inputs, zeros);
    v128_t xs = wasm_f64x2_abs(xs_inputs);
    v128_t ys = wasm_f64x2_abs(ys_inputs);

    // Ensure proper fraction where numerator < denominator
    v128_t swap_mask = wasm_f64x2_gt(ys, xs);
    v128_t temps = xs;
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    xs = wasm_i64x2_relaxed_laneselect(ys, xs, swap_mask);
    ys = wasm_i64x2_relaxed_laneselect(wasm_f64x2_neg(temps), ys, swap_mask);

    // Division for f64 precision
    v128_t const ratio = wasm_f64x2_div(ys, xs);
    v128_t const ratio_squared = wasm_f64x2_mul(ratio, ratio);
    v128_t const ratio_cubed = wasm_f64x2_mul(ratio, ratio_squared);

    // Polynomial evaluation using Horner's method
    v128_t polynomials = coeff_19;
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_18);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_17);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_16);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_15);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_14);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_13);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_12);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_11);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_10);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_9);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_8);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_7);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_6);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_5);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_4);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_3);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_2);
    polynomials = wasm_f64x2_relaxed_madd(polynomials, ratio_squared, coeff_1);

    // Compute the result
    v128_t results = wasm_f64x2_relaxed_madd(ratio_cubed, polynomials, ratio);

    // Compute quadrant value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    v128_t quadrant = wasm_f64x2_splat(0.0);
    v128_t neg_two = wasm_f64x2_splat(-2.0);
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    quadrant = wasm_i64x2_relaxed_laneselect(neg_two, quadrant, xs_negative_mask);
    v128_t quadrant_incremented = wasm_f64x2_add(quadrant, wasm_f64x2_splat(1.0));
    quadrant = wasm_i64x2_relaxed_laneselect(quadrant_incremented, quadrant, swap_mask);

    // Adjust for quadrant: result += quadrant * pi/2
    results = wasm_f64x2_relaxed_madd(quadrant, half_pi, results);

    // Transfer sign from x and y by XOR with sign bits
    v128_t sign_mask = wasm_f64x2_splat(-0.0);
    v128_t xs_sign = wasm_v128_and(xs_inputs, sign_mask);
    v128_t ys_sign = wasm_v128_and(ys_inputs, sign_mask);
    results = wasm_v128_xor(results, xs_sign);
    results = wasm_v128_xor(results, ys_sign);

    return results;
}

/*  NK_PUBLIC wrappers — same loop+tail pattern as neon.h.
 *  Full loads use wasm_v128_load/wasm_v128_store.
 *  Tails use nk_partial_load_b32x4_serial_/nk_partial_store_b32x4_serial_ via .v128 union member.
 */

NK_PUBLIC void nk_each_sin_f32_v128relaxed(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        v128_t angles = wasm_v128_load(ins + i);
        v128_t results = nk_f32x4_sin_v128relaxed_(angles);
        wasm_v128_store(outs + i, results);
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
        v128_t angles = wasm_v128_load(ins + i);
        v128_t results = nk_f32x4_cos_v128relaxed_(angles);
        wasm_v128_store(outs + i, results);
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
        v128_t values = wasm_v128_load(ins + i);
        v128_t results = nk_f32x4_atan_v128relaxed_(values);
        wasm_v128_store(outs + i, results);
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
        v128_t angles = wasm_v128_load(ins + i);
        v128_t results = nk_f64x2_sin_v128relaxed_(angles);
        wasm_v128_store(outs + i, results);
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
        v128_t angles = wasm_v128_load(ins + i);
        v128_t results = nk_f64x2_cos_v128relaxed_(angles);
        wasm_v128_store(outs + i, results);
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
        v128_t values = wasm_v128_load(ins + i);
        v128_t results = nk_f64x2_atan_v128relaxed_(values);
        wasm_v128_store(outs + i, results);
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
