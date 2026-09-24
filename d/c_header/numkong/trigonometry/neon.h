/**
 *  @brief SIMD-accelerated Trigonometric Functions for NEON.
 *  @file include/numkong/trigonometry/neon.h
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *
 *  @sa include/numkong/trigonometry.h
 *  @see https://sleef.org
 *
 *  @section trigonometry_neon_instructions ARM NEON Instructions
 *
 *      Intrinsic         Instruction                   Latency     Throughput
 *                                                                  A76     M4+/V1+/Oryon
 *      vfmaq_f32         FMLA (V.4S, V.4S, V.4S)       4cy         2/cy    4/cy
 *      vfmsq_f32         FMLS (V.4S, V.4S, V.4S)       4cy         2/cy    4/cy
 *      vmulq_f32         FMUL (V.4S, V.4S, V.4S)       3cy         2/cy    4/cy
 *      vaddq_f32         FADD (V.4S, V.4S, V.4S)       2cy         2/cy    4/cy
 *      vsubq_f32         FSUB (V.4S, V.4S, V.4S)       2cy         2/cy    4/cy
 *      vcvtnq_s32_f32    FCVTNS (V.4S, V.4S)           3cy         2/cy    2/cy
 *      vcvtq_f32_s32     SCVTF (V.4S, V.4S)            3cy         2/cy    2/cy
 *      vbslq_f32         BSL (V.16B, V.16B, V.16B)     2cy         2/cy    4/cy
 *      vrecpeq_f32       FRECPE (V.4S, V.4S)           2cy         2/cy    2/cy
 *      vrecpsq_f32       FRECPS (V.4S, V.4S, V.4S)     4cy         2/cy    4/cy
 *      vfmaq_f64         FMLA (V.2D, V.2D, V.2D)       4cy         2/cy    4/cy
 *      vdivq_f64         FDIV (V.2D, V.2D, V.2D)       15cy        0.5/cy  0.5/cy
 *
 *  Polynomial approximations for sin/cos/atan are FMA-dominated. On 4-pipe cores (Apple M4+,
 *  Graviton3+, Oryon), FMA throughput is 4/cy with 4cy latency.
 *
 *  Division (vdivq_f64) remains slow at 0.5/cy on all cores. For f32, use fast reciprocal
 *  (vrecpeq_f32 + Newton-Raphson) instead when precision allows.
 */
#ifndef NK_TRIGONOMETRY_NEON_H
#define NK_TRIGONOMETRY_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/types.h"
#include "numkong/reduce/neon.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

/*  NEON trigonometry kernels (4-way f32, 2-way f64)
 *  These implement polynomial approximations using 128-bit NEON vectors.
 */

NK_INTERNAL float32x4_t nk_sin_f32x4_neon_(float32x4_t const angles_radians) {
    // Cody-Waite constants for argument reduction
    float32x4_t const pi_hi_f32x4 = vdupq_n_f32(3.1415927f);
    float32x4_t const pi_lo_f32x4 = vdupq_n_f32(-8.742278e-8f);
    float32x4_t const pi_reciprocal = vdupq_n_f32(0.31830988618379067154f);
    // Degree-9 minimax coefficients
    float32x4_t const coeff_9 = vdupq_n_f32(+2.7557319224e-6f);
    float32x4_t const coeff_7 = vdupq_n_f32(-1.9841269841e-4f);
    float32x4_t const coeff_5 = vdupq_n_f32(+8.3333293855e-3f);
    float32x4_t const coeff_3 = vdupq_n_f32(-1.6666666641e-1f);

    // Compute (multiples_of_pi) = round(angle / π) using vcvtnq which rounds to nearest
    float32x4_t quotients = vmulq_f32(angles_radians, pi_reciprocal);
    int32x4_t multiples_of_pi = vcvtnq_s32_f32(quotients);
    float32x4_t rounded_quotients = vcvtq_f32_s32(multiples_of_pi);

    // Cody-Waite range reduction
    float32x4_t angles = vfmsq_f32(angles_radians, rounded_quotients, pi_hi_f32x4);
    angles = vfmsq_f32(angles, rounded_quotients, pi_lo_f32x4);
    float32x4_t const angles_squared = vmulq_f32(angles, angles);
    float32x4_t const angles_cubed = vmulq_f32(angles, angles_squared);

    // Degree-9 polynomial via Horner's method
    float32x4_t polynomials = coeff_9;
    polynomials = vfmaq_f32(coeff_7, polynomials, angles_squared);
    polynomials = vfmaq_f32(coeff_5, polynomials, angles_squared);
    polynomials = vfmaq_f32(coeff_3, polynomials, angles_squared);
    float32x4_t results = vfmaq_f32(angles, angles_cubed, polynomials);

    // If multiples_of_pi is odd, flip the sign
    int32x4_t parity = vandq_s32(multiples_of_pi, vdupq_n_s32(1));
    uint32x4_t odd_mask = vceqq_s32(parity, vdupq_n_s32(1));
    float32x4_t negated = vnegq_f32(results);
    results = vbslq_f32(odd_mask, negated, results);
    return results;
}

NK_INTERNAL float32x4_t nk_cos_f32x4_neon_(float32x4_t const angles_radians) {
    // Cody-Waite constants for argument reduction
    float32x4_t const pi_hi_f32x4 = vdupq_n_f32(3.1415927f);
    float32x4_t const pi_lo_f32x4 = vdupq_n_f32(-8.742278e-8f);
    float32x4_t const pi_half = vdupq_n_f32(1.57079632679489661923f);
    float32x4_t const pi_reciprocal = vdupq_n_f32(0.31830988618379067154f);
    // Degree-9 minimax coefficients
    float32x4_t const coeff_9 = vdupq_n_f32(+2.7557319224e-6f);
    float32x4_t const coeff_7 = vdupq_n_f32(-1.9841269841e-4f);
    float32x4_t const coeff_5 = vdupq_n_f32(+8.3333293855e-3f);
    float32x4_t const coeff_3 = vdupq_n_f32(-1.6666666641e-1f);

    // Compute round((angle / π) - 0.5)
    float32x4_t quotients = vsubq_f32(vmulq_f32(angles_radians, pi_reciprocal), vdupq_n_f32(0.5f));
    int32x4_t multiples_of_pi = vcvtnq_s32_f32(quotients);
    float32x4_t rounded_quotients = vcvtq_f32_s32(multiples_of_pi);

    // Cody-Waite range reduction: angle = (angle - pi/2) - rounded * (pi_hi + pi_lo)
    float32x4_t shifted = vsubq_f32(angles_radians, pi_half);
    float32x4_t angles = vfmsq_f32(shifted, rounded_quotients, pi_hi_f32x4);
    angles = vfmsq_f32(angles, rounded_quotients, pi_lo_f32x4);
    float32x4_t const angles_squared = vmulq_f32(angles, angles);
    float32x4_t const angles_cubed = vmulq_f32(angles, angles_squared);

    // Degree-9 polynomial via Horner's method
    float32x4_t polynomials = coeff_9;
    polynomials = vfmaq_f32(coeff_7, polynomials, angles_squared);
    polynomials = vfmaq_f32(coeff_5, polynomials, angles_squared);
    polynomials = vfmaq_f32(coeff_3, polynomials, angles_squared);
    float32x4_t results = vfmaq_f32(angles, angles_cubed, polynomials);

    // If multiples_of_pi is even, flip the sign
    int32x4_t parity = vandq_s32(multiples_of_pi, vdupq_n_s32(1));
    uint32x4_t even_mask = vceqq_s32(parity, vdupq_n_s32(0));
    float32x4_t negated = vnegq_f32(results);
    results = vbslq_f32(even_mask, negated, results);
    return results;
}

NK_INTERNAL float32x4_t nk_atan_f32x4_neon_(float32x4_t const inputs) {
    // Polynomial coefficients for atan approximation (8 terms)
    float32x4_t const coeff_8 = vdupq_n_f32(-0.333331018686294555664062f);
    float32x4_t const coeff_7 = vdupq_n_f32(+0.199926957488059997558594f);
    float32x4_t const coeff_6 = vdupq_n_f32(-0.142027363181114196777344f);
    float32x4_t const coeff_5 = vdupq_n_f32(+0.106347933411598205566406f);
    float32x4_t const coeff_4 = vdupq_n_f32(-0.0748900920152664184570312f);
    float32x4_t const coeff_3 = vdupq_n_f32(+0.0425049886107444763183594f);
    float32x4_t const coeff_2 = vdupq_n_f32(-0.0159569028764963150024414f);
    float32x4_t const coeff_1 = vdupq_n_f32(+0.00282363896258175373077393f);
    float32x4_t const half_pi = vdupq_n_f32(1.5707963267948966f);

    // Detect negative values and take absolute value
    float32x4_t const zeros = vdupq_n_f32(0);
    uint32x4_t negative_mask = vcltq_f32(inputs, zeros);
    float32x4_t values = vabsq_f32(inputs);

    // Check if values > 1 (need reciprocal)
    uint32x4_t reciprocal_mask = vcgtq_f32(values, vdupq_n_f32(1.0f));

    // Fast reciprocal using vrecpeq + Newton-Raphson (faster than vdivq on many Arm cores)
    float32x4_t recip = vrecpeq_f32(values);
    recip = vmulq_f32(recip, vrecpsq_f32(values, recip));
    recip = vmulq_f32(recip, vrecpsq_f32(values, recip));
    values = vbslq_f32(reciprocal_mask, recip, values);

    // Compute powers
    float32x4_t const values_squared = vmulq_f32(values, values);
    float32x4_t const values_cubed = vmulq_f32(values, values_squared);

    // Polynomial evaluation using Horner's method
    float32x4_t polynomials = coeff_1;
    polynomials = vfmaq_f32(coeff_2, polynomials, values_squared);
    polynomials = vfmaq_f32(coeff_3, polynomials, values_squared);
    polynomials = vfmaq_f32(coeff_4, polynomials, values_squared);
    polynomials = vfmaq_f32(coeff_5, polynomials, values_squared);
    polynomials = vfmaq_f32(coeff_6, polynomials, values_squared);
    polynomials = vfmaq_f32(coeff_7, polynomials, values_squared);
    polynomials = vfmaq_f32(coeff_8, polynomials, values_squared);

    // Compute result: atan(x) ≈ x + x³ * P(x²)
    float32x4_t result = vfmaq_f32(values, values_cubed, polynomials);

    // Adjust for reciprocal: result = π/2 - result
    float32x4_t adjusted = vsubq_f32(half_pi, result);
    result = vbslq_f32(reciprocal_mask, adjusted, result);

    // Adjust for negative: result = -result
    float32x4_t negated = vnegq_f32(result);
    result = vbslq_f32(negative_mask, negated, result);
    return result;
}

NK_INTERNAL float32x4_t nk_atan2_f32x4_neon_(float32x4_t const ys_inputs, float32x4_t const xs_inputs) {
    // Polynomial coefficients (same as atan)
    float32x4_t const coeff_8 = vdupq_n_f32(-0.333331018686294555664062f);
    float32x4_t const coeff_7 = vdupq_n_f32(+0.199926957488059997558594f);
    float32x4_t const coeff_6 = vdupq_n_f32(-0.142027363181114196777344f);
    float32x4_t const coeff_5 = vdupq_n_f32(+0.106347933411598205566406f);
    float32x4_t const coeff_4 = vdupq_n_f32(-0.0748900920152664184570312f);
    float32x4_t const coeff_3 = vdupq_n_f32(+0.0425049886107444763183594f);
    float32x4_t const coeff_2 = vdupq_n_f32(-0.0159569028764963150024414f);
    float32x4_t const coeff_1 = vdupq_n_f32(+0.00282363896258175373077393f);
    float32x4_t const half_pi = vdupq_n_f32(1.5707963267948966f);
    float32x4_t const zeros = vdupq_n_f32(0);

    // Quadrant adjustments - take absolute values
    uint32x4_t xs_negative_mask = vcltq_f32(xs_inputs, zeros);
    float32x4_t xs = vabsq_f32(xs_inputs);
    float32x4_t ys = vabsq_f32(ys_inputs);

    // Ensure proper fraction where numerator < denominator
    uint32x4_t swap_mask = vcgtq_f32(ys, xs);
    float32x4_t temps = xs;
    xs = vbslq_f32(swap_mask, ys, xs);
    ys = vbslq_f32(swap_mask, vnegq_f32(temps), ys);

    // Fast reciprocal for division: ratio = ys / xs ≈ ys * recip(xs)
    float32x4_t recip = vrecpeq_f32(xs);
    recip = vmulq_f32(recip, vrecpsq_f32(xs, recip));
    recip = vmulq_f32(recip, vrecpsq_f32(xs, recip));
    float32x4_t const ratio = vmulq_f32(ys, recip);
    float32x4_t const ratio_squared = vmulq_f32(ratio, ratio);
    float32x4_t const ratio_cubed = vmulq_f32(ratio, ratio_squared);

    // Polynomial evaluation using Horner's method
    float32x4_t polynomials = coeff_1;
    polynomials = vfmaq_f32(coeff_2, polynomials, ratio_squared);
    polynomials = vfmaq_f32(coeff_3, polynomials, ratio_squared);
    polynomials = vfmaq_f32(coeff_4, polynomials, ratio_squared);
    polynomials = vfmaq_f32(coeff_5, polynomials, ratio_squared);
    polynomials = vfmaq_f32(coeff_6, polynomials, ratio_squared);
    polynomials = vfmaq_f32(coeff_7, polynomials, ratio_squared);
    polynomials = vfmaq_f32(coeff_8, polynomials, ratio_squared);

    // Compute the result
    float32x4_t results = vfmaq_f32(ratio, ratio_cubed, polynomials);

    // Compute quadrant value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    float32x4_t quadrant = vdupq_n_f32(0.0f);
    float32x4_t neg_two = vdupq_n_f32(-2.0f);
    quadrant = vbslq_f32(xs_negative_mask, neg_two, quadrant);
    float32x4_t quadrant_incremented = vaddq_f32(quadrant, vdupq_n_f32(1.0f));
    quadrant = vbslq_f32(swap_mask, quadrant_incremented, quadrant);

    // Adjust for quadrant: result += quadrant * π/2
    results = vfmaq_f32(results, quadrant, half_pi);

    // Transfer sign from x and y by XOR with sign bits
    uint32x4_t sign_mask = vreinterpretq_u32_f32(vdupq_n_f32(-0.0f));
    uint32x4_t xs_sign = vandq_u32(vreinterpretq_u32_f32(xs_inputs), sign_mask);
    uint32x4_t ys_sign = vandq_u32(vreinterpretq_u32_f32(ys_inputs), sign_mask);
    uint32x4_t result_bits = vreinterpretq_u32_f32(results);
    result_bits = veorq_u32(result_bits, xs_sign);
    result_bits = veorq_u32(result_bits, ys_sign);
    results = vreinterpretq_f32_u32(result_bits);

    return results;
}

NK_INTERNAL float64x2_t nk_sin_f64x2_neon_(float64x2_t const angles_radians) {
    // Constants for argument reduction
    float64x2_t const pi_high = vdupq_n_f64(3.141592653589793116);
    float64x2_t const pi_low = vdupq_n_f64(1.2246467991473532072e-16);
    float64x2_t const pi_reciprocal = vdupq_n_f64(0.31830988618379067154);

    // Polynomial coefficients for sine approximation
    float64x2_t const coeff_0 = vdupq_n_f64(+0.00833333333333332974823815);
    float64x2_t const coeff_1 = vdupq_n_f64(-0.000198412698412696162806809);
    float64x2_t const coeff_2 = vdupq_n_f64(+2.75573192239198747630416e-06);
    float64x2_t const coeff_3 = vdupq_n_f64(-2.50521083763502045810755e-08);
    float64x2_t const coeff_4 = vdupq_n_f64(+1.60590430605664501629054e-10);
    float64x2_t const coeff_5 = vdupq_n_f64(-7.64712219118158833288484e-13);
    float64x2_t const coeff_6 = vdupq_n_f64(+2.81009972710863200091251e-15);
    float64x2_t const coeff_7 = vdupq_n_f64(-7.97255955009037868891952e-18);
    float64x2_t const coeff_8 = vdupq_n_f64(-0.166666666666666657414808);

    // Compute round(angle / π)
    float64x2_t const quotients = vmulq_f64(angles_radians, pi_reciprocal);
    int64x2_t multiples_of_pi = vcvtnq_s64_f64(quotients);
    float64x2_t rounded_quotients = vcvtq_f64_s64(multiples_of_pi);

    // Two-step Cody-Waite reduction: angle - rounded * π_high - rounded * π_low
    float64x2_t angles = angles_radians;
    angles = vfmsq_f64(angles, rounded_quotients, pi_high);
    angles = vfmsq_f64(angles, rounded_quotients, pi_low);

    // If multiples_of_pi is odd, negate the angle
    int64x2_t parity = vandq_s64(multiples_of_pi, vdupq_n_s64(1));
    uint64x2_t odd_mask = vceqq_s64(parity, vdupq_n_s64(1));
    float64x2_t negated_angles = vnegq_f64(angles);
    angles = vbslq_f64(odd_mask, negated_angles, angles);

    float64x2_t const angles_squared = vmulq_f64(angles, angles);
    float64x2_t const angles_cubed = vmulq_f64(angles, angles_squared);
    float64x2_t const angles_quadratic = vmulq_f64(angles_squared, angles_squared);
    float64x2_t const angles_octic = vmulq_f64(angles_quadratic, angles_quadratic);

    // Compute polynomial terms using Estrin's scheme for better ILP
    float64x2_t const poly_67 = vfmaq_f64(coeff_6, angles_squared, coeff_7);
    float64x2_t const poly_45 = vfmaq_f64(coeff_4, angles_squared, coeff_5);
    float64x2_t const poly_4567 = vfmaq_f64(poly_45, angles_quadratic, poly_67);

    float64x2_t const poly_23 = vfmaq_f64(coeff_2, angles_squared, coeff_3);
    float64x2_t const poly_01 = vfmaq_f64(coeff_0, angles_squared, coeff_1);
    float64x2_t const poly_0123 = vfmaq_f64(poly_01, angles_quadratic, poly_23);

    // Combine polynomial terms
    float64x2_t results = vfmaq_f64(poly_0123, angles_octic, poly_4567);
    results = vfmaq_f64(coeff_8, results, angles_squared);
    results = vfmaq_f64(angles, results, angles_cubed);

    // Handle zero input (preserve sign of zero)
    uint64x2_t const non_zero_mask = vceqq_f64(angles_radians, vdupq_n_f64(0));
    results = vbslq_f64(non_zero_mask, angles_radians, results);
    return results;
}

NK_INTERNAL float64x2_t nk_cos_f64x2_neon_(float64x2_t const angles_radians) {
    // Constants for argument reduction
    float64x2_t const pi_high_half = vdupq_n_f64(3.141592653589793116 * 0.5);
    float64x2_t const pi_low_half = vdupq_n_f64(1.2246467991473532072e-16 * 0.5);
    float64x2_t const pi_reciprocal = vdupq_n_f64(0.31830988618379067154);

    // Polynomial coefficients for cosine approximation
    float64x2_t const coeff_0 = vdupq_n_f64(+0.00833333333333332974823815);
    float64x2_t const coeff_1 = vdupq_n_f64(-0.000198412698412696162806809);
    float64x2_t const coeff_2 = vdupq_n_f64(+2.75573192239198747630416e-06);
    float64x2_t const coeff_3 = vdupq_n_f64(-2.50521083763502045810755e-08);
    float64x2_t const coeff_4 = vdupq_n_f64(+1.60590430605664501629054e-10);
    float64x2_t const coeff_5 = vdupq_n_f64(-7.64712219118158833288484e-13);
    float64x2_t const coeff_6 = vdupq_n_f64(+2.81009972710863200091251e-15);
    float64x2_t const coeff_7 = vdupq_n_f64(-7.97255955009037868891952e-18);
    float64x2_t const coeff_8 = vdupq_n_f64(-0.166666666666666657414808);

    // Compute 2 * round(angle / π - 0.5) + 1
    float64x2_t const quotients = vsubq_f64(vmulq_f64(angles_radians, pi_reciprocal), vdupq_n_f64(0.5));
    float64x2_t const rounded = vcvtq_f64_s64(vcvtnq_s64_f64(quotients));
    float64x2_t const rounded_quotients = vfmaq_f64(vdupq_n_f64(1.0), vdupq_n_f64(2.0), rounded);
    int64x2_t quotients_i64 = vcvtnq_s64_f64(rounded_quotients);

    // Two-step Cody-Waite reduction
    float64x2_t angles = angles_radians;
    angles = vfmsq_f64(angles, rounded_quotients, pi_high_half);
    angles = vfmsq_f64(angles, rounded_quotients, pi_low_half);

    // If (rounded_quotients & 2) == 0, negate the angle
    int64x2_t bit2 = vandq_s64(quotients_i64, vdupq_n_s64(2));
    uint64x2_t flip_mask = vceqq_s64(bit2, vdupq_n_s64(0));
    float64x2_t negated_angles = vnegq_f64(angles);
    angles = vbslq_f64(flip_mask, negated_angles, angles);

    float64x2_t const angles_squared = vmulq_f64(angles, angles);
    float64x2_t const angles_cubed = vmulq_f64(angles, angles_squared);
    float64x2_t const angles_quadratic = vmulq_f64(angles_squared, angles_squared);
    float64x2_t const angles_octic = vmulq_f64(angles_quadratic, angles_quadratic);

    // Compute polynomial terms using Estrin's scheme
    float64x2_t const poly_67 = vfmaq_f64(coeff_6, angles_squared, coeff_7);
    float64x2_t const poly_45 = vfmaq_f64(coeff_4, angles_squared, coeff_5);
    float64x2_t const poly_4567 = vfmaq_f64(poly_45, angles_quadratic, poly_67);

    float64x2_t const poly_23 = vfmaq_f64(coeff_2, angles_squared, coeff_3);
    float64x2_t const poly_01 = vfmaq_f64(coeff_0, angles_squared, coeff_1);
    float64x2_t const poly_0123 = vfmaq_f64(poly_01, angles_quadratic, poly_23);

    // Combine polynomial terms
    float64x2_t results = vfmaq_f64(poly_0123, angles_octic, poly_4567);
    results = vfmaq_f64(coeff_8, results, angles_squared);
    results = vfmaq_f64(angles, results, angles_cubed);
    return results;
}

NK_INTERNAL float64x2_t nk_atan_f64x2_neon_(float64x2_t const inputs) {
    // Polynomial coefficients for atan approximation (19 terms)
    float64x2_t const coeff_19 = vdupq_n_f64(-1.88796008463073496563746e-05);
    float64x2_t const coeff_18 = vdupq_n_f64(+0.000209850076645816976906797);
    float64x2_t const coeff_17 = vdupq_n_f64(-0.00110611831486672482563471);
    float64x2_t const coeff_16 = vdupq_n_f64(+0.00370026744188713119232403);
    float64x2_t const coeff_15 = vdupq_n_f64(-0.00889896195887655491740809);
    float64x2_t const coeff_14 = vdupq_n_f64(+0.016599329773529201970117);
    float64x2_t const coeff_13 = vdupq_n_f64(-0.0254517624932312641616861);
    float64x2_t const coeff_12 = vdupq_n_f64(+0.0337852580001353069993897);
    float64x2_t const coeff_11 = vdupq_n_f64(-0.0407629191276836500001934);
    float64x2_t const coeff_10 = vdupq_n_f64(+0.0466667150077840625632675);
    float64x2_t const coeff_9 = vdupq_n_f64(-0.0523674852303482457616113);
    float64x2_t const coeff_8 = vdupq_n_f64(+0.0587666392926673580854313);
    float64x2_t const coeff_7 = vdupq_n_f64(-0.0666573579361080525984562);
    float64x2_t const coeff_6 = vdupq_n_f64(+0.0769219538311769618355029);
    float64x2_t const coeff_5 = vdupq_n_f64(-0.090908995008245008229153);
    float64x2_t const coeff_4 = vdupq_n_f64(+0.111111105648261418443745);
    float64x2_t const coeff_3 = vdupq_n_f64(-0.14285714266771329383765);
    float64x2_t const coeff_2 = vdupq_n_f64(+0.199999999996591265594148);
    float64x2_t const coeff_1 = vdupq_n_f64(-0.333333333333311110369124);
    float64x2_t const half_pi = vdupq_n_f64(1.5707963267948966);
    float64x2_t const zeros = vdupq_n_f64(0);

    // Detect negative and take absolute value
    uint64x2_t negative_mask = vcltq_f64(inputs, zeros);
    float64x2_t values = vabsq_f64(inputs);

    // Check if values > 1 (need reciprocal) - use division for f64 precision
    uint64x2_t reciprocal_mask = vcgtq_f64(values, vdupq_n_f64(1.0));
    float64x2_t reciprocal_values = vdivq_f64(vdupq_n_f64(1.0), values);
    values = vbslq_f64(reciprocal_mask, reciprocal_values, values);

    // Compute powers
    float64x2_t const values_squared = vmulq_f64(values, values);
    float64x2_t const values_cubed = vmulq_f64(values, values_squared);

    // Polynomial evaluation using Horner's method
    float64x2_t polynomials = coeff_19;
    polynomials = vfmaq_f64(coeff_18, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_17, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_16, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_15, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_14, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_13, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_12, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_11, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_10, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_9, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_8, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_7, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_6, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_5, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_4, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_3, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_2, polynomials, values_squared);
    polynomials = vfmaq_f64(coeff_1, polynomials, values_squared);

    // Compute result
    float64x2_t result = vfmaq_f64(values, values_cubed, polynomials);

    // Adjust for reciprocal: result = π/2 - result
    float64x2_t adjusted = vsubq_f64(half_pi, result);
    result = vbslq_f64(reciprocal_mask, adjusted, result);

    // Adjust for negative: result = -result
    float64x2_t negated = vnegq_f64(result);
    result = vbslq_f64(negative_mask, negated, result);
    return result;
}

NK_INTERNAL float64x2_t nk_atan2_f64x2_neon_(float64x2_t const ys_inputs, float64x2_t const xs_inputs) {
    // Polynomial coefficients (same as atan)
    float64x2_t const coeff_19 = vdupq_n_f64(-1.88796008463073496563746e-05);
    float64x2_t const coeff_18 = vdupq_n_f64(+0.000209850076645816976906797);
    float64x2_t const coeff_17 = vdupq_n_f64(-0.00110611831486672482563471);
    float64x2_t const coeff_16 = vdupq_n_f64(+0.00370026744188713119232403);
    float64x2_t const coeff_15 = vdupq_n_f64(-0.00889896195887655491740809);
    float64x2_t const coeff_14 = vdupq_n_f64(+0.016599329773529201970117);
    float64x2_t const coeff_13 = vdupq_n_f64(-0.0254517624932312641616861);
    float64x2_t const coeff_12 = vdupq_n_f64(+0.0337852580001353069993897);
    float64x2_t const coeff_11 = vdupq_n_f64(-0.0407629191276836500001934);
    float64x2_t const coeff_10 = vdupq_n_f64(+0.0466667150077840625632675);
    float64x2_t const coeff_9 = vdupq_n_f64(-0.0523674852303482457616113);
    float64x2_t const coeff_8 = vdupq_n_f64(+0.0587666392926673580854313);
    float64x2_t const coeff_7 = vdupq_n_f64(-0.0666573579361080525984562);
    float64x2_t const coeff_6 = vdupq_n_f64(+0.0769219538311769618355029);
    float64x2_t const coeff_5 = vdupq_n_f64(-0.090908995008245008229153);
    float64x2_t const coeff_4 = vdupq_n_f64(+0.111111105648261418443745);
    float64x2_t const coeff_3 = vdupq_n_f64(-0.14285714266771329383765);
    float64x2_t const coeff_2 = vdupq_n_f64(+0.199999999996591265594148);
    float64x2_t const coeff_1 = vdupq_n_f64(-0.333333333333311110369124);
    float64x2_t const half_pi = vdupq_n_f64(1.5707963267948966);
    float64x2_t const zeros = vdupq_n_f64(0);

    // Quadrant adjustments - take absolute values
    uint64x2_t xs_negative_mask = vcltq_f64(xs_inputs, zeros);
    float64x2_t xs = vabsq_f64(xs_inputs);
    float64x2_t ys = vabsq_f64(ys_inputs);

    // Ensure proper fraction where numerator < denominator
    uint64x2_t swap_mask = vcgtq_f64(ys, xs);
    float64x2_t temps = xs;
    xs = vbslq_f64(swap_mask, ys, xs);
    ys = vbslq_f64(swap_mask, vnegq_f64(temps), ys);

    // Division for f64 precision
    float64x2_t const ratio = vdivq_f64(ys, xs);
    float64x2_t const ratio_squared = vmulq_f64(ratio, ratio);
    float64x2_t const ratio_cubed = vmulq_f64(ratio, ratio_squared);

    // Polynomial evaluation using Horner's method
    float64x2_t polynomials = coeff_19;
    polynomials = vfmaq_f64(coeff_18, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_17, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_16, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_15, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_14, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_13, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_12, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_11, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_10, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_9, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_8, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_7, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_6, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_5, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_4, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_3, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_2, polynomials, ratio_squared);
    polynomials = vfmaq_f64(coeff_1, polynomials, ratio_squared);

    // Compute the result
    float64x2_t results = vfmaq_f64(ratio, ratio_cubed, polynomials);

    // Compute quadrant value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    float64x2_t quadrant = vdupq_n_f64(0.0);
    float64x2_t neg_two = vdupq_n_f64(-2.0);
    quadrant = vbslq_f64(xs_negative_mask, neg_two, quadrant);
    float64x2_t quadrant_incremented = vaddq_f64(quadrant, vdupq_n_f64(1.0));
    quadrant = vbslq_f64(swap_mask, quadrant_incremented, quadrant);

    // Adjust for quadrant: result += quadrant * π/2
    results = vfmaq_f64(results, quadrant, half_pi);

    // Transfer sign from x and y by XOR with sign bits
    uint64x2_t sign_mask = vreinterpretq_u64_f64(vdupq_n_f64(-0.0));
    uint64x2_t xs_sign = vandq_u64(vreinterpretq_u64_f64(xs_inputs), sign_mask);
    uint64x2_t ys_sign = vandq_u64(vreinterpretq_u64_f64(ys_inputs), sign_mask);
    uint64x2_t result_bits = vreinterpretq_u64_f64(results);
    result_bits = veorq_u64(result_bits, xs_sign);
    result_bits = veorq_u64(result_bits, ys_sign);
    results = vreinterpretq_f64_u64(result_bits);

    return results;
}

NK_PUBLIC void nk_each_sin_f32_neon(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t angles = vld1q_f32(ins + i);
        float32x4_t results = nk_sin_f32x4_neon_(angles);
        vst1q_f32(outs + i, results);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b128_vec_t angles_vec;
        nk_partial_load_b32x4_serial_(ins + i, &angles_vec, remaining);
        nk_b128_vec_t results_vec;
        results_vec.f32x4 = nk_sin_f32x4_neon_(angles_vec.f32x4);
        nk_partial_store_b32x4_serial_(&results_vec, outs + i, remaining);
    }
}

NK_PUBLIC void nk_each_cos_f32_neon(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t angles = vld1q_f32(ins + i);
        float32x4_t results = nk_cos_f32x4_neon_(angles);
        vst1q_f32(outs + i, results);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b128_vec_t angles_vec;
        nk_partial_load_b32x4_serial_(ins + i, &angles_vec, remaining);
        nk_b128_vec_t results_vec;
        results_vec.f32x4 = nk_cos_f32x4_neon_(angles_vec.f32x4);
        nk_partial_store_b32x4_serial_(&results_vec, outs + i, remaining);
    }
}

NK_PUBLIC void nk_each_atan_f32_neon(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t values = vld1q_f32(ins + i);
        float32x4_t results = nk_atan_f32x4_neon_(values);
        vst1q_f32(outs + i, results);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b128_vec_t values_vec;
        nk_partial_load_b32x4_serial_(ins + i, &values_vec, remaining);
        nk_b128_vec_t results_vec;
        results_vec.f32x4 = nk_atan_f32x4_neon_(values_vec.f32x4);
        nk_partial_store_b32x4_serial_(&results_vec, outs + i, remaining);
    }
}

NK_PUBLIC void nk_each_sin_f64_neon(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t angles = vld1q_f64(ins + i);
        float64x2_t results = nk_sin_f64x2_neon_(angles);
        vst1q_f64(outs + i, results);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b128_vec_t angles_vec;
        nk_partial_load_b64x2_serial_(ins + i, &angles_vec, remaining);
        nk_b128_vec_t results_vec;
        results_vec.f64x2 = nk_sin_f64x2_neon_(angles_vec.f64x2);
        nk_partial_store_b64x2_serial_(&results_vec, outs + i, remaining);
    }
}

NK_PUBLIC void nk_each_cos_f64_neon(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t angles = vld1q_f64(ins + i);
        float64x2_t results = nk_cos_f64x2_neon_(angles);
        vst1q_f64(outs + i, results);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b128_vec_t angles_vec;
        nk_partial_load_b64x2_serial_(ins + i, &angles_vec, remaining);
        nk_b128_vec_t results_vec;
        results_vec.f64x2 = nk_cos_f64x2_neon_(angles_vec.f64x2);
        nk_partial_store_b64x2_serial_(&results_vec, outs + i, remaining);
    }
}

NK_PUBLIC void nk_each_atan_f64_neon(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t values = vld1q_f64(ins + i);
        float64x2_t results = nk_atan_f64x2_neon_(values);
        vst1q_f64(outs + i, results);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b128_vec_t values_vec;
        nk_partial_load_b64x2_serial_(ins + i, &values_vec, remaining);
        nk_b128_vec_t results_vec;
        results_vec.f64x2 = nk_atan_f64x2_neon_(values_vec.f64x2);
        nk_partial_store_b64x2_serial_(&results_vec, outs + i, remaining);
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_
#endif // NK_TRIGONOMETRY_NEON_H
