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
 *      Intrinsic       Instruction                A76        M5
 *      vfmaq_f32       FMLA (V.4S, V.4S, V.4S)    4cy @ 2p   3cy @ 4p
 *      vfmsq_f32       FMLS (V.4S, V.4S, V.4S)    4cy @ 2p   3cy @ 4p
 *      vmulq_f32       FMUL (V.4S, V.4S, V.4S)    3cy @ 2p   3cy @ 4p
 *      vaddq_f32       FADD (V.4S, V.4S, V.4S)    2cy @ 2p   2cy @ 4p
 *      vsubq_f32       FSUB (V.4S, V.4S, V.4S)    2cy @ 2p   2cy @ 4p
 *      vcvtnq_s32_f32  FCVTNS (V.4S, V.4S)        3cy @ 2p   3cy @ 4p
 *      vcvtq_f32_s32   SCVTF (V.4S, V.4S)         3cy @ 2p   3cy @ 4p
 *      vbslq_f32       BSL (V.16B, V.16B, V.16B)  1cy @ 2p   2cy @ 4p
 *      vrecpeq_f32     FRECPE (V.4S, V.4S)        2cy @ 2p   3cy @ 1p
 *      vrecpsq_f32     FRECPS (V.4S, V.4S, V.4S)  4cy @ 2p   3cy @ 2p
 *      vfmaq_f64       FMLA (V.2D, V.2D, V.2D)    4cy @ 2p   3cy @ 4p
 *      vdivq_f64       FDIV (V.2D, V.2D, V.2D)    12cy @ 1p  7cy @ 1p
 *
 *  Polynomial approximations for sin/cos/atan are FMA-dominated. On 4-pipe cores (Apple M4+,
 *  Graviton3+, Oryon), FMA throughput is 4/cy with 4cy latency.
 *
 *  Division (vdivq_f64) remains slow at 0.5/cy on all cores. For f32, use fast reciprocal
 *  (vrecpeq_f32 + Newton-Raphson) instead when precision allows.
 */
#ifndef NK_TRIGONOMETRY_NEON_H
#define NK_TRIGONOMETRY_NEON_H

#if NK_TARGET_ARM64_
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

NK_HELPER_INLINE float32x4_t nk_sin_f32x4_neon_(float32x4_t const angles_radians) {
    // Cody-Waite constants for argument reduction
    float32x4_t const pi_high_f32x4 = vdupq_n_f32(3.1415927f);
    float32x4_t const pi_low_f32x4 = vdupq_n_f32(-8.742278e-8f);
    float32x4_t const pi_reciprocal_f32x4 = vdupq_n_f32(0.31830988618379067154f);
    // Degree-9 minimax coefficients
    float32x4_t const coeff_9_f32x4 = vdupq_n_f32(+2.7557319224e-6f);
    float32x4_t const coeff_7_f32x4 = vdupq_n_f32(-1.9841269841e-4f);
    float32x4_t const coeff_5_f32x4 = vdupq_n_f32(+8.3333293855e-3f);
    float32x4_t const coeff_3_f32x4 = vdupq_n_f32(-1.6666666641e-1f);

    // Compute (multiples_of_pi_i32x4) = round(angle / π) using vcvtnq which rounds to nearest
    float32x4_t quotients_f32x4 = vmulq_f32(angles_radians, pi_reciprocal_f32x4);
    int32x4_t multiples_of_pi_i32x4 = vcvtnq_s32_f32(quotients_f32x4);
    float32x4_t rounded_quotients_f32x4 = vcvtq_f32_s32(multiples_of_pi_i32x4);

    // Cody-Waite range reduction
    float32x4_t angles_f32x4 = vfmsq_f32(angles_radians, rounded_quotients_f32x4, pi_high_f32x4);
    angles_f32x4 = vfmsq_f32(angles_f32x4, rounded_quotients_f32x4, pi_low_f32x4);
    float32x4_t const angles_sq_f32x4 = vmulq_f32(angles_f32x4, angles_f32x4);
    float32x4_t const angles_cubed_f32x4 = vmulq_f32(angles_f32x4, angles_sq_f32x4);

    // Degree-9 polynomial via Horner's method
    float32x4_t polynomials_f32x4 = coeff_9_f32x4;
    polynomials_f32x4 = vfmaq_f32(coeff_7_f32x4, polynomials_f32x4, angles_sq_f32x4);
    polynomials_f32x4 = vfmaq_f32(coeff_5_f32x4, polynomials_f32x4, angles_sq_f32x4);
    polynomials_f32x4 = vfmaq_f32(coeff_3_f32x4, polynomials_f32x4, angles_sq_f32x4);
    float32x4_t results_f32x4 = vfmaq_f32(angles_f32x4, angles_cubed_f32x4, polynomials_f32x4);

    // If multiples_of_pi_i32x4 is odd, flip the sign
    int32x4_t parity_i32x4 = vandq_s32(multiples_of_pi_i32x4, vdupq_n_s32(1));
    uint32x4_t odd_mask_u32x4 = vceqq_s32(parity_i32x4, vdupq_n_s32(1));
    float32x4_t negated_f32x4 = vnegq_f32(results_f32x4);
    results_f32x4 = vbslq_f32(odd_mask_u32x4, negated_f32x4, results_f32x4);
    return results_f32x4;
}

NK_HELPER_INLINE float32x4_t nk_cos_f32x4_neon_(float32x4_t const angles_radians) {
    // Cody-Waite constants for argument reduction
    float32x4_t const pi_high_f32x4 = vdupq_n_f32(3.1415927f);
    float32x4_t const pi_low_f32x4 = vdupq_n_f32(-8.742278e-8f);
    float32x4_t const pi_half_f32x4 = vdupq_n_f32(1.57079632679489661923f);
    float32x4_t const pi_reciprocal_f32x4 = vdupq_n_f32(0.31830988618379067154f);
    // Degree-9 minimax coefficients
    float32x4_t const coeff_9_f32x4 = vdupq_n_f32(+2.7557319224e-6f);
    float32x4_t const coeff_7_f32x4 = vdupq_n_f32(-1.9841269841e-4f);
    float32x4_t const coeff_5_f32x4 = vdupq_n_f32(+8.3333293855e-3f);
    float32x4_t const coeff_3_f32x4 = vdupq_n_f32(-1.6666666641e-1f);

    // Compute round((angle / π) - 0.5)
    float32x4_t quotients_f32x4 = vsubq_f32(vmulq_f32(angles_radians, pi_reciprocal_f32x4), vdupq_n_f32(0.5f));
    int32x4_t multiples_of_pi_i32x4 = vcvtnq_s32_f32(quotients_f32x4);
    float32x4_t rounded_quotients_f32x4 = vcvtq_f32_s32(multiples_of_pi_i32x4);

    // Cody-Waite range reduction: angle = (angle - pi/2) - rounded * (pi_high + pi_low)
    float32x4_t shifted_f32x4 = vsubq_f32(angles_radians, pi_half_f32x4);
    float32x4_t angles_f32x4 = vfmsq_f32(shifted_f32x4, rounded_quotients_f32x4, pi_high_f32x4);
    angles_f32x4 = vfmsq_f32(angles_f32x4, rounded_quotients_f32x4, pi_low_f32x4);
    float32x4_t const angles_sq_f32x4 = vmulq_f32(angles_f32x4, angles_f32x4);
    float32x4_t const angles_cubed_f32x4 = vmulq_f32(angles_f32x4, angles_sq_f32x4);

    // Degree-9 polynomial via Horner's method
    float32x4_t polynomials_f32x4 = coeff_9_f32x4;
    polynomials_f32x4 = vfmaq_f32(coeff_7_f32x4, polynomials_f32x4, angles_sq_f32x4);
    polynomials_f32x4 = vfmaq_f32(coeff_5_f32x4, polynomials_f32x4, angles_sq_f32x4);
    polynomials_f32x4 = vfmaq_f32(coeff_3_f32x4, polynomials_f32x4, angles_sq_f32x4);
    float32x4_t results_f32x4 = vfmaq_f32(angles_f32x4, angles_cubed_f32x4, polynomials_f32x4);

    // If multiples_of_pi_i32x4 is even, flip the sign
    int32x4_t parity_i32x4 = vandq_s32(multiples_of_pi_i32x4, vdupq_n_s32(1));
    uint32x4_t even_mask_u32x4 = vceqq_s32(parity_i32x4, vdupq_n_s32(0));
    float32x4_t negated_f32x4 = vnegq_f32(results_f32x4);
    results_f32x4 = vbslq_f32(even_mask_u32x4, negated_f32x4, results_f32x4);
    return results_f32x4;
}

NK_HELPER_INLINE float32x4_t nk_atan_f32x4_neon_(float32x4_t const inputs) {
    // Polynomial coefficients for atan approximation (8 terms)
    float32x4_t const coeff_8_f32x4 = vdupq_n_f32(-0.333331018686294555664062f);
    float32x4_t const coeff_7_f32x4 = vdupq_n_f32(+0.199926957488059997558594f);
    float32x4_t const coeff_6_f32x4 = vdupq_n_f32(-0.142027363181114196777344f);
    float32x4_t const coeff_5_f32x4 = vdupq_n_f32(+0.106347933411598205566406f);
    float32x4_t const coeff_4_f32x4 = vdupq_n_f32(-0.0748900920152664184570312f);
    float32x4_t const coeff_3_f32x4 = vdupq_n_f32(+0.0425049886107444763183594f);
    float32x4_t const coeff_2_f32x4 = vdupq_n_f32(-0.0159569028764963150024414f);
    float32x4_t const coeff_1_f32x4 = vdupq_n_f32(+0.00282363896258175373077393f);
    float32x4_t const half_pi_f32x4 = vdupq_n_f32(1.5707963267948966f);

    // Detect negative values_f32x4 and take absolute value
    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);
    uint32x4_t negative_mask_u32x4 = vcltq_f32(inputs, zeros_f32x4);
    float32x4_t values_f32x4 = vabsq_f32(inputs);

    // Check if values_f32x4 > 1 (need reciprocal)
    uint32x4_t reciprocal_mask_u32x4 = vcgtq_f32(values_f32x4, vdupq_n_f32(1.0f));

    // Fast reciprocal using vrecpeq + Newton-Raphson (faster than vdivq on many Arm cores)
    float32x4_t reciprocal_f32x4 = vrecpeq_f32(values_f32x4);
    reciprocal_f32x4 = vmulq_f32(reciprocal_f32x4, vrecpsq_f32(values_f32x4, reciprocal_f32x4));
    reciprocal_f32x4 = vmulq_f32(reciprocal_f32x4, vrecpsq_f32(values_f32x4, reciprocal_f32x4));
    values_f32x4 = vbslq_f32(reciprocal_mask_u32x4, reciprocal_f32x4, values_f32x4);

    // Compute powers
    float32x4_t const values_sq_f32x4 = vmulq_f32(values_f32x4, values_f32x4);
    float32x4_t const values_cubed_f32x4 = vmulq_f32(values_f32x4, values_sq_f32x4);

    // Polynomial evaluation using Horner's method
    float32x4_t polynomials_f32x4 = coeff_1_f32x4;
    polynomials_f32x4 = vfmaq_f32(coeff_2_f32x4, polynomials_f32x4, values_sq_f32x4);
    polynomials_f32x4 = vfmaq_f32(coeff_3_f32x4, polynomials_f32x4, values_sq_f32x4);
    polynomials_f32x4 = vfmaq_f32(coeff_4_f32x4, polynomials_f32x4, values_sq_f32x4);
    polynomials_f32x4 = vfmaq_f32(coeff_5_f32x4, polynomials_f32x4, values_sq_f32x4);
    polynomials_f32x4 = vfmaq_f32(coeff_6_f32x4, polynomials_f32x4, values_sq_f32x4);
    polynomials_f32x4 = vfmaq_f32(coeff_7_f32x4, polynomials_f32x4, values_sq_f32x4);
    polynomials_f32x4 = vfmaq_f32(coeff_8_f32x4, polynomials_f32x4, values_sq_f32x4);

    // Compute result_f32x4: atan(x) ≈ x + x³ * P(x²)
    float32x4_t result_f32x4 = vfmaq_f32(values_f32x4, values_cubed_f32x4, polynomials_f32x4);

    // Adjust for reciprocal: result_f32x4 = π/2 - result_f32x4
    float32x4_t adjusted_f32x4 = vsubq_f32(half_pi_f32x4, result_f32x4);
    result_f32x4 = vbslq_f32(reciprocal_mask_u32x4, adjusted_f32x4, result_f32x4);

    // Adjust for negative: result_f32x4 = -result_f32x4
    float32x4_t negated_f32x4 = vnegq_f32(result_f32x4);
    result_f32x4 = vbslq_f32(negative_mask_u32x4, negated_f32x4, result_f32x4);
    return result_f32x4;
}

NK_HELPER_INLINE float32x4_t nk_atan2_f32x4_neon_(float32x4_t const ys_inputs, float32x4_t const xs_inputs) {
    // Polynomial coefficients (same as atan)
    float32x4_t const coeff_8_f32x4 = vdupq_n_f32(-0.333331018686294555664062f);
    float32x4_t const coeff_7_f32x4 = vdupq_n_f32(+0.199926957488059997558594f);
    float32x4_t const coeff_6_f32x4 = vdupq_n_f32(-0.142027363181114196777344f);
    float32x4_t const coeff_5_f32x4 = vdupq_n_f32(+0.106347933411598205566406f);
    float32x4_t const coeff_4_f32x4 = vdupq_n_f32(-0.0748900920152664184570312f);
    float32x4_t const coeff_3_f32x4 = vdupq_n_f32(+0.0425049886107444763183594f);
    float32x4_t const coeff_2_f32x4 = vdupq_n_f32(-0.0159569028764963150024414f);
    float32x4_t const coeff_1_f32x4 = vdupq_n_f32(+0.00282363896258175373077393f);
    float32x4_t const half_pi_f32x4 = vdupq_n_f32(1.5707963267948966f);
    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);

    // Quadrant adjustments - take absolute values
    uint32x4_t xs_negative_mask_u32x4 = vcltq_f32(xs_inputs, zeros_f32x4);
    float32x4_t xs_f32x4 = vabsq_f32(xs_inputs);
    float32x4_t ys_f32x4 = vabsq_f32(ys_inputs);

    // Ensure proper fraction where numerator < denominator
    uint32x4_t swap_mask_u32x4 = vcgtq_f32(ys_f32x4, xs_f32x4);
    float32x4_t temps_f32x4 = xs_f32x4;
    xs_f32x4 = vbslq_f32(swap_mask_u32x4, ys_f32x4, xs_f32x4);
    ys_f32x4 = vbslq_f32(swap_mask_u32x4, vnegq_f32(temps_f32x4), ys_f32x4);

    // Fast reciprocal for division: ratio_f32x4 = ys_f32x4 / xs_f32x4 ≈ ys_f32x4 * recip_f32x4(xs_f32x4)
    float32x4_t reciprocal_f32x4 = vrecpeq_f32(xs_f32x4);
    reciprocal_f32x4 = vmulq_f32(reciprocal_f32x4, vrecpsq_f32(xs_f32x4, reciprocal_f32x4));
    reciprocal_f32x4 = vmulq_f32(reciprocal_f32x4, vrecpsq_f32(xs_f32x4, reciprocal_f32x4));
    float32x4_t const ratio_f32x4 = vmulq_f32(ys_f32x4, reciprocal_f32x4);
    float32x4_t const ratio_sq_f32x4 = vmulq_f32(ratio_f32x4, ratio_f32x4);
    float32x4_t const ratio_cubed_f32x4 = vmulq_f32(ratio_f32x4, ratio_sq_f32x4);

    // Polynomial evaluation using Horner's method
    float32x4_t polynomials_f32x4 = coeff_1_f32x4;
    polynomials_f32x4 = vfmaq_f32(coeff_2_f32x4, polynomials_f32x4, ratio_sq_f32x4);
    polynomials_f32x4 = vfmaq_f32(coeff_3_f32x4, polynomials_f32x4, ratio_sq_f32x4);
    polynomials_f32x4 = vfmaq_f32(coeff_4_f32x4, polynomials_f32x4, ratio_sq_f32x4);
    polynomials_f32x4 = vfmaq_f32(coeff_5_f32x4, polynomials_f32x4, ratio_sq_f32x4);
    polynomials_f32x4 = vfmaq_f32(coeff_6_f32x4, polynomials_f32x4, ratio_sq_f32x4);
    polynomials_f32x4 = vfmaq_f32(coeff_7_f32x4, polynomials_f32x4, ratio_sq_f32x4);
    polynomials_f32x4 = vfmaq_f32(coeff_8_f32x4, polynomials_f32x4, ratio_sq_f32x4);

    // Compute the result
    float32x4_t results_f32x4 = vfmaq_f32(ratio_f32x4, ratio_cubed_f32x4, polynomials_f32x4);

    // Compute quadrant_f32x4 value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    float32x4_t quadrant_f32x4 = vdupq_n_f32(0.0f);
    float32x4_t neg_two_f32x4 = vdupq_n_f32(-2.0f);
    quadrant_f32x4 = vbslq_f32(xs_negative_mask_u32x4, neg_two_f32x4, quadrant_f32x4);
    float32x4_t quadrant_incremented_f32x4 = vaddq_f32(quadrant_f32x4, vdupq_n_f32(1.0f));
    quadrant_f32x4 = vbslq_f32(swap_mask_u32x4, quadrant_incremented_f32x4, quadrant_f32x4);

    // Adjust for quadrant_f32x4: result += quadrant_f32x4 * π/2
    results_f32x4 = vfmaq_f32(results_f32x4, quadrant_f32x4, half_pi_f32x4);

    // Transfer sign from x and y by XOR with sign bits
    uint32x4_t sign_mask_u32x4 = vreinterpretq_u32_f32(vdupq_n_f32(-0.0f));
    uint32x4_t xs_sign_u32x4 = vandq_u32(vreinterpretq_u32_f32(xs_inputs), sign_mask_u32x4);
    uint32x4_t ys_sign_u32x4 = vandq_u32(vreinterpretq_u32_f32(ys_inputs), sign_mask_u32x4);
    uint32x4_t result_u32x4 = vreinterpretq_u32_f32(results_f32x4);
    result_u32x4 = veorq_u32(result_u32x4, xs_sign_u32x4);
    result_u32x4 = veorq_u32(result_u32x4, ys_sign_u32x4);
    results_f32x4 = vreinterpretq_f32_u32(result_u32x4);

    return results_f32x4;
}

NK_HELPER_INLINE float64x2_t nk_sin_f64x2_neon_(float64x2_t const angles_radians) {
    // Constants for argument reduction
    float64x2_t const pi_high_f64x2 = vdupq_n_f64(3.141592653589793116);
    float64x2_t const pi_low_f64x2 = vdupq_n_f64(1.2246467991473532072e-16);
    float64x2_t const pi_reciprocal_f64x2 = vdupq_n_f64(0.31830988618379067154);

    // Polynomial coefficients for sine approximation
    float64x2_t const coeff_0_f64x2 = vdupq_n_f64(+0.00833333333333332974823815);
    float64x2_t const coeff_1_f64x2 = vdupq_n_f64(-0.000198412698412696162806809);
    float64x2_t const coeff_2_f64x2 = vdupq_n_f64(+2.75573192239198747630416e-06);
    float64x2_t const coeff_3_f64x2 = vdupq_n_f64(-2.50521083763502045810755e-08);
    float64x2_t const coeff_4_f64x2 = vdupq_n_f64(+1.60590430605664501629054e-10);
    float64x2_t const coeff_5_f64x2 = vdupq_n_f64(-7.64712219118158833288484e-13);
    float64x2_t const coeff_6_f64x2 = vdupq_n_f64(+2.81009972710863200091251e-15);
    float64x2_t const coeff_7_f64x2 = vdupq_n_f64(-7.97255955009037868891952e-18);
    float64x2_t const coeff_8_f64x2 = vdupq_n_f64(-0.166666666666666657414808);

    // Compute round(angle / π)
    float64x2_t const quotients_f64x2 = vmulq_f64(angles_radians, pi_reciprocal_f64x2);
    int64x2_t multiples_of_pi_i64x2 = vcvtnq_s64_f64(quotients_f64x2);
    float64x2_t rounded_quotients_f64x2 = vcvtq_f64_s64(multiples_of_pi_i64x2);

    // Two-step Cody-Waite reduction: angle - rounded * π_high - rounded * π_low
    float64x2_t angles_f64x2 = angles_radians;
    angles_f64x2 = vfmsq_f64(angles_f64x2, rounded_quotients_f64x2, pi_high_f64x2);
    angles_f64x2 = vfmsq_f64(angles_f64x2, rounded_quotients_f64x2, pi_low_f64x2);

    // If multiples_of_pi_i64x2 is odd, negate the angle
    int64x2_t parity_i64x2 = vandq_s64(multiples_of_pi_i64x2, vdupq_n_s64(1));
    uint64x2_t odd_mask_u64x2 = vceqq_s64(parity_i64x2, vdupq_n_s64(1));
    float64x2_t negated_angles_f64x2 = vnegq_f64(angles_f64x2);
    angles_f64x2 = vbslq_f64(odd_mask_u64x2, negated_angles_f64x2, angles_f64x2);

    float64x2_t const angles_sq_f64x2 = vmulq_f64(angles_f64x2, angles_f64x2);
    float64x2_t const angles_cubed_f64x2 = vmulq_f64(angles_f64x2, angles_sq_f64x2);
    float64x2_t const angles_quadratic_f64x2 = vmulq_f64(angles_sq_f64x2, angles_sq_f64x2);
    float64x2_t const angles_octic_f64x2 = vmulq_f64(angles_quadratic_f64x2, angles_quadratic_f64x2);

    // Compute polynomial terms using Estrin's scheme for better ILP
    float64x2_t const poly_67_f64x2 = vfmaq_f64(coeff_6_f64x2, angles_sq_f64x2, coeff_7_f64x2);
    float64x2_t const poly_45_f64x2 = vfmaq_f64(coeff_4_f64x2, angles_sq_f64x2, coeff_5_f64x2);
    float64x2_t const poly_4567_f64x2 = vfmaq_f64(poly_45_f64x2, angles_quadratic_f64x2, poly_67_f64x2);

    float64x2_t const poly_23_f64x2 = vfmaq_f64(coeff_2_f64x2, angles_sq_f64x2, coeff_3_f64x2);
    float64x2_t const poly_01_f64x2 = vfmaq_f64(coeff_0_f64x2, angles_sq_f64x2, coeff_1_f64x2);
    float64x2_t const poly_0123_f64x2 = vfmaq_f64(poly_01_f64x2, angles_quadratic_f64x2, poly_23_f64x2);

    // Combine polynomial terms
    float64x2_t results_f64x2 = vfmaq_f64(poly_0123_f64x2, angles_octic_f64x2, poly_4567_f64x2);
    results_f64x2 = vfmaq_f64(coeff_8_f64x2, results_f64x2, angles_sq_f64x2);
    results_f64x2 = vfmaq_f64(angles_f64x2, results_f64x2, angles_cubed_f64x2);

    // Handle zero input (preserve sign of zero)
    uint64x2_t const non_zero_mask_u64x2 = vceqq_f64(angles_radians, vdupq_n_f64(0));
    results_f64x2 = vbslq_f64(non_zero_mask_u64x2, angles_radians, results_f64x2);
    return results_f64x2;
}

NK_HELPER_INLINE float64x2_t nk_cos_f64x2_neon_(float64x2_t const angles_radians) {
    // Constants for argument reduction
    float64x2_t const pi_high_half_f64x2 = vdupq_n_f64(3.141592653589793116 * 0.5);
    float64x2_t const pi_low_half_f64x2 = vdupq_n_f64(1.2246467991473532072e-16 * 0.5);
    float64x2_t const pi_reciprocal_f64x2 = vdupq_n_f64(0.31830988618379067154);

    // Polynomial coefficients for cosine approximation
    float64x2_t const coeff_0_f64x2 = vdupq_n_f64(+0.00833333333333332974823815);
    float64x2_t const coeff_1_f64x2 = vdupq_n_f64(-0.000198412698412696162806809);
    float64x2_t const coeff_2_f64x2 = vdupq_n_f64(+2.75573192239198747630416e-06);
    float64x2_t const coeff_3_f64x2 = vdupq_n_f64(-2.50521083763502045810755e-08);
    float64x2_t const coeff_4_f64x2 = vdupq_n_f64(+1.60590430605664501629054e-10);
    float64x2_t const coeff_5_f64x2 = vdupq_n_f64(-7.64712219118158833288484e-13);
    float64x2_t const coeff_6_f64x2 = vdupq_n_f64(+2.81009972710863200091251e-15);
    float64x2_t const coeff_7_f64x2 = vdupq_n_f64(-7.97255955009037868891952e-18);
    float64x2_t const coeff_8_f64x2 = vdupq_n_f64(-0.166666666666666657414808);

    // Compute 2 * round(angle / π - 0.5) + 1
    float64x2_t const quotients_f64x2 = vsubq_f64(vmulq_f64(angles_radians, pi_reciprocal_f64x2), vdupq_n_f64(0.5));
    float64x2_t const rounded_f64x2 = vcvtq_f64_s64(vcvtnq_s64_f64(quotients_f64x2));
    float64x2_t const rounded_quotients_f64x2 = vfmaq_f64(vdupq_n_f64(1.0), vdupq_n_f64(2.0), rounded_f64x2);
    int64x2_t quotients_i64_i64x2 = vcvtnq_s64_f64(rounded_quotients_f64x2);

    // Two-step Cody-Waite reduction
    float64x2_t angles_f64x2 = angles_radians;
    angles_f64x2 = vfmsq_f64(angles_f64x2, rounded_quotients_f64x2, pi_high_half_f64x2);
    angles_f64x2 = vfmsq_f64(angles_f64x2, rounded_quotients_f64x2, pi_low_half_f64x2);

    // If (rounded_quotients_f64x2 & 2) == 0, negate the angle
    int64x2_t bit2_i64x2 = vandq_s64(quotients_i64_i64x2, vdupq_n_s64(2));
    uint64x2_t flip_mask_u64x2 = vceqq_s64(bit2_i64x2, vdupq_n_s64(0));
    float64x2_t negated_angles_f64x2 = vnegq_f64(angles_f64x2);
    angles_f64x2 = vbslq_f64(flip_mask_u64x2, negated_angles_f64x2, angles_f64x2);

    float64x2_t const angles_sq_f64x2 = vmulq_f64(angles_f64x2, angles_f64x2);
    float64x2_t const angles_cubed_f64x2 = vmulq_f64(angles_f64x2, angles_sq_f64x2);
    float64x2_t const angles_quadratic_f64x2 = vmulq_f64(angles_sq_f64x2, angles_sq_f64x2);
    float64x2_t const angles_octic_f64x2 = vmulq_f64(angles_quadratic_f64x2, angles_quadratic_f64x2);

    // Compute polynomial terms using Estrin's scheme
    float64x2_t const poly_67_f64x2 = vfmaq_f64(coeff_6_f64x2, angles_sq_f64x2, coeff_7_f64x2);
    float64x2_t const poly_45_f64x2 = vfmaq_f64(coeff_4_f64x2, angles_sq_f64x2, coeff_5_f64x2);
    float64x2_t const poly_4567_f64x2 = vfmaq_f64(poly_45_f64x2, angles_quadratic_f64x2, poly_67_f64x2);

    float64x2_t const poly_23_f64x2 = vfmaq_f64(coeff_2_f64x2, angles_sq_f64x2, coeff_3_f64x2);
    float64x2_t const poly_01_f64x2 = vfmaq_f64(coeff_0_f64x2, angles_sq_f64x2, coeff_1_f64x2);
    float64x2_t const poly_0123_f64x2 = vfmaq_f64(poly_01_f64x2, angles_quadratic_f64x2, poly_23_f64x2);

    // Combine polynomial terms
    float64x2_t results_f64x2 = vfmaq_f64(poly_0123_f64x2, angles_octic_f64x2, poly_4567_f64x2);
    results_f64x2 = vfmaq_f64(coeff_8_f64x2, results_f64x2, angles_sq_f64x2);
    results_f64x2 = vfmaq_f64(angles_f64x2, results_f64x2, angles_cubed_f64x2);
    return results_f64x2;
}

NK_HELPER_INLINE float64x2_t nk_atan_f64x2_neon_(float64x2_t const inputs) {
    // Polynomial coefficients for atan approximation (19 terms)
    float64x2_t const coeff_19_f64x2 = vdupq_n_f64(-1.88796008463073496563746e-05);
    float64x2_t const coeff_18_f64x2 = vdupq_n_f64(+0.000209850076645816976906797);
    float64x2_t const coeff_17_f64x2 = vdupq_n_f64(-0.00110611831486672482563471);
    float64x2_t const coeff_16_f64x2 = vdupq_n_f64(+0.00370026744188713119232403);
    float64x2_t const coeff_15_f64x2 = vdupq_n_f64(-0.00889896195887655491740809);
    float64x2_t const coeff_14_f64x2 = vdupq_n_f64(+0.016599329773529201970117);
    float64x2_t const coeff_13_f64x2 = vdupq_n_f64(-0.0254517624932312641616861);
    float64x2_t const coeff_12_f64x2 = vdupq_n_f64(+0.0337852580001353069993897);
    float64x2_t const coeff_11_f64x2 = vdupq_n_f64(-0.0407629191276836500001934);
    float64x2_t const coeff_10_f64x2 = vdupq_n_f64(+0.0466667150077840625632675);
    float64x2_t const coeff_9_f64x2 = vdupq_n_f64(-0.0523674852303482457616113);
    float64x2_t const coeff_8_f64x2 = vdupq_n_f64(+0.0587666392926673580854313);
    float64x2_t const coeff_7_f64x2 = vdupq_n_f64(-0.0666573579361080525984562);
    float64x2_t const coeff_6_f64x2 = vdupq_n_f64(+0.0769219538311769618355029);
    float64x2_t const coeff_5_f64x2 = vdupq_n_f64(-0.090908995008245008229153);
    float64x2_t const coeff_4_f64x2 = vdupq_n_f64(+0.111111105648261418443745);
    float64x2_t const coeff_3_f64x2 = vdupq_n_f64(-0.14285714266771329383765);
    float64x2_t const coeff_2_f64x2 = vdupq_n_f64(+0.199999999996591265594148);
    float64x2_t const coeff_1_f64x2 = vdupq_n_f64(-0.333333333333311110369124);
    float64x2_t const half_pi_f64x2 = vdupq_n_f64(1.5707963267948966);
    float64x2_t const zeros_f64x2 = vdupq_n_f64(0);

    // Detect negative and take absolute value
    uint64x2_t negative_mask_u64x2 = vcltq_f64(inputs, zeros_f64x2);
    float64x2_t values_f64x2 = vabsq_f64(inputs);

    // Check if values_f64x2 > 1 (need reciprocal) - use division for f64 precision
    uint64x2_t reciprocal_mask_u64x2 = vcgtq_f64(values_f64x2, vdupq_n_f64(1.0));
    float64x2_t reciprocal_values_f64x2 = vdivq_f64(vdupq_n_f64(1.0), values_f64x2);
    values_f64x2 = vbslq_f64(reciprocal_mask_u64x2, reciprocal_values_f64x2, values_f64x2);

    // Compute powers
    float64x2_t const values_sq_f64x2 = vmulq_f64(values_f64x2, values_f64x2);
    float64x2_t const values_cubed_f64x2 = vmulq_f64(values_f64x2, values_sq_f64x2);

    // Polynomial evaluation using Horner's method
    float64x2_t polynomials_f64x2 = coeff_19_f64x2;
    polynomials_f64x2 = vfmaq_f64(coeff_18_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_17_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_16_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_15_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_14_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_13_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_12_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_11_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_10_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_9_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_8_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_7_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_6_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_5_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_4_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_3_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_2_f64x2, polynomials_f64x2, values_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_1_f64x2, polynomials_f64x2, values_sq_f64x2);

    // Compute result_f64x2
    float64x2_t result_f64x2 = vfmaq_f64(values_f64x2, values_cubed_f64x2, polynomials_f64x2);

    // Adjust for reciprocal: result_f64x2 = π/2 - result_f64x2
    float64x2_t adjusted_f64x2 = vsubq_f64(half_pi_f64x2, result_f64x2);
    result_f64x2 = vbslq_f64(reciprocal_mask_u64x2, adjusted_f64x2, result_f64x2);

    // Adjust for negative: result_f64x2 = -result_f64x2
    float64x2_t negated_f64x2 = vnegq_f64(result_f64x2);
    result_f64x2 = vbslq_f64(negative_mask_u64x2, negated_f64x2, result_f64x2);
    return result_f64x2;
}

NK_HELPER_INLINE float64x2_t nk_atan2_f64x2_neon_(float64x2_t const ys_inputs, float64x2_t const xs_inputs) {
    // Polynomial coefficients (same as atan)
    float64x2_t const coeff_19_f64x2 = vdupq_n_f64(-1.88796008463073496563746e-05);
    float64x2_t const coeff_18_f64x2 = vdupq_n_f64(+0.000209850076645816976906797);
    float64x2_t const coeff_17_f64x2 = vdupq_n_f64(-0.00110611831486672482563471);
    float64x2_t const coeff_16_f64x2 = vdupq_n_f64(+0.00370026744188713119232403);
    float64x2_t const coeff_15_f64x2 = vdupq_n_f64(-0.00889896195887655491740809);
    float64x2_t const coeff_14_f64x2 = vdupq_n_f64(+0.016599329773529201970117);
    float64x2_t const coeff_13_f64x2 = vdupq_n_f64(-0.0254517624932312641616861);
    float64x2_t const coeff_12_f64x2 = vdupq_n_f64(+0.0337852580001353069993897);
    float64x2_t const coeff_11_f64x2 = vdupq_n_f64(-0.0407629191276836500001934);
    float64x2_t const coeff_10_f64x2 = vdupq_n_f64(+0.0466667150077840625632675);
    float64x2_t const coeff_9_f64x2 = vdupq_n_f64(-0.0523674852303482457616113);
    float64x2_t const coeff_8_f64x2 = vdupq_n_f64(+0.0587666392926673580854313);
    float64x2_t const coeff_7_f64x2 = vdupq_n_f64(-0.0666573579361080525984562);
    float64x2_t const coeff_6_f64x2 = vdupq_n_f64(+0.0769219538311769618355029);
    float64x2_t const coeff_5_f64x2 = vdupq_n_f64(-0.090908995008245008229153);
    float64x2_t const coeff_4_f64x2 = vdupq_n_f64(+0.111111105648261418443745);
    float64x2_t const coeff_3_f64x2 = vdupq_n_f64(-0.14285714266771329383765);
    float64x2_t const coeff_2_f64x2 = vdupq_n_f64(+0.199999999996591265594148);
    float64x2_t const coeff_1_f64x2 = vdupq_n_f64(-0.333333333333311110369124);
    float64x2_t const half_pi_f64x2 = vdupq_n_f64(1.5707963267948966);
    float64x2_t const zeros_f64x2 = vdupq_n_f64(0);

    // Quadrant adjustments - take absolute values
    uint64x2_t xs_negative_mask_u64x2 = vcltq_f64(xs_inputs, zeros_f64x2);
    float64x2_t xs_f64x2 = vabsq_f64(xs_inputs);
    float64x2_t ys_f64x2 = vabsq_f64(ys_inputs);

    // Ensure proper fraction where numerator < denominator
    uint64x2_t swap_mask_u64x2 = vcgtq_f64(ys_f64x2, xs_f64x2);
    float64x2_t temps_f64x2 = xs_f64x2;
    xs_f64x2 = vbslq_f64(swap_mask_u64x2, ys_f64x2, xs_f64x2);
    ys_f64x2 = vbslq_f64(swap_mask_u64x2, vnegq_f64(temps_f64x2), ys_f64x2);

    // Division for f64 precision
    float64x2_t const ratio_f64x2 = vdivq_f64(ys_f64x2, xs_f64x2);
    float64x2_t const ratio_sq_f64x2 = vmulq_f64(ratio_f64x2, ratio_f64x2);
    float64x2_t const ratio_cubed_f64x2 = vmulq_f64(ratio_f64x2, ratio_sq_f64x2);

    // Polynomial evaluation using Horner's method
    float64x2_t polynomials_f64x2 = coeff_19_f64x2;
    polynomials_f64x2 = vfmaq_f64(coeff_18_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_17_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_16_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_15_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_14_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_13_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_12_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_11_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_10_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_9_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_8_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_7_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_6_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_5_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_4_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_3_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_2_f64x2, polynomials_f64x2, ratio_sq_f64x2);
    polynomials_f64x2 = vfmaq_f64(coeff_1_f64x2, polynomials_f64x2, ratio_sq_f64x2);

    // Compute the result
    float64x2_t results_f64x2 = vfmaq_f64(ratio_f64x2, ratio_cubed_f64x2, polynomials_f64x2);

    // Compute quadrant_f64x2 value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    float64x2_t quadrant_f64x2 = vdupq_n_f64(0.0);
    float64x2_t neg_two_f64x2 = vdupq_n_f64(-2.0);
    quadrant_f64x2 = vbslq_f64(xs_negative_mask_u64x2, neg_two_f64x2, quadrant_f64x2);
    float64x2_t quadrant_incremented_f64x2 = vaddq_f64(quadrant_f64x2, vdupq_n_f64(1.0));
    quadrant_f64x2 = vbslq_f64(swap_mask_u64x2, quadrant_incremented_f64x2, quadrant_f64x2);

    // Adjust for quadrant_f64x2: result += quadrant_f64x2 * π/2
    results_f64x2 = vfmaq_f64(results_f64x2, quadrant_f64x2, half_pi_f64x2);

    // Transfer sign from x and y by XOR with sign bits
    uint64x2_t sign_mask_u64x2 = vreinterpretq_u64_f64(vdupq_n_f64(-0.0));
    uint64x2_t xs_sign_u64x2 = vandq_u64(vreinterpretq_u64_f64(xs_inputs), sign_mask_u64x2);
    uint64x2_t ys_sign_u64x2 = vandq_u64(vreinterpretq_u64_f64(ys_inputs), sign_mask_u64x2);
    uint64x2_t result_u64x2 = vreinterpretq_u64_f64(results_f64x2);
    result_u64x2 = veorq_u64(result_u64x2, xs_sign_u64x2);
    result_u64x2 = veorq_u64(result_u64x2, ys_sign_u64x2);
    results_f64x2 = vreinterpretq_f64_u64(result_u64x2);

    return results_f64x2;
}

NK_API_COMPTIME void nk_trig_sin_f32_neon(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t angles_f32x4 = vld1q_f32(ins + i);
        float32x4_t results_f32x4 = nk_sin_f32x4_neon_(angles_f32x4);
        vst1q_f32(outs + i, results_f32x4);
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

NK_API_COMPTIME void nk_trig_cos_f32_neon(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t angles_f32x4 = vld1q_f32(ins + i);
        float32x4_t results_f32x4 = nk_cos_f32x4_neon_(angles_f32x4);
        vst1q_f32(outs + i, results_f32x4);
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

NK_API_COMPTIME void nk_trig_atan_f32_neon(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t values_f32x4 = vld1q_f32(ins + i);
        float32x4_t results_f32x4 = nk_atan_f32x4_neon_(values_f32x4);
        vst1q_f32(outs + i, results_f32x4);
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

NK_API_COMPTIME void nk_trig_sin_f64_neon(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t angles_f64x2 = vld1q_f64(ins + i);
        float64x2_t results_f64x2 = nk_sin_f64x2_neon_(angles_f64x2);
        vst1q_f64(outs + i, results_f64x2);
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

NK_API_COMPTIME void nk_trig_cos_f64_neon(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t angles_f64x2 = vld1q_f64(ins + i);
        float64x2_t results_f64x2 = nk_cos_f64x2_neon_(angles_f64x2);
        vst1q_f64(outs + i, results_f64x2);
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

NK_API_COMPTIME void nk_trig_atan_f64_neon(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t values_f64x2 = vld1q_f64(ins + i);
        float64x2_t results_f64x2 = nk_atan_f64x2_neon_(values_f64x2);
        vst1q_f64(outs + i, results_f64x2);
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
#endif // NK_TARGET_ARM64_
#endif // NK_TRIGONOMETRY_NEON_H
