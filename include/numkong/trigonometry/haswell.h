/**
 *  @brief SIMD-accelerated Trigonometric Functions for Haswell.
 *  @file include/numkong/trigonometry/haswell.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/trigonometry.h
 *  @see https://sleef.org
 *
 *  @section haswell_trig_instructions Key AVX2 Trigonometry Instructions
 *
 *      Intrinsic            Instruction                   Haswell     Genoa
 *      _mm256_fmadd_ps/pd   VFMADD (YMM, YMM, YMM)        5cy @ p01   4cy @ p01
 *      _mm256_mul_ps/pd     VMULPS/PD (YMM, YMM, YMM)     5cy @ p01   3cy @ p01
 *      _mm256_blendv_ps/pd  VBLENDVPS/PD (YMM, YMM, YMM)  2cy @ p015  1cy @ p01
 *      _mm256_round_ps/pd   VROUNDPS/PD (YMM, YMM, I8)    6cy @ p01   3cy @ p23
 *      _mm256_div_ps        VDIVPS (YMM, YMM, YMM)        13cy @ p0   11cy @ p01
 *
 *  Polynomial evaluation uses Horner's method with FMA for sin/cos/atan approximation. For large
 *  arrays, out-of-order execution across loop iterations hides FMA latency better than Estrin's
 *  scheme. Range reduction uses argument folding modulo pi with high/low precision constants.
 */
#ifndef NK_TRIGONOMETRY_HASWELL_H
#define NK_TRIGONOMETRY_HASWELL_H

#if NK_TARGET_X8664_
#if NK_TARGET_HASWELL

#include "numkong/types.h"
#include "numkong/reduce/haswell.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

/*  Haswell AVX2 trigonometry kernels (8-way f32, 4-way f64)
 *  These implement the same polynomial approximations as Skylake but with 256-bit vectors.
 */

NK_INTERNAL __m256 nk_sin_f32x8_haswell_(__m256 const angles_radians) {
    // Cody-Waite constants for argument reduction
    __m256 const pi_high_f32x8 = _mm256_set1_ps(3.1415927f);
    __m256 const pi_low_f32x8 = _mm256_set1_ps(-8.742278e-8f);
    __m256 const pi_reciprocal_f32x8 = _mm256_set1_ps(0.31830988618379067154f); // 1/π
    // Degree-9 minimax coefficients
    __m256 const coeff_9_f32x8 = _mm256_set1_ps(+2.7557319224e-6f);
    __m256 const coeff_7_f32x8 = _mm256_set1_ps(-1.9841269841e-4f);
    __m256 const coeff_5_f32x8 = _mm256_set1_ps(+8.3333293855e-3f);
    __m256 const coeff_3_f32x8 = _mm256_set1_ps(-1.6666666641e-1f);

    // Compute (multiples_of_pi_i32x8) = round(angle / π)
    __m256 quotients_f32x8 = _mm256_mul_ps(angles_radians, pi_reciprocal_f32x8);
    __m256 rounded_quotients_f32x8 = _mm256_round_ps(quotients_f32x8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // Use truncation (MXCSR-independent) since rounded_quotients_f32x8 is already integer-valued
    __m256i multiples_of_pi_i32x8 = _mm256_cvttps_epi32(rounded_quotients_f32x8);

    // Cody-Waite range reduction
    __m256 angles_f32x8 = _mm256_fnmadd_ps(rounded_quotients_f32x8, pi_high_f32x8, angles_radians);
    angles_f32x8 = _mm256_fnmadd_ps(rounded_quotients_f32x8, pi_low_f32x8, angles_f32x8);
    __m256 const angles_squared_f32x8 = _mm256_mul_ps(angles_f32x8, angles_f32x8);
    __m256 const angles_cubed_f32x8 = _mm256_mul_ps(angles_f32x8, angles_squared_f32x8);

    // Degree-9 polynomial via Horner's method
    __m256 polynomials_f32x8 = coeff_9_f32x8;
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, angles_squared_f32x8, coeff_7_f32x8);
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, angles_squared_f32x8, coeff_5_f32x8);
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, angles_squared_f32x8, coeff_3_f32x8);
    __m256 results_f32x8 = _mm256_fmadd_ps(angles_cubed_f32x8, polynomials_f32x8, angles_f32x8);

    // If multiples_of_pi_i32x8 is odd, flip the sign of the results_f32x8
    __m256i parity_i32x8 = _mm256_and_si256(multiples_of_pi_i32x8, _mm256_set1_epi32(1));
    __m256i odd_mask_i32x8 = _mm256_cmpeq_epi32(parity_i32x8, _mm256_set1_epi32(1));
    __m256 float_mask_f32x8 = _mm256_castsi256_ps(odd_mask_i32x8);
    __m256 negated_f32x8 = _mm256_sub_ps(_mm256_setzero_ps(), results_f32x8);
    results_f32x8 = _mm256_blendv_ps(results_f32x8, negated_f32x8, float_mask_f32x8);
    return results_f32x8;
}

NK_INTERNAL __m256 nk_cos_f32x8_haswell_(__m256 const angles_radians) {
    // Cody-Waite constants for argument reduction
    __m256 const pi_high_f32x8 = _mm256_set1_ps(3.1415927f);
    __m256 const pi_low_f32x8 = _mm256_set1_ps(-8.742278e-8f);
    __m256 const pi_half_f32x8 = _mm256_set1_ps(1.57079632679489661923f);       // π/2
    __m256 const pi_reciprocal_f32x8 = _mm256_set1_ps(0.31830988618379067154f); // 1/π
    // Degree-9 minimax coefficients
    __m256 const coeff_9_f32x8 = _mm256_set1_ps(+2.7557319224e-6f);
    __m256 const coeff_7_f32x8 = _mm256_set1_ps(-1.9841269841e-4f);
    __m256 const coeff_5_f32x8 = _mm256_set1_ps(+8.3333293855e-3f);
    __m256 const coeff_3_f32x8 = _mm256_set1_ps(-1.6666666641e-1f);

    // Compute (multiples_of_pi_i32x8) = round((angle / π) - 0.5)
    __m256 quotients_f32x8 = _mm256_fmsub_ps(angles_radians, pi_reciprocal_f32x8, _mm256_set1_ps(0.5f));
    __m256 rounded_quotients_f32x8 = _mm256_round_ps(quotients_f32x8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // Use truncation (MXCSR-independent) since rounded_quotients_f32x8 is already integer-valued
    __m256i multiples_of_pi_i32x8 = _mm256_cvttps_epi32(rounded_quotients_f32x8);

    // Cody-Waite range reduction: angle = angle_radians - (multiples * pi + pi/2)
    __m256 const offset_f32x8 = _mm256_fmadd_ps(rounded_quotients_f32x8, pi_high_f32x8, pi_half_f32x8);
    __m256 angles_f32x8 = _mm256_sub_ps(angles_radians, offset_f32x8);
    angles_f32x8 = _mm256_fnmadd_ps(rounded_quotients_f32x8, pi_low_f32x8, angles_f32x8);
    __m256 const angles_squared_f32x8 = _mm256_mul_ps(angles_f32x8, angles_f32x8);
    __m256 const angles_cubed_f32x8 = _mm256_mul_ps(angles_f32x8, angles_squared_f32x8);

    // Degree-9 polynomial via Horner's method
    __m256 polynomials_f32x8 = coeff_9_f32x8;
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, angles_squared_f32x8, coeff_7_f32x8);
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, angles_squared_f32x8, coeff_5_f32x8);
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, angles_squared_f32x8, coeff_3_f32x8);
    __m256 results_f32x8 = _mm256_fmadd_ps(angles_cubed_f32x8, polynomials_f32x8, angles_f32x8);

    // If multiples_of_pi_i32x8 is even, flip the sign of the results_f32x8
    __m256i parity_i32x8 = _mm256_and_si256(multiples_of_pi_i32x8, _mm256_set1_epi32(1));
    __m256i even_mask_i32x8 = _mm256_cmpeq_epi32(parity_i32x8, _mm256_setzero_si256());
    __m256 float_mask_f32x8 = _mm256_castsi256_ps(even_mask_i32x8);
    __m256 negated_f32x8 = _mm256_sub_ps(_mm256_setzero_ps(), results_f32x8);
    results_f32x8 = _mm256_blendv_ps(results_f32x8, negated_f32x8, float_mask_f32x8);
    return results_f32x8;
}

NK_INTERNAL __m256 nk_atan_f32x8_haswell_(__m256 const inputs) {
    // Polynomial coefficients for atan approximation (8 terms)
    // These coefficients approximate: atan(x) ≈ x + c8 × x³ + c7 × x⁵ + c6 × x⁷ + ... + c1 × x¹⁵
    __m256 const coeff_8_f32x8 = _mm256_set1_ps(-0.333331018686294555664062f);
    __m256 const coeff_7_f32x8 = _mm256_set1_ps(+0.199926957488059997558594f);
    __m256 const coeff_6_f32x8 = _mm256_set1_ps(-0.142027363181114196777344f);
    __m256 const coeff_5_f32x8 = _mm256_set1_ps(+0.106347933411598205566406f);
    __m256 const coeff_4_f32x8 = _mm256_set1_ps(-0.0748900920152664184570312f);
    __m256 const coeff_3_f32x8 = _mm256_set1_ps(+0.0425049886107444763183594f);
    __m256 const coeff_2_f32x8 = _mm256_set1_ps(-0.0159569028764963150024414f);
    __m256 const coeff_1_f32x8 = _mm256_set1_ps(+0.00282363896258175373077393f);
    __m256 const sign_mask_f32x8 = _mm256_set1_ps(-0.0f);

    // Adjust for quadrant - detect negative values_f32x8
    __m256 values_f32x8 = inputs;
    __m256 negative_mask_f32x8 = _mm256_cmp_ps(values_f32x8, _mm256_setzero_ps(), _CMP_LT_OS);
    values_f32x8 = _mm256_andnot_ps(sign_mask_f32x8, values_f32x8); // abs(values_f32x8)

    // Check if values_f32x8 > 1 (need reciprocal)
    __m256 reciprocal_mask_f32x8 = _mm256_cmp_ps(values_f32x8, _mm256_set1_ps(1.0f), _CMP_GT_OS);
    __m256 reciprocal_values_f32x8 = _mm256_div_ps(_mm256_set1_ps(1.0f), values_f32x8);
    values_f32x8 = _mm256_blendv_ps(values_f32x8, reciprocal_values_f32x8, reciprocal_mask_f32x8);

    // Argument reduction
    __m256 const values_squared_f32x8 = _mm256_mul_ps(values_f32x8, values_f32x8);
    __m256 const values_cubed_f32x8 = _mm256_mul_ps(values_f32x8, values_squared_f32x8);

    // Polynomial evaluation using Horner's method.
    // For large arrays, out-of-order execution across loop iterations already hides
    // FMA latency. Estrin's scheme was tested but showed ~20% regression because
    // the extra power computations (y², y⁴) hurt throughput more than the reduced
    // dependency depth helps latency.
    __m256 polynomials_f32x8 = coeff_1_f32x8;
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, values_squared_f32x8, coeff_2_f32x8);
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, values_squared_f32x8, coeff_3_f32x8);
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, values_squared_f32x8, coeff_4_f32x8);
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, values_squared_f32x8, coeff_5_f32x8);
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, values_squared_f32x8, coeff_6_f32x8);
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, values_squared_f32x8, coeff_7_f32x8);
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, values_squared_f32x8, coeff_8_f32x8);

    // Compute result_f32x8: atan(x) ≈ x + x³ * P(x²)
    __m256 result_f32x8 = _mm256_fmadd_ps(values_cubed_f32x8, polynomials_f32x8, values_f32x8);

    // Adjust for reciprocal: result_f32x8 = π/2 - result_f32x8
    __m256 adjusted_f32x8 = _mm256_sub_ps(_mm256_set1_ps(1.5707963267948966f), result_f32x8);
    result_f32x8 = _mm256_blendv_ps(result_f32x8, adjusted_f32x8, reciprocal_mask_f32x8);

    // Adjust for negative: result_f32x8 = -result_f32x8
    __m256 negated_f32x8 = _mm256_sub_ps(_mm256_setzero_ps(), result_f32x8);
    result_f32x8 = _mm256_blendv_ps(result_f32x8, negated_f32x8, negative_mask_f32x8);
    return result_f32x8;
}

NK_INTERNAL __m256 nk_atan2_f32x8_haswell_(__m256 const ys_inputs, __m256 const xs_inputs) {
    // Polynomial coefficients (same as atan)
    __m256 const coeff_8_f32x8 = _mm256_set1_ps(-0.333331018686294555664062f);
    __m256 const coeff_7_f32x8 = _mm256_set1_ps(+0.199926957488059997558594f);
    __m256 const coeff_6_f32x8 = _mm256_set1_ps(-0.142027363181114196777344f);
    __m256 const coeff_5_f32x8 = _mm256_set1_ps(+0.106347933411598205566406f);
    __m256 const coeff_4_f32x8 = _mm256_set1_ps(-0.0748900920152664184570312f);
    __m256 const coeff_3_f32x8 = _mm256_set1_ps(+0.0425049886107444763183594f);
    __m256 const coeff_2_f32x8 = _mm256_set1_ps(-0.0159569028764963150024414f);
    __m256 const coeff_1_f32x8 = _mm256_set1_ps(+0.00282363896258175373077393f);
    __m256 const sign_mask_f32x8 = _mm256_set1_ps(-0.0f);

    // Quadrant adjustments normalizing to absolute values of x and y
    __m256 xs_negative_mask_f32x8 = _mm256_cmp_ps(xs_inputs, _mm256_setzero_ps(), _CMP_LT_OS);
    __m256 xs_f32x8 = _mm256_andnot_ps(sign_mask_f32x8, xs_inputs); // abs(xs_inputs)
    __m256 ys_f32x8 = _mm256_andnot_ps(sign_mask_f32x8, ys_inputs); // abs(ys_inputs)

    // Ensure proper fraction where the numerator is smaller than the denominator
    __m256 swap_mask_f32x8 = _mm256_cmp_ps(ys_f32x8, xs_f32x8, _CMP_GT_OS);
    __m256 temps_f32x8 = xs_f32x8;
    xs_f32x8 = _mm256_blendv_ps(xs_f32x8, ys_f32x8, swap_mask_f32x8);
    __m256 neg_temps_f32x8 = _mm256_sub_ps(_mm256_setzero_ps(), temps_f32x8);
    ys_f32x8 = _mm256_blendv_ps(ys_f32x8, neg_temps_f32x8, swap_mask_f32x8);

    // Compute ratio_f32x8 and powers
    __m256 const ratio_f32x8 = _mm256_div_ps(ys_f32x8, xs_f32x8);
    __m256 const ratio_squared_f32x8 = _mm256_mul_ps(ratio_f32x8, ratio_f32x8);
    __m256 const ratio_cubed_f32x8 = _mm256_mul_ps(ratio_f32x8, ratio_squared_f32x8);

    // Polynomial evaluation using Horner's method
    __m256 polynomials_f32x8 = coeff_1_f32x8;
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, ratio_squared_f32x8, coeff_2_f32x8);
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, ratio_squared_f32x8, coeff_3_f32x8);
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, ratio_squared_f32x8, coeff_4_f32x8);
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, ratio_squared_f32x8, coeff_5_f32x8);
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, ratio_squared_f32x8, coeff_6_f32x8);
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, ratio_squared_f32x8, coeff_7_f32x8);
    polynomials_f32x8 = _mm256_fmadd_ps(polynomials_f32x8, ratio_squared_f32x8, coeff_8_f32x8);

    // Compute the result using masks for quadrant_f32x8 adjustments
    __m256 results_f32x8 = _mm256_fmadd_ps(ratio_cubed_f32x8, polynomials_f32x8, ratio_f32x8);

    // Compute quadrant_f32x8 value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    __m256 quadrant_f32x8 = _mm256_setzero_ps();
    __m256 neg_two_f32x8 = _mm256_set1_ps(-2.0f);
    quadrant_f32x8 = _mm256_blendv_ps(quadrant_f32x8, neg_two_f32x8, xs_negative_mask_f32x8);
    __m256 one_f32x8 = _mm256_set1_ps(1.0f);
    __m256 quadrant_incremented_f32x8 = _mm256_add_ps(quadrant_f32x8, one_f32x8);
    quadrant_f32x8 = _mm256_blendv_ps(quadrant_f32x8, quadrant_incremented_f32x8, swap_mask_f32x8);

    // Adjust for quadrant_f32x8: result += quadrant_f32x8 * π/2
    __m256 pi_half_f32x8 = _mm256_set1_ps(1.5707963267948966f);
    results_f32x8 = _mm256_fmadd_ps(quadrant_f32x8, pi_half_f32x8, results_f32x8);

    // Transfer sign from x (XOR with sign bit of x_input)
    __m256 xs_sign_bits_f32x8 = _mm256_and_ps(xs_inputs, sign_mask_f32x8);
    results_f32x8 = _mm256_xor_ps(results_f32x8, xs_sign_bits_f32x8);

    // Transfer sign from y (XOR with sign bit of y_input)
    __m256 ys_sign_bits_f32x8 = _mm256_and_ps(ys_inputs, sign_mask_f32x8);
    results_f32x8 = _mm256_xor_ps(results_f32x8, ys_sign_bits_f32x8);

    return results_f32x8;
}

NK_INTERNAL __m256d nk_sin_f64x4_haswell_(__m256d const angles_radians) {
    // Constants for argument reduction
    __m256d const pi_high_f64x4 = _mm256_set1_pd(3.141592653589793116);         // High-digits part of π
    __m256d const pi_low_f64x4 = _mm256_set1_pd(1.2246467991473532072e-16);     // Low-digits part of π
    __m256d const pi_reciprocal_f64x4 = _mm256_set1_pd(0.31830988618379067154); // 1/π

    // Polynomial coefficients for sine approximation (minimax polynomial)
    __m256d const coeff_0_f64x4 = _mm256_set1_pd(+0.00833333333333332974823815);
    __m256d const coeff_1_f64x4 = _mm256_set1_pd(-0.000198412698412696162806809);
    __m256d const coeff_2_f64x4 = _mm256_set1_pd(+2.75573192239198747630416e-06);
    __m256d const coeff_3_f64x4 = _mm256_set1_pd(-2.50521083763502045810755e-08);
    __m256d const coeff_4_f64x4 = _mm256_set1_pd(+1.60590430605664501629054e-10);
    __m256d const coeff_5_f64x4 = _mm256_set1_pd(-7.64712219118158833288484e-13);
    __m256d const coeff_6_f64x4 = _mm256_set1_pd(+2.81009972710863200091251e-15);
    __m256d const coeff_7_f64x4 = _mm256_set1_pd(-7.97255955009037868891952e-18);
    __m256d const coeff_8_f64x4 = _mm256_set1_pd(-0.166666666666666657414808);

    // Compute (rounded_quotients_f64x4) = round(angle / π)
    __m256d const quotients_f64x4 = _mm256_mul_pd(angles_radians, pi_reciprocal_f64x4);
    __m256d const rounded_quotients_f64x4 = _mm256_round_pd(quotients_f64x4,
                                                            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Reduce the angle: angle - (rounded_quotients_f64x4 * π_high + rounded_quotients_f64x4 * π_low)
    __m256d angles_f64x4 = angles_radians;
    angles_f64x4 = _mm256_fnmadd_pd(rounded_quotients_f64x4, pi_high_f64x4, angles_f64x4);
    angles_f64x4 = _mm256_fnmadd_pd(rounded_quotients_f64x4, pi_low_f64x4, angles_f64x4);

    // If rounded_quotients_f64x4 is odd (bit 0 set), negate the angle
    // Convert to 32-bit int (returns __m128i with 4 x 32-bit ints)
    // Use truncation (MXCSR-independent) since rounded_quotients_f64x4 is already integer-valued
    __m128i quotients_i32_i32x4 = _mm256_cvttpd_epi32(rounded_quotients_f64x4);
    __m128i parity_i32x4 = _mm_and_si128(quotients_i32_i32x4, _mm_set1_epi32(1));
    __m128i odd_mask_i32_i32x4 = _mm_cmpeq_epi32(parity_i32x4, _mm_set1_epi32(1));
    // Expand 32-bit mask to 64-bit by shuffling
    __m256i odd_mask_i64_i32x8 = _mm256_cvtepi32_epi64(odd_mask_i32_i32x4);
    __m256d float_mask_f64x4 = _mm256_castsi256_pd(odd_mask_i64_i32x8);
    __m256d negated_angles_f64x4 = _mm256_sub_pd(_mm256_setzero_pd(), angles_f64x4);
    angles_f64x4 = _mm256_blendv_pd(angles_f64x4, negated_angles_f64x4, float_mask_f64x4);

    __m256d const angles_squared_f64x4 = _mm256_mul_pd(angles_f64x4, angles_f64x4);
    __m256d const angles_cubed_f64x4 = _mm256_mul_pd(angles_f64x4, angles_squared_f64x4);
    __m256d const angles_quadratic_f64x4 = _mm256_mul_pd(angles_squared_f64x4, angles_squared_f64x4);
    __m256d const angles_octic_f64x4 = _mm256_mul_pd(angles_quadratic_f64x4, angles_quadratic_f64x4);

    // Compute higher-degree polynomial terms
    __m256d const poly_67_f64x4 = _mm256_fmadd_pd(angles_squared_f64x4, coeff_7_f64x4, coeff_6_f64x4);
    __m256d const poly_45_f64x4 = _mm256_fmadd_pd(angles_squared_f64x4, coeff_5_f64x4, coeff_4_f64x4);
    __m256d const poly_4567_f64x4 = _mm256_fmadd_pd(angles_quadratic_f64x4, poly_67_f64x4, poly_45_f64x4);

    // Compute lower-degree polynomial terms
    __m256d const poly_23_f64x4 = _mm256_fmadd_pd(angles_squared_f64x4, coeff_3_f64x4, coeff_2_f64x4);
    __m256d const poly_01_f64x4 = _mm256_fmadd_pd(angles_squared_f64x4, coeff_1_f64x4, coeff_0_f64x4);
    __m256d const poly_0123_f64x4 = _mm256_fmadd_pd(angles_quadratic_f64x4, poly_23_f64x4, poly_01_f64x4);

    // Combine polynomial terms
    __m256d results_f64x4 = _mm256_fmadd_pd(angles_octic_f64x4, poly_4567_f64x4, poly_0123_f64x4);
    results_f64x4 = _mm256_fmadd_pd(results_f64x4, angles_squared_f64x4, coeff_8_f64x4);
    results_f64x4 = _mm256_fmadd_pd(results_f64x4, angles_cubed_f64x4, angles_f64x4);

    // Handle the special case of negative zero input
    __m256d const non_zero_mask_f64x4 = _mm256_cmp_pd(angles_radians, _mm256_setzero_pd(), _CMP_NEQ_UQ);
    results_f64x4 = _mm256_and_pd(results_f64x4, non_zero_mask_f64x4);
    return results_f64x4;
}

NK_INTERNAL __m256d nk_cos_f64x4_haswell_(__m256d const angles_radians) {
    // Constants for argument reduction
    __m256d const pi_high_half_f64x4 = _mm256_set1_pd(3.141592653589793116 * 0.5);     // High-digits part of π/2
    __m256d const pi_low_half_f64x4 = _mm256_set1_pd(1.2246467991473532072e-16 * 0.5); // Low-digits part of π/2
    __m256d const pi_reciprocal_f64x4 = _mm256_set1_pd(0.31830988618379067154);        // 1/π

    // Polynomial coefficients for cosine approximation
    __m256d const coeff_0_f64x4 = _mm256_set1_pd(+0.00833333333333332974823815);
    __m256d const coeff_1_f64x4 = _mm256_set1_pd(-0.000198412698412696162806809);
    __m256d const coeff_2_f64x4 = _mm256_set1_pd(+2.75573192239198747630416e-06);
    __m256d const coeff_3_f64x4 = _mm256_set1_pd(-2.50521083763502045810755e-08);
    __m256d const coeff_4_f64x4 = _mm256_set1_pd(+1.60590430605664501629054e-10);
    __m256d const coeff_5_f64x4 = _mm256_set1_pd(-7.64712219118158833288484e-13);
    __m256d const coeff_6_f64x4 = _mm256_set1_pd(+2.81009972710863200091251e-15);
    __m256d const coeff_7_f64x4 = _mm256_set1_pd(-7.97255955009037868891952e-18);
    __m256d const coeff_8_f64x4 = _mm256_set1_pd(-0.166666666666666657414808);

    // Compute (rounded_quotients_f64x4) = 2 * round(angle / π - 0.5) + 1
    // Use fmsub: a*b - c = angles_f64x4 * (1/π) - 0.5
    __m256d const quotients_f64x4 = _mm256_fmsub_pd(angles_radians, pi_reciprocal_f64x4, _mm256_set1_pd(0.5));
    __m256d const rounded_quotients_f64x4 = _mm256_fmadd_pd(                             //
        _mm256_set1_pd(2.0),                                                             //
        _mm256_round_pd(quotients_f64x4, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), //
        _mm256_set1_pd(1.0));

    // Reduce the angle: angle - (rounded_quotients_f64x4 * π_high_half + rounded_quotients_f64x4 * π_low_half)
    __m256d angles_f64x4 = angles_radians;
    angles_f64x4 = _mm256_fnmadd_pd(rounded_quotients_f64x4, pi_high_half_f64x4, angles_f64x4);
    angles_f64x4 = _mm256_fnmadd_pd(rounded_quotients_f64x4, pi_low_half_f64x4, angles_f64x4);

    // If (rounded_quotients_f64x4 & 2) == 0, negate the angle
    // Use truncation (MXCSR-independent) since rounded_quotients_f64x4 is already integer-valued
    __m128i quotients_i32_i32x4 = _mm256_cvttpd_epi32(rounded_quotients_f64x4);
    __m128i bit2_i32x4 = _mm_and_si128(quotients_i32_i32x4, _mm_set1_epi32(2));
    __m128i flip_mask_i32_i32x4 = _mm_cmpeq_epi32(bit2_i32x4, _mm_setzero_si128());
    __m256i flip_mask_i64_i32x8 = _mm256_cvtepi32_epi64(flip_mask_i32_i32x4);
    __m256d float_mask_f64x4 = _mm256_castsi256_pd(flip_mask_i64_i32x8);
    __m256d negated_angles_f64x4 = _mm256_sub_pd(_mm256_setzero_pd(), angles_f64x4);
    angles_f64x4 = _mm256_blendv_pd(angles_f64x4, negated_angles_f64x4, float_mask_f64x4);

    __m256d const angles_squared_f64x4 = _mm256_mul_pd(angles_f64x4, angles_f64x4);
    __m256d const angles_cubed_f64x4 = _mm256_mul_pd(angles_f64x4, angles_squared_f64x4);
    __m256d const angles_quadratic_f64x4 = _mm256_mul_pd(angles_squared_f64x4, angles_squared_f64x4);
    __m256d const angles_octic_f64x4 = _mm256_mul_pd(angles_quadratic_f64x4, angles_quadratic_f64x4);

    // Compute higher-degree polynomial terms
    __m256d const poly_67_f64x4 = _mm256_fmadd_pd(angles_squared_f64x4, coeff_7_f64x4, coeff_6_f64x4);
    __m256d const poly_45_f64x4 = _mm256_fmadd_pd(angles_squared_f64x4, coeff_5_f64x4, coeff_4_f64x4);
    __m256d const poly_4567_f64x4 = _mm256_fmadd_pd(angles_quadratic_f64x4, poly_67_f64x4, poly_45_f64x4);

    // Compute lower-degree polynomial terms
    __m256d const poly_23_f64x4 = _mm256_fmadd_pd(angles_squared_f64x4, coeff_3_f64x4, coeff_2_f64x4);
    __m256d const poly_01_f64x4 = _mm256_fmadd_pd(angles_squared_f64x4, coeff_1_f64x4, coeff_0_f64x4);
    __m256d const poly_0123_f64x4 = _mm256_fmadd_pd(angles_quadratic_f64x4, poly_23_f64x4, poly_01_f64x4);

    // Combine polynomial terms
    __m256d results_f64x4 = _mm256_fmadd_pd(angles_octic_f64x4, poly_4567_f64x4, poly_0123_f64x4);
    results_f64x4 = _mm256_fmadd_pd(results_f64x4, angles_squared_f64x4, coeff_8_f64x4);
    results_f64x4 = _mm256_fmadd_pd(results_f64x4, angles_cubed_f64x4, angles_f64x4);
    return results_f64x4;
}

NK_INTERNAL __m256d nk_atan_f64x4_haswell_(__m256d const inputs) {
    // Polynomial coefficients for atan approximation (19 coefficients)
    // The polynomial approximates: atan(x) ≈ x + x³ * P(x²) where P has 19 terms
    __m256d const coeff_19_f64x4 = _mm256_set1_pd(-1.88796008463073496563746e-05);
    __m256d const coeff_18_f64x4 = _mm256_set1_pd(+0.000209850076645816976906797);
    __m256d const coeff_17_f64x4 = _mm256_set1_pd(-0.00110611831486672482563471);
    __m256d const coeff_16_f64x4 = _mm256_set1_pd(+0.00370026744188713119232403);
    __m256d const coeff_15_f64x4 = _mm256_set1_pd(-0.00889896195887655491740809);
    __m256d const coeff_14_f64x4 = _mm256_set1_pd(+0.016599329773529201970117);
    __m256d const coeff_13_f64x4 = _mm256_set1_pd(-0.0254517624932312641616861);
    __m256d const coeff_12_f64x4 = _mm256_set1_pd(+0.0337852580001353069993897);
    __m256d const coeff_11_f64x4 = _mm256_set1_pd(-0.0407629191276836500001934);
    __m256d const coeff_10_f64x4 = _mm256_set1_pd(+0.0466667150077840625632675);
    __m256d const coeff_9_f64x4 = _mm256_set1_pd(-0.0523674852303482457616113);
    __m256d const coeff_8_f64x4 = _mm256_set1_pd(+0.0587666392926673580854313);
    __m256d const coeff_7_f64x4 = _mm256_set1_pd(-0.0666573579361080525984562);
    __m256d const coeff_6_f64x4 = _mm256_set1_pd(+0.0769219538311769618355029);
    __m256d const coeff_5_f64x4 = _mm256_set1_pd(-0.090908995008245008229153);
    __m256d const coeff_4_f64x4 = _mm256_set1_pd(+0.111111105648261418443745);
    __m256d const coeff_3_f64x4 = _mm256_set1_pd(-0.14285714266771329383765);
    __m256d const coeff_2_f64x4 = _mm256_set1_pd(+0.199999999996591265594148);
    __m256d const coeff_1_f64x4 = _mm256_set1_pd(-0.333333333333311110369124);
    __m256d const sign_mask_f64x4 = _mm256_set1_pd(-0.0);

    // Adjust for quadrant - detect negative values_f64x4
    __m256d values_f64x4 = inputs;
    __m256d negative_mask_f64x4 = _mm256_cmp_pd(values_f64x4, _mm256_setzero_pd(), _CMP_LT_OS);
    values_f64x4 = _mm256_andnot_pd(sign_mask_f64x4, values_f64x4); // abs(values_f64x4)

    // Check if values_f64x4 > 1 (need reciprocal)
    // Note: For f64, we keep VDIVPD since RCPPD doesn't exist and Newton-Raphson
    // would need 2 iterations for sufficient precision (~44 bits needed for f64)
    __m256d reciprocal_mask_f64x4 = _mm256_cmp_pd(values_f64x4, _mm256_set1_pd(1.0), _CMP_GT_OS);
    __m256d reciprocal_values_f64x4 = _mm256_div_pd(_mm256_set1_pd(1.0), values_f64x4);
    values_f64x4 = _mm256_blendv_pd(values_f64x4, reciprocal_values_f64x4, reciprocal_mask_f64x4);

    // Argument reduction
    __m256d const values_squared_f64x4 = _mm256_mul_pd(values_f64x4, values_f64x4);
    __m256d const values_cubed_f64x4 = _mm256_mul_pd(values_f64x4, values_squared_f64x4);

    // Polynomial evaluation using Horner's method.
    // For large arrays, out-of-order execution across loop iterations already hides
    // FMA latency. Estrin's scheme was tested but showed minimal improvement (~1%)
    // while adding complexity. Keeping Horner for maintainability.
    __m256d polynomials_f64x4 = coeff_19_f64x4;
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_18_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_17_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_16_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_15_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_14_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_13_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_12_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_11_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_10_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_9_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_8_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_7_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_6_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_5_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_4_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_3_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_2_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, values_squared_f64x4, coeff_1_f64x4);

    // Compute result_f64x4
    __m256d result_f64x4 = _mm256_fmadd_pd(values_cubed_f64x4, polynomials_f64x4, values_f64x4);

    // Adjust for reciprocal: result_f64x4 = π/2 - result_f64x4
    __m256d adjusted_f64x4 = _mm256_sub_pd(_mm256_set1_pd(1.5707963267948966), result_f64x4);
    result_f64x4 = _mm256_blendv_pd(result_f64x4, adjusted_f64x4, reciprocal_mask_f64x4);

    // Adjust for negative: result_f64x4 = -result_f64x4
    __m256d negated_f64x4 = _mm256_sub_pd(_mm256_setzero_pd(), result_f64x4);
    result_f64x4 = _mm256_blendv_pd(result_f64x4, negated_f64x4, negative_mask_f64x4);
    return result_f64x4;
}

NK_INTERNAL __m256d nk_atan2_f64x4_haswell_(__m256d const ys_inputs, __m256d const xs_inputs) {
    // Polynomial coefficients for atan approximation (19 coefficients, same as atan)
    __m256d const coeff_19_f64x4 = _mm256_set1_pd(-1.88796008463073496563746e-05);
    __m256d const coeff_18_f64x4 = _mm256_set1_pd(+0.000209850076645816976906797);
    __m256d const coeff_17_f64x4 = _mm256_set1_pd(-0.00110611831486672482563471);
    __m256d const coeff_16_f64x4 = _mm256_set1_pd(+0.00370026744188713119232403);
    __m256d const coeff_15_f64x4 = _mm256_set1_pd(-0.00889896195887655491740809);
    __m256d const coeff_14_f64x4 = _mm256_set1_pd(+0.016599329773529201970117);
    __m256d const coeff_13_f64x4 = _mm256_set1_pd(-0.0254517624932312641616861);
    __m256d const coeff_12_f64x4 = _mm256_set1_pd(+0.0337852580001353069993897);
    __m256d const coeff_11_f64x4 = _mm256_set1_pd(-0.0407629191276836500001934);
    __m256d const coeff_10_f64x4 = _mm256_set1_pd(+0.0466667150077840625632675);
    __m256d const coeff_9_f64x4 = _mm256_set1_pd(-0.0523674852303482457616113);
    __m256d const coeff_8_f64x4 = _mm256_set1_pd(+0.0587666392926673580854313);
    __m256d const coeff_7_f64x4 = _mm256_set1_pd(-0.0666573579361080525984562);
    __m256d const coeff_6_f64x4 = _mm256_set1_pd(+0.0769219538311769618355029);
    __m256d const coeff_5_f64x4 = _mm256_set1_pd(-0.090908995008245008229153);
    __m256d const coeff_4_f64x4 = _mm256_set1_pd(+0.111111105648261418443745);
    __m256d const coeff_3_f64x4 = _mm256_set1_pd(-0.14285714266771329383765);
    __m256d const coeff_2_f64x4 = _mm256_set1_pd(+0.199999999996591265594148);
    __m256d const coeff_1_f64x4 = _mm256_set1_pd(-0.333333333333311110369124);
    __m256d const sign_mask_f64x4 = _mm256_set1_pd(-0.0);

    // Quadrant adjustments normalizing to absolute values of x and y
    __m256d xs_negative_mask_f64x4 = _mm256_cmp_pd(xs_inputs, _mm256_setzero_pd(), _CMP_LT_OS);
    __m256d xs_f64x4 = _mm256_andnot_pd(sign_mask_f64x4, xs_inputs); // abs(xs_inputs)
    __m256d ys_f64x4 = _mm256_andnot_pd(sign_mask_f64x4, ys_inputs); // abs(ys_inputs)

    // Ensure proper fraction where the numerator is smaller than the denominator
    __m256d swap_mask_f64x4 = _mm256_cmp_pd(ys_f64x4, xs_f64x4, _CMP_GT_OS);
    __m256d temps_f64x4 = xs_f64x4;
    xs_f64x4 = _mm256_blendv_pd(xs_f64x4, ys_f64x4, swap_mask_f64x4);
    __m256d neg_temps_f64x4 = _mm256_sub_pd(_mm256_setzero_pd(), temps_f64x4);
    ys_f64x4 = _mm256_blendv_pd(ys_f64x4, neg_temps_f64x4, swap_mask_f64x4);

    // Compute ratio_f64x4 and powers
    __m256d const ratio_f64x4 = _mm256_div_pd(ys_f64x4, xs_f64x4);
    __m256d const ratio_squared_f64x4 = _mm256_mul_pd(ratio_f64x4, ratio_f64x4);
    __m256d const ratio_cubed_f64x4 = _mm256_mul_pd(ratio_f64x4, ratio_squared_f64x4);

    // Polynomial evaluation using Horner's method
    __m256d polynomials_f64x4 = coeff_19_f64x4;
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_18_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_17_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_16_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_15_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_14_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_13_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_12_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_11_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_10_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_9_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_8_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_7_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_6_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_5_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_4_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_3_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_2_f64x4);
    polynomials_f64x4 = _mm256_fmadd_pd(polynomials_f64x4, ratio_squared_f64x4, coeff_1_f64x4);

    // Compute the result using masks for quadrant_f64x4 adjustments
    __m256d results_f64x4 = _mm256_fmadd_pd(ratio_cubed_f64x4, polynomials_f64x4, ratio_f64x4);

    // Compute quadrant_f64x4 value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    __m256d quadrant_f64x4 = _mm256_setzero_pd();
    __m256d neg_two_f64x4 = _mm256_set1_pd(-2.0);
    quadrant_f64x4 = _mm256_blendv_pd(quadrant_f64x4, neg_two_f64x4, xs_negative_mask_f64x4);
    __m256d one_f64x4 = _mm256_set1_pd(1.0);
    __m256d quadrant_incremented_f64x4 = _mm256_add_pd(quadrant_f64x4, one_f64x4);
    quadrant_f64x4 = _mm256_blendv_pd(quadrant_f64x4, quadrant_incremented_f64x4, swap_mask_f64x4);

    // Adjust for quadrant_f64x4: result += quadrant_f64x4 * π/2
    __m256d pi_half_f64x4 = _mm256_set1_pd(1.5707963267948966);
    results_f64x4 = _mm256_fmadd_pd(quadrant_f64x4, pi_half_f64x4, results_f64x4);

    // Transfer sign from x (XOR with sign bit of x_input)
    __m256d xs_sign_bits_f64x4 = _mm256_and_pd(xs_inputs, sign_mask_f64x4);
    results_f64x4 = _mm256_xor_pd(results_f64x4, xs_sign_bits_f64x4);

    // Transfer sign from y (XOR with sign bit of y_input)
    __m256d ys_sign_bits_f64x4 = _mm256_and_pd(ys_inputs, sign_mask_f64x4);
    results_f64x4 = _mm256_xor_pd(results_f64x4, ys_sign_bits_f64x4);

    return results_f64x4;
}

NK_PUBLIC void nk_each_sin_f32_haswell(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 angles_f32x8 = _mm256_loadu_ps(ins + i);
        __m256 results_f32x8 = nk_sin_f32x8_haswell_(angles_f32x8);
        _mm256_storeu_ps(outs + i, results_f32x8);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b256_vec_t angles_vec;
        nk_partial_load_b32x8_serial_(ins + i, &angles_vec, remaining);
        nk_b256_vec_t results_vec;
        results_vec.ymm_ps = nk_sin_f32x8_haswell_(angles_vec.ymm_ps);
        nk_partial_store_b32x8_serial_(&results_vec, outs + i, remaining);
    }
}

NK_PUBLIC void nk_each_cos_f32_haswell(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 angles_f32x8 = _mm256_loadu_ps(ins + i);
        __m256 results_f32x8 = nk_cos_f32x8_haswell_(angles_f32x8);
        _mm256_storeu_ps(outs + i, results_f32x8);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b256_vec_t angles_vec;
        nk_partial_load_b32x8_serial_(ins + i, &angles_vec, remaining);
        nk_b256_vec_t results_vec;
        results_vec.ymm_ps = nk_cos_f32x8_haswell_(angles_vec.ymm_ps);
        nk_partial_store_b32x8_serial_(&results_vec, outs + i, remaining);
    }
}

NK_PUBLIC void nk_each_atan_f32_haswell(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 values_f32x8 = _mm256_loadu_ps(ins + i);
        __m256 results_f32x8 = nk_atan_f32x8_haswell_(values_f32x8);
        _mm256_storeu_ps(outs + i, results_f32x8);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b256_vec_t values_vec;
        nk_partial_load_b32x8_serial_(ins + i, &values_vec, remaining);
        nk_b256_vec_t results_vec;
        results_vec.ymm_ps = nk_atan_f32x8_haswell_(values_vec.ymm_ps);
        nk_partial_store_b32x8_serial_(&results_vec, outs + i, remaining);
    }
}

NK_PUBLIC void nk_each_sin_f64_haswell(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d angles_f64x4 = _mm256_loadu_pd(ins + i);
        __m256d results_f64x4 = nk_sin_f64x4_haswell_(angles_f64x4);
        _mm256_storeu_pd(outs + i, results_f64x4);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b256_vec_t angles_vec;
        nk_partial_load_b64x4_haswell_(ins + i, &angles_vec, remaining);
        nk_b256_vec_t results_vec;
        results_vec.ymm_pd = nk_sin_f64x4_haswell_(angles_vec.ymm_pd);
        nk_partial_store_b64x4_haswell_(&results_vec, outs + i, remaining);
    }
}

NK_PUBLIC void nk_each_cos_f64_haswell(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d angles_f64x4 = _mm256_loadu_pd(ins + i);
        __m256d results_f64x4 = nk_cos_f64x4_haswell_(angles_f64x4);
        _mm256_storeu_pd(outs + i, results_f64x4);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b256_vec_t angles_vec;
        nk_partial_load_b64x4_haswell_(ins + i, &angles_vec, remaining);
        nk_b256_vec_t results_vec;
        results_vec.ymm_pd = nk_cos_f64x4_haswell_(angles_vec.ymm_pd);
        nk_partial_store_b64x4_haswell_(&results_vec, outs + i, remaining);
    }
}

NK_PUBLIC void nk_each_atan_f64_haswell(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d values_f64x4 = _mm256_loadu_pd(ins + i);
        __m256d results_f64x4 = nk_atan_f64x4_haswell_(values_f64x4);
        _mm256_storeu_pd(outs + i, results_f64x4);
    }
    if (i < n) {
        nk_size_t remaining = n - i;
        nk_b256_vec_t values_vec;
        nk_partial_load_b64x4_haswell_(ins + i, &values_vec, remaining);
        nk_b256_vec_t results_vec;
        results_vec.ymm_pd = nk_atan_f64x4_haswell_(values_vec.ymm_pd);
        nk_partial_store_b64x4_haswell_(&results_vec, outs + i, remaining);
    }
}

NK_PUBLIC void nk_each_rope_f32_haswell(nk_f32_t const *x, nk_f32_t *y, nk_rope_angle_t const *cos,
                                        nk_rope_angle_t const *sin, nk_size_t rows, nk_size_t heads, nk_size_t half_dim,
                                        nk_size_t x_row_stride, nk_size_t y_row_stride, nk_f32_t input_scale) {
    __m256 scale_f32x8 = _mm256_set1_ps(input_scale);
    for (nk_size_t r = 0; r != rows; ++r) {
        nk_f32_t const *cos_row = cos + r * half_dim;
        nk_f32_t const *sin_row = sin + r * half_dim;
        nk_f32_t const *x_row = (nk_f32_t const *)((unsigned char const *)x + r * x_row_stride);
        nk_f32_t *y_row = (nk_f32_t *)((unsigned char *)y + r * y_row_stride);
        for (nk_size_t h = 0; h != heads; ++h) {
            nk_f32_t const *x_base = x_row + h * 2 * half_dim;
            nk_f32_t *y_base = y_row + h * 2 * half_dim;
            nk_size_t i = 0;
            for (; i + 8 <= half_dim; i += 8) {
                __m256 low_f32x8 = _mm256_mul_ps(_mm256_loadu_ps(x_base + i), scale_f32x8);
                __m256 high_f32x8 = _mm256_mul_ps(_mm256_loadu_ps(x_base + i + half_dim), scale_f32x8);
                __m256 cos_f32x8 = _mm256_loadu_ps(cos_row + i), sin_f32x8 = _mm256_loadu_ps(sin_row + i);
                _mm256_storeu_ps(y_base + i,
                                 _mm256_fmsub_ps(low_f32x8, cos_f32x8, _mm256_mul_ps(high_f32x8, sin_f32x8)));
                _mm256_storeu_ps(y_base + i + half_dim,
                                 _mm256_fmadd_ps(low_f32x8, sin_f32x8, _mm256_mul_ps(high_f32x8, cos_f32x8)));
            }
            for (; i != half_dim; ++i) {
                nk_f32_t low = x_base[i] * input_scale, high = x_base[i + half_dim] * input_scale;
                nk_f32_t cosine = cos_row[i], sine = sin_row[i];
                y_base[i] = low * cosine - high * sine;
                y_base[i + half_dim] = low * sine + high * cosine;
            }
        }
    }
}

NK_PUBLIC void nk_each_rope_bf16_haswell(nk_bf16_t const *x, nk_bf16_t *y, nk_rope_angle_t const *cos,
                                         nk_rope_angle_t const *sin, nk_size_t rows, nk_size_t heads,
                                         nk_size_t half_dim, nk_size_t x_row_stride, nk_size_t y_row_stride,
                                         nk_f32_t input_scale) {
    __m256 scale_f32x8 = _mm256_set1_ps(input_scale);
    for (nk_size_t r = 0; r != rows; ++r) {
        nk_f32_t const *cos_row = cos + r * half_dim;
        nk_f32_t const *sin_row = sin + r * half_dim;
        nk_bf16_t const *x_row = (nk_bf16_t const *)((unsigned char const *)x + r * x_row_stride);
        nk_bf16_t *y_row = (nk_bf16_t *)((unsigned char *)y + r * y_row_stride);
        for (nk_size_t h = 0; h != heads; ++h) {
            nk_bf16_t const *x_base = x_row + h * 2 * half_dim;
            nk_bf16_t *y_base = y_row + h * 2 * half_dim;
            nk_size_t i = 0;
            for (; i + 8 <= half_dim; i += 8) {
                nk_b256_vec_t low_vec, high_vec;
                nk_load_bf16x8_to_f32x8_haswell_(x_base + i, &low_vec);
                nk_load_bf16x8_to_f32x8_haswell_(x_base + i + half_dim, &high_vec);
                __m256 low_f32x8 = _mm256_mul_ps(low_vec.ymm_ps, scale_f32x8),
                       high_f32x8 = _mm256_mul_ps(high_vec.ymm_ps, scale_f32x8);
                __m256 cos_f32x8 = _mm256_loadu_ps(cos_row + i), sin_f32x8 = _mm256_loadu_ps(sin_row + i);
                _mm_storeu_si128((__m128i *)(y_base + i),
                                 nk_f32x8_to_bf16x8_haswell_(
                                     _mm256_fmsub_ps(low_f32x8, cos_f32x8, _mm256_mul_ps(high_f32x8, sin_f32x8))));
                _mm_storeu_si128((__m128i *)(y_base + i + half_dim),
                                 nk_f32x8_to_bf16x8_haswell_(
                                     _mm256_fmadd_ps(low_f32x8, sin_f32x8, _mm256_mul_ps(high_f32x8, cos_f32x8))));
            }
            for (; i != half_dim; ++i) {
                nk_f32_t low, high;
                nk_bf16_to_f32_serial(x_base + i, &low);
                nk_bf16_to_f32_serial(x_base + i + half_dim, &high);
                low *= input_scale, high *= input_scale;
                nk_f32_t cosine = cos_row[i], sine = sin_row[i], rotated_low = low * cosine - high * sine,
                         rotated_high = low * sine + high * cosine;
                nk_f32_to_bf16_serial(&rotated_low, y_base + i);
                nk_f32_to_bf16_serial(&rotated_high, y_base + i + half_dim);
            }
        }
    }
}

NK_PUBLIC void nk_each_rope_e4m3_haswell(nk_e4m3_t const *x, nk_e4m3_t *y, nk_rope_angle_t const *cos,
                                         nk_rope_angle_t const *sin, nk_size_t rows, nk_size_t heads,
                                         nk_size_t half_dim, nk_size_t x_row_stride, nk_size_t y_row_stride,
                                         nk_f32_t input_scale) {
    __m256 scale_f32x8 = _mm256_set1_ps(input_scale);
    for (nk_size_t r = 0; r != rows; ++r) {
        nk_f32_t const *cos_row = cos + r * half_dim;
        nk_f32_t const *sin_row = sin + r * half_dim;
        nk_e4m3_t const *x_row = (nk_e4m3_t const *)((unsigned char const *)x + r * x_row_stride);
        nk_e4m3_t *y_row = (nk_e4m3_t *)((unsigned char *)y + r * y_row_stride);
        for (nk_size_t h = 0; h != heads; ++h) {
            nk_e4m3_t const *x_base = x_row + h * 2 * half_dim;
            nk_e4m3_t *y_base = y_row + h * 2 * half_dim;
            nk_size_t i = 0;
            for (; i + 8 <= half_dim; i += 8) {
                nk_b256_vec_t low_vec, high_vec;
                nk_load_e4m3x8_to_f32x8_haswell_(x_base + i, &low_vec);
                nk_load_e4m3x8_to_f32x8_haswell_(x_base + i + half_dim, &high_vec);
                __m256 low_f32x8 = _mm256_mul_ps(low_vec.ymm_ps, scale_f32x8),
                       high_f32x8 = _mm256_mul_ps(high_vec.ymm_ps, scale_f32x8);
                __m256 cos_f32x8 = _mm256_loadu_ps(cos_row + i), sin_f32x8 = _mm256_loadu_ps(sin_row + i);
                _mm_storel_epi64((__m128i *)(y_base + i),
                                 nk_f32x8_to_e4m3x8_haswell_(
                                     _mm256_fmsub_ps(low_f32x8, cos_f32x8, _mm256_mul_ps(high_f32x8, sin_f32x8))));
                _mm_storel_epi64((__m128i *)(y_base + i + half_dim),
                                 nk_f32x8_to_e4m3x8_haswell_(
                                     _mm256_fmadd_ps(low_f32x8, sin_f32x8, _mm256_mul_ps(high_f32x8, cos_f32x8))));
            }
            for (; i != half_dim; ++i) {
                nk_f32_t low, high;
                nk_e4m3_to_f32_serial(x_base + i, &low);
                nk_e4m3_to_f32_serial(x_base + i + half_dim, &high);
                low *= input_scale, high *= input_scale;
                nk_f32_t cosine = cos_row[i], sine = sin_row[i], rotated_low = low * cosine - high * sine,
                         rotated_high = low * sine + high * cosine;
                nk_f32_to_e4m3_serial(&rotated_low, y_base + i);
                nk_f32_to_e4m3_serial(&rotated_high, y_base + i + half_dim);
            }
        }
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

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X8664_
#endif // NK_TRIGONOMETRY_HASWELL_H
