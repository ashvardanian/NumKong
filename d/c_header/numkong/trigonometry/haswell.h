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
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      _mm256_fmadd_ps/pd          VFMADD (YMM, YMM, YMM)          5cy         0.5/cy      p01
 *      _mm256_mul_ps/pd            VMULPS/PD (YMM, YMM, YMM)       5cy         0.5/cy      p01
 *      _mm256_blendv_ps/pd         VBLENDVPS/PD (YMM, YMM, YMM)    2cy         1/cy        p015
 *      _mm256_round_ps/pd          VROUNDPS/PD (YMM, YMM, I8)      6cy         1/cy        p01
 *      _mm256_div_ps               VDIVPS (YMM, YMM, YMM)          13cy        5/cy        p0
 *
 *  Polynomial evaluation uses Horner's method with FMA for sin/cos/atan approximation. For large
 *  arrays, out-of-order execution across loop iterations hides FMA latency better than Estrin's
 *  scheme. Range reduction uses argument folding modulo pi with high/low precision constants.
 */
#ifndef NK_TRIGONOMETRY_HASWELL_H
#define NK_TRIGONOMETRY_HASWELL_H

#if NK_TARGET_X86_
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
    __m256 const pi_hi_f32x8 = _mm256_set1_ps(3.1415927f);
    __m256 const pi_lo_f32x8 = _mm256_set1_ps(-8.742278e-8f);
    __m256 const pi_reciprocal = _mm256_set1_ps(0.31830988618379067154f); // 1/π
    // Degree-9 minimax coefficients
    __m256 const coeff_9 = _mm256_set1_ps(+2.7557319224e-6f);
    __m256 const coeff_7 = _mm256_set1_ps(-1.9841269841e-4f);
    __m256 const coeff_5 = _mm256_set1_ps(+8.3333293855e-3f);
    __m256 const coeff_3 = _mm256_set1_ps(-1.6666666641e-1f);

    // Compute (multiples_of_pi) = round(angle / π)
    __m256 quotients = _mm256_mul_ps(angles_radians, pi_reciprocal);
    __m256 rounded_quotients = _mm256_round_ps(quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // Use truncation (MXCSR-independent) since rounded_quotients is already integer-valued
    __m256i multiples_of_pi = _mm256_cvttps_epi32(rounded_quotients);

    // Cody-Waite range reduction
    __m256 angles = _mm256_fnmadd_ps(rounded_quotients, pi_hi_f32x8, angles_radians);
    angles = _mm256_fnmadd_ps(rounded_quotients, pi_lo_f32x8, angles);
    __m256 const angles_squared = _mm256_mul_ps(angles, angles);
    __m256 const angles_cubed = _mm256_mul_ps(angles, angles_squared);

    // Degree-9 polynomial via Horner's method
    __m256 polynomials = coeff_9;
    polynomials = _mm256_fmadd_ps(polynomials, angles_squared, coeff_7);
    polynomials = _mm256_fmadd_ps(polynomials, angles_squared, coeff_5);
    polynomials = _mm256_fmadd_ps(polynomials, angles_squared, coeff_3);
    __m256 results = _mm256_fmadd_ps(angles_cubed, polynomials, angles);

    // If multiples_of_pi is odd, flip the sign of the results
    __m256i parity = _mm256_and_si256(multiples_of_pi, _mm256_set1_epi32(1));
    __m256i odd_mask = _mm256_cmpeq_epi32(parity, _mm256_set1_epi32(1));
    __m256 float_mask = _mm256_castsi256_ps(odd_mask);
    __m256 negated = _mm256_sub_ps(_mm256_setzero_ps(), results);
    results = _mm256_blendv_ps(results, negated, float_mask);
    return results;
}

NK_INTERNAL __m256 nk_cos_f32x8_haswell_(__m256 const angles_radians) {
    // Cody-Waite constants for argument reduction
    __m256 const pi_hi_f32x8 = _mm256_set1_ps(3.1415927f);
    __m256 const pi_lo_f32x8 = _mm256_set1_ps(-8.742278e-8f);
    __m256 const pi_half = _mm256_set1_ps(1.57079632679489661923f);       // π/2
    __m256 const pi_reciprocal = _mm256_set1_ps(0.31830988618379067154f); // 1/π
    // Degree-9 minimax coefficients
    __m256 const coeff_9 = _mm256_set1_ps(+2.7557319224e-6f);
    __m256 const coeff_7 = _mm256_set1_ps(-1.9841269841e-4f);
    __m256 const coeff_5 = _mm256_set1_ps(+8.3333293855e-3f);
    __m256 const coeff_3 = _mm256_set1_ps(-1.6666666641e-1f);

    // Compute (multiples_of_pi) = round((angle / π) - 0.5)
    __m256 quotients = _mm256_fmsub_ps(angles_radians, pi_reciprocal, _mm256_set1_ps(0.5f));
    __m256 rounded_quotients = _mm256_round_ps(quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // Use truncation (MXCSR-independent) since rounded_quotients is already integer-valued
    __m256i multiples_of_pi = _mm256_cvttps_epi32(rounded_quotients);

    // Cody-Waite range reduction: angle = angle_radians - (multiples * pi + pi/2)
    __m256 const offset = _mm256_fmadd_ps(rounded_quotients, pi_hi_f32x8, pi_half);
    __m256 angles = _mm256_sub_ps(angles_radians, offset);
    angles = _mm256_fnmadd_ps(rounded_quotients, pi_lo_f32x8, angles);
    __m256 const angles_squared = _mm256_mul_ps(angles, angles);
    __m256 const angles_cubed = _mm256_mul_ps(angles, angles_squared);

    // Degree-9 polynomial via Horner's method
    __m256 polynomials = coeff_9;
    polynomials = _mm256_fmadd_ps(polynomials, angles_squared, coeff_7);
    polynomials = _mm256_fmadd_ps(polynomials, angles_squared, coeff_5);
    polynomials = _mm256_fmadd_ps(polynomials, angles_squared, coeff_3);
    __m256 results = _mm256_fmadd_ps(angles_cubed, polynomials, angles);

    // If multiples_of_pi is even, flip the sign of the results
    __m256i parity = _mm256_and_si256(multiples_of_pi, _mm256_set1_epi32(1));
    __m256i even_mask = _mm256_cmpeq_epi32(parity, _mm256_setzero_si256());
    __m256 float_mask = _mm256_castsi256_ps(even_mask);
    __m256 negated = _mm256_sub_ps(_mm256_setzero_ps(), results);
    results = _mm256_blendv_ps(results, negated, float_mask);
    return results;
}

NK_INTERNAL __m256 nk_atan_f32x8_haswell_(__m256 const inputs) {
    // Polynomial coefficients for atan approximation (8 terms)
    // These coefficients approximate: atan(x) ≈ x + c8 × x³ + c7 × x⁵ + c6 × x⁷ + ... + c1 × x¹⁵
    __m256 const coeff_8 = _mm256_set1_ps(-0.333331018686294555664062f);
    __m256 const coeff_7 = _mm256_set1_ps(+0.199926957488059997558594f);
    __m256 const coeff_6 = _mm256_set1_ps(-0.142027363181114196777344f);
    __m256 const coeff_5 = _mm256_set1_ps(+0.106347933411598205566406f);
    __m256 const coeff_4 = _mm256_set1_ps(-0.0748900920152664184570312f);
    __m256 const coeff_3 = _mm256_set1_ps(+0.0425049886107444763183594f);
    __m256 const coeff_2 = _mm256_set1_ps(-0.0159569028764963150024414f);
    __m256 const coeff_1 = _mm256_set1_ps(+0.00282363896258175373077393f);
    __m256 const sign_mask = _mm256_set1_ps(-0.0f);

    // Adjust for quadrant - detect negative values
    __m256 values = inputs;
    __m256 negative_mask = _mm256_cmp_ps(values, _mm256_setzero_ps(), _CMP_LT_OS);
    values = _mm256_andnot_ps(sign_mask, values); // abs(values)

    // Check if values > 1 (need reciprocal)
    __m256 reciprocal_mask = _mm256_cmp_ps(values, _mm256_set1_ps(1.0f), _CMP_GT_OS);
    __m256 reciprocal_values = _mm256_div_ps(_mm256_set1_ps(1.0f), values);
    values = _mm256_blendv_ps(values, reciprocal_values, reciprocal_mask);

    // Argument reduction
    __m256 const values_squared = _mm256_mul_ps(values, values);
    __m256 const values_cubed = _mm256_mul_ps(values, values_squared);

    // Polynomial evaluation using Horner's method.
    // For large arrays, out-of-order execution across loop iterations already hides
    // FMA latency. Estrin's scheme was tested but showed ~20% regression because
    // the extra power computations (y², y⁴) hurt throughput more than the reduced
    // dependency depth helps latency.
    __m256 polynomials = coeff_1;
    polynomials = _mm256_fmadd_ps(polynomials, values_squared, coeff_2);
    polynomials = _mm256_fmadd_ps(polynomials, values_squared, coeff_3);
    polynomials = _mm256_fmadd_ps(polynomials, values_squared, coeff_4);
    polynomials = _mm256_fmadd_ps(polynomials, values_squared, coeff_5);
    polynomials = _mm256_fmadd_ps(polynomials, values_squared, coeff_6);
    polynomials = _mm256_fmadd_ps(polynomials, values_squared, coeff_7);
    polynomials = _mm256_fmadd_ps(polynomials, values_squared, coeff_8);

    // Compute result: atan(x) ≈ x + x³ * P(x²)
    __m256 result = _mm256_fmadd_ps(values_cubed, polynomials, values);

    // Adjust for reciprocal: result = π/2 - result
    __m256 adjusted = _mm256_sub_ps(_mm256_set1_ps(1.5707963267948966f), result);
    result = _mm256_blendv_ps(result, adjusted, reciprocal_mask);

    // Adjust for negative: result = -result
    __m256 negated = _mm256_sub_ps(_mm256_setzero_ps(), result);
    result = _mm256_blendv_ps(result, negated, negative_mask);
    return result;
}

NK_INTERNAL __m256 nk_atan2_f32x8_haswell_(__m256 const ys_inputs, __m256 const xs_inputs) {
    // Polynomial coefficients (same as atan)
    __m256 const coeff_8 = _mm256_set1_ps(-0.333331018686294555664062f);
    __m256 const coeff_7 = _mm256_set1_ps(+0.199926957488059997558594f);
    __m256 const coeff_6 = _mm256_set1_ps(-0.142027363181114196777344f);
    __m256 const coeff_5 = _mm256_set1_ps(+0.106347933411598205566406f);
    __m256 const coeff_4 = _mm256_set1_ps(-0.0748900920152664184570312f);
    __m256 const coeff_3 = _mm256_set1_ps(+0.0425049886107444763183594f);
    __m256 const coeff_2 = _mm256_set1_ps(-0.0159569028764963150024414f);
    __m256 const coeff_1 = _mm256_set1_ps(+0.00282363896258175373077393f);
    __m256 const sign_mask = _mm256_set1_ps(-0.0f);

    // Quadrant adjustments normalizing to absolute values of x and y
    __m256 xs_negative_mask = _mm256_cmp_ps(xs_inputs, _mm256_setzero_ps(), _CMP_LT_OS);
    __m256 xs = _mm256_andnot_ps(sign_mask, xs_inputs); // abs(xs_inputs)
    __m256 ys = _mm256_andnot_ps(sign_mask, ys_inputs); // abs(ys_inputs)

    // Ensure proper fraction where the numerator is smaller than the denominator
    __m256 swap_mask = _mm256_cmp_ps(ys, xs, _CMP_GT_OS);
    __m256 temps = xs;
    xs = _mm256_blendv_ps(xs, ys, swap_mask);
    __m256 neg_temps = _mm256_sub_ps(_mm256_setzero_ps(), temps);
    ys = _mm256_blendv_ps(ys, neg_temps, swap_mask);

    // Compute ratio and powers
    __m256 const ratio = _mm256_div_ps(ys, xs);
    __m256 const ratio_squared = _mm256_mul_ps(ratio, ratio);
    __m256 const ratio_cubed = _mm256_mul_ps(ratio, ratio_squared);

    // Polynomial evaluation using Horner's method
    __m256 polynomials = coeff_1;
    polynomials = _mm256_fmadd_ps(polynomials, ratio_squared, coeff_2);
    polynomials = _mm256_fmadd_ps(polynomials, ratio_squared, coeff_3);
    polynomials = _mm256_fmadd_ps(polynomials, ratio_squared, coeff_4);
    polynomials = _mm256_fmadd_ps(polynomials, ratio_squared, coeff_5);
    polynomials = _mm256_fmadd_ps(polynomials, ratio_squared, coeff_6);
    polynomials = _mm256_fmadd_ps(polynomials, ratio_squared, coeff_7);
    polynomials = _mm256_fmadd_ps(polynomials, ratio_squared, coeff_8);

    // Compute the result using masks for quadrant adjustments
    __m256 results = _mm256_fmadd_ps(ratio_cubed, polynomials, ratio);

    // Compute quadrant value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    __m256 quadrant = _mm256_setzero_ps();
    __m256 neg_two = _mm256_set1_ps(-2.0f);
    quadrant = _mm256_blendv_ps(quadrant, neg_two, xs_negative_mask);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 quadrant_incremented = _mm256_add_ps(quadrant, one);
    quadrant = _mm256_blendv_ps(quadrant, quadrant_incremented, swap_mask);

    // Adjust for quadrant: result += quadrant * π/2
    __m256 pi_half = _mm256_set1_ps(1.5707963267948966f);
    results = _mm256_fmadd_ps(quadrant, pi_half, results);

    // Transfer sign from x (XOR with sign bit of x_input)
    __m256 xs_sign_bits = _mm256_and_ps(xs_inputs, sign_mask);
    results = _mm256_xor_ps(results, xs_sign_bits);

    // Transfer sign from y (XOR with sign bit of y_input)
    __m256 ys_sign_bits = _mm256_and_ps(ys_inputs, sign_mask);
    results = _mm256_xor_ps(results, ys_sign_bits);

    return results;
}

NK_INTERNAL __m256d nk_sin_f64x4_haswell_(__m256d const angles_radians) {
    // Constants for argument reduction
    __m256d const pi_high = _mm256_set1_pd(3.141592653589793116);         // High-digits part of π
    __m256d const pi_low = _mm256_set1_pd(1.2246467991473532072e-16);     // Low-digits part of π
    __m256d const pi_reciprocal = _mm256_set1_pd(0.31830988618379067154); // 1/π

    // Polynomial coefficients for sine approximation (minimax polynomial)
    __m256d const coeff_0 = _mm256_set1_pd(+0.00833333333333332974823815);
    __m256d const coeff_1 = _mm256_set1_pd(-0.000198412698412696162806809);
    __m256d const coeff_2 = _mm256_set1_pd(+2.75573192239198747630416e-06);
    __m256d const coeff_3 = _mm256_set1_pd(-2.50521083763502045810755e-08);
    __m256d const coeff_4 = _mm256_set1_pd(+1.60590430605664501629054e-10);
    __m256d const coeff_5 = _mm256_set1_pd(-7.64712219118158833288484e-13);
    __m256d const coeff_6 = _mm256_set1_pd(+2.81009972710863200091251e-15);
    __m256d const coeff_7 = _mm256_set1_pd(-7.97255955009037868891952e-18);
    __m256d const coeff_8 = _mm256_set1_pd(-0.166666666666666657414808);

    // Compute (rounded_quotients) = round(angle / π)
    __m256d const quotients = _mm256_mul_pd(angles_radians, pi_reciprocal);
    __m256d const rounded_quotients = _mm256_round_pd(quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Reduce the angle: angle - (rounded_quotients * π_high + rounded_quotients * π_low)
    __m256d angles = angles_radians;
    angles = _mm256_fnmadd_pd(rounded_quotients, pi_high, angles);
    angles = _mm256_fnmadd_pd(rounded_quotients, pi_low, angles);

    // If rounded_quotients is odd (bit 0 set), negate the angle
    // Convert to 32-bit int (returns __m128i with 4 x 32-bit ints)
    // Use truncation (MXCSR-independent) since rounded_quotients is already integer-valued
    __m128i quotients_i32 = _mm256_cvttpd_epi32(rounded_quotients);
    __m128i parity = _mm_and_si128(quotients_i32, _mm_set1_epi32(1));
    __m128i odd_mask_i32 = _mm_cmpeq_epi32(parity, _mm_set1_epi32(1));
    // Expand 32-bit mask to 64-bit by shuffling
    __m256i odd_mask_i64 = _mm256_cvtepi32_epi64(odd_mask_i32);
    __m256d float_mask = _mm256_castsi256_pd(odd_mask_i64);
    __m256d negated_angles = _mm256_sub_pd(_mm256_setzero_pd(), angles);
    angles = _mm256_blendv_pd(angles, negated_angles, float_mask);

    __m256d const angles_squared = _mm256_mul_pd(angles, angles);
    __m256d const angles_cubed = _mm256_mul_pd(angles, angles_squared);
    __m256d const angles_quadratic = _mm256_mul_pd(angles_squared, angles_squared);
    __m256d const angles_octic = _mm256_mul_pd(angles_quadratic, angles_quadratic);

    // Compute higher-degree polynomial terms
    __m256d const poly_67 = _mm256_fmadd_pd(angles_squared, coeff_7, coeff_6);
    __m256d const poly_45 = _mm256_fmadd_pd(angles_squared, coeff_5, coeff_4);
    __m256d const poly_4567 = _mm256_fmadd_pd(angles_quadratic, poly_67, poly_45);

    // Compute lower-degree polynomial terms
    __m256d const poly_23 = _mm256_fmadd_pd(angles_squared, coeff_3, coeff_2);
    __m256d const poly_01 = _mm256_fmadd_pd(angles_squared, coeff_1, coeff_0);
    __m256d const poly_0123 = _mm256_fmadd_pd(angles_quadratic, poly_23, poly_01);

    // Combine polynomial terms
    __m256d results = _mm256_fmadd_pd(angles_octic, poly_4567, poly_0123);
    results = _mm256_fmadd_pd(results, angles_squared, coeff_8);
    results = _mm256_fmadd_pd(results, angles_cubed, angles);

    // Handle the special case of negative zero input
    __m256d const non_zero_mask = _mm256_cmp_pd(angles_radians, _mm256_setzero_pd(), _CMP_NEQ_UQ);
    results = _mm256_and_pd(results, non_zero_mask);
    return results;
}

NK_INTERNAL __m256d nk_cos_f64x4_haswell_(__m256d const angles_radians) {
    // Constants for argument reduction
    __m256d const pi_high_half = _mm256_set1_pd(3.141592653589793116 * 0.5);     // High-digits part of π/2
    __m256d const pi_low_half = _mm256_set1_pd(1.2246467991473532072e-16 * 0.5); // Low-digits part of π/2
    __m256d const pi_reciprocal = _mm256_set1_pd(0.31830988618379067154);        // 1/π

    // Polynomial coefficients for cosine approximation
    __m256d const coeff_0 = _mm256_set1_pd(+0.00833333333333332974823815);
    __m256d const coeff_1 = _mm256_set1_pd(-0.000198412698412696162806809);
    __m256d const coeff_2 = _mm256_set1_pd(+2.75573192239198747630416e-06);
    __m256d const coeff_3 = _mm256_set1_pd(-2.50521083763502045810755e-08);
    __m256d const coeff_4 = _mm256_set1_pd(+1.60590430605664501629054e-10);
    __m256d const coeff_5 = _mm256_set1_pd(-7.64712219118158833288484e-13);
    __m256d const coeff_6 = _mm256_set1_pd(+2.81009972710863200091251e-15);
    __m256d const coeff_7 = _mm256_set1_pd(-7.97255955009037868891952e-18);
    __m256d const coeff_8 = _mm256_set1_pd(-0.166666666666666657414808);

    // Compute (rounded_quotients) = 2 * round(angle / π - 0.5) + 1
    // Use fmsub: a*b - c = angles * (1/π) - 0.5
    __m256d const quotients = _mm256_fmsub_pd(angles_radians, pi_reciprocal, _mm256_set1_pd(0.5));
    __m256d const rounded_quotients = _mm256_fmadd_pd(                             //
        _mm256_set1_pd(2.0),                                                       //
        _mm256_round_pd(quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), //
        _mm256_set1_pd(1.0));

    // Reduce the angle: angle - (rounded_quotients * π_high_half + rounded_quotients * π_low_half)
    __m256d angles = angles_radians;
    angles = _mm256_fnmadd_pd(rounded_quotients, pi_high_half, angles);
    angles = _mm256_fnmadd_pd(rounded_quotients, pi_low_half, angles);

    // If (rounded_quotients & 2) == 0, negate the angle
    // Use truncation (MXCSR-independent) since rounded_quotients is already integer-valued
    __m128i quotients_i32 = _mm256_cvttpd_epi32(rounded_quotients);
    __m128i bit2 = _mm_and_si128(quotients_i32, _mm_set1_epi32(2));
    __m128i flip_mask_i32 = _mm_cmpeq_epi32(bit2, _mm_setzero_si128());
    __m256i flip_mask_i64 = _mm256_cvtepi32_epi64(flip_mask_i32);
    __m256d float_mask = _mm256_castsi256_pd(flip_mask_i64);
    __m256d negated_angles = _mm256_sub_pd(_mm256_setzero_pd(), angles);
    angles = _mm256_blendv_pd(angles, negated_angles, float_mask);

    __m256d const angles_squared = _mm256_mul_pd(angles, angles);
    __m256d const angles_cubed = _mm256_mul_pd(angles, angles_squared);
    __m256d const angles_quadratic = _mm256_mul_pd(angles_squared, angles_squared);
    __m256d const angles_octic = _mm256_mul_pd(angles_quadratic, angles_quadratic);

    // Compute higher-degree polynomial terms
    __m256d const poly_67 = _mm256_fmadd_pd(angles_squared, coeff_7, coeff_6);
    __m256d const poly_45 = _mm256_fmadd_pd(angles_squared, coeff_5, coeff_4);
    __m256d const poly_4567 = _mm256_fmadd_pd(angles_quadratic, poly_67, poly_45);

    // Compute lower-degree polynomial terms
    __m256d const poly_23 = _mm256_fmadd_pd(angles_squared, coeff_3, coeff_2);
    __m256d const poly_01 = _mm256_fmadd_pd(angles_squared, coeff_1, coeff_0);
    __m256d const poly_0123 = _mm256_fmadd_pd(angles_quadratic, poly_23, poly_01);

    // Combine polynomial terms
    __m256d results = _mm256_fmadd_pd(angles_octic, poly_4567, poly_0123);
    results = _mm256_fmadd_pd(results, angles_squared, coeff_8);
    results = _mm256_fmadd_pd(results, angles_cubed, angles);
    return results;
}

NK_INTERNAL __m256d nk_atan_f64x4_haswell_(__m256d const inputs) {
    // Polynomial coefficients for atan approximation (19 coefficients)
    // The polynomial approximates: atan(x) ≈ x + x³ * P(x²) where P has 19 terms
    __m256d const coeff_19 = _mm256_set1_pd(-1.88796008463073496563746e-05);
    __m256d const coeff_18 = _mm256_set1_pd(+0.000209850076645816976906797);
    __m256d const coeff_17 = _mm256_set1_pd(-0.00110611831486672482563471);
    __m256d const coeff_16 = _mm256_set1_pd(+0.00370026744188713119232403);
    __m256d const coeff_15 = _mm256_set1_pd(-0.00889896195887655491740809);
    __m256d const coeff_14 = _mm256_set1_pd(+0.016599329773529201970117);
    __m256d const coeff_13 = _mm256_set1_pd(-0.0254517624932312641616861);
    __m256d const coeff_12 = _mm256_set1_pd(+0.0337852580001353069993897);
    __m256d const coeff_11 = _mm256_set1_pd(-0.0407629191276836500001934);
    __m256d const coeff_10 = _mm256_set1_pd(+0.0466667150077840625632675);
    __m256d const coeff_9 = _mm256_set1_pd(-0.0523674852303482457616113);
    __m256d const coeff_8 = _mm256_set1_pd(+0.0587666392926673580854313);
    __m256d const coeff_7 = _mm256_set1_pd(-0.0666573579361080525984562);
    __m256d const coeff_6 = _mm256_set1_pd(+0.0769219538311769618355029);
    __m256d const coeff_5 = _mm256_set1_pd(-0.090908995008245008229153);
    __m256d const coeff_4 = _mm256_set1_pd(+0.111111105648261418443745);
    __m256d const coeff_3 = _mm256_set1_pd(-0.14285714266771329383765);
    __m256d const coeff_2 = _mm256_set1_pd(+0.199999999996591265594148);
    __m256d const coeff_1 = _mm256_set1_pd(-0.333333333333311110369124);
    __m256d const sign_mask = _mm256_set1_pd(-0.0);

    // Adjust for quadrant - detect negative values
    __m256d values = inputs;
    __m256d negative_mask = _mm256_cmp_pd(values, _mm256_setzero_pd(), _CMP_LT_OS);
    values = _mm256_andnot_pd(sign_mask, values); // abs(values)

    // Check if values > 1 (need reciprocal)
    // Note: For f64, we keep VDIVPD since RCPPD doesn't exist and Newton-Raphson
    // would need 2 iterations for sufficient precision (~44 bits needed for f64)
    __m256d reciprocal_mask = _mm256_cmp_pd(values, _mm256_set1_pd(1.0), _CMP_GT_OS);
    __m256d reciprocal_values = _mm256_div_pd(_mm256_set1_pd(1.0), values);
    values = _mm256_blendv_pd(values, reciprocal_values, reciprocal_mask);

    // Argument reduction
    __m256d const values_squared = _mm256_mul_pd(values, values);
    __m256d const values_cubed = _mm256_mul_pd(values, values_squared);

    // Polynomial evaluation using Horner's method.
    // For large arrays, out-of-order execution across loop iterations already hides
    // FMA latency. Estrin's scheme was tested but showed minimal improvement (~1%)
    // while adding complexity. Keeping Horner for maintainability.
    __m256d polynomials = coeff_19;
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_18);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_17);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_16);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_15);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_14);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_13);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_12);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_11);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_10);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_9);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_8);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_7);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_6);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_5);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_4);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_3);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_2);
    polynomials = _mm256_fmadd_pd(polynomials, values_squared, coeff_1);

    // Compute result
    __m256d result = _mm256_fmadd_pd(values_cubed, polynomials, values);

    // Adjust for reciprocal: result = π/2 - result
    __m256d adjusted = _mm256_sub_pd(_mm256_set1_pd(1.5707963267948966), result);
    result = _mm256_blendv_pd(result, adjusted, reciprocal_mask);

    // Adjust for negative: result = -result
    __m256d negated = _mm256_sub_pd(_mm256_setzero_pd(), result);
    result = _mm256_blendv_pd(result, negated, negative_mask);
    return result;
}

NK_INTERNAL __m256d nk_atan2_f64x4_haswell_(__m256d const ys_inputs, __m256d const xs_inputs) {
    // Polynomial coefficients for atan approximation (19 coefficients, same as atan)
    __m256d const coeff_19 = _mm256_set1_pd(-1.88796008463073496563746e-05);
    __m256d const coeff_18 = _mm256_set1_pd(+0.000209850076645816976906797);
    __m256d const coeff_17 = _mm256_set1_pd(-0.00110611831486672482563471);
    __m256d const coeff_16 = _mm256_set1_pd(+0.00370026744188713119232403);
    __m256d const coeff_15 = _mm256_set1_pd(-0.00889896195887655491740809);
    __m256d const coeff_14 = _mm256_set1_pd(+0.016599329773529201970117);
    __m256d const coeff_13 = _mm256_set1_pd(-0.0254517624932312641616861);
    __m256d const coeff_12 = _mm256_set1_pd(+0.0337852580001353069993897);
    __m256d const coeff_11 = _mm256_set1_pd(-0.0407629191276836500001934);
    __m256d const coeff_10 = _mm256_set1_pd(+0.0466667150077840625632675);
    __m256d const coeff_9 = _mm256_set1_pd(-0.0523674852303482457616113);
    __m256d const coeff_8 = _mm256_set1_pd(+0.0587666392926673580854313);
    __m256d const coeff_7 = _mm256_set1_pd(-0.0666573579361080525984562);
    __m256d const coeff_6 = _mm256_set1_pd(+0.0769219538311769618355029);
    __m256d const coeff_5 = _mm256_set1_pd(-0.090908995008245008229153);
    __m256d const coeff_4 = _mm256_set1_pd(+0.111111105648261418443745);
    __m256d const coeff_3 = _mm256_set1_pd(-0.14285714266771329383765);
    __m256d const coeff_2 = _mm256_set1_pd(+0.199999999996591265594148);
    __m256d const coeff_1 = _mm256_set1_pd(-0.333333333333311110369124);
    __m256d const sign_mask = _mm256_set1_pd(-0.0);

    // Quadrant adjustments normalizing to absolute values of x and y
    __m256d xs_negative_mask = _mm256_cmp_pd(xs_inputs, _mm256_setzero_pd(), _CMP_LT_OS);
    __m256d xs = _mm256_andnot_pd(sign_mask, xs_inputs); // abs(xs_inputs)
    __m256d ys = _mm256_andnot_pd(sign_mask, ys_inputs); // abs(ys_inputs)

    // Ensure proper fraction where the numerator is smaller than the denominator
    __m256d swap_mask = _mm256_cmp_pd(ys, xs, _CMP_GT_OS);
    __m256d temps = xs;
    xs = _mm256_blendv_pd(xs, ys, swap_mask);
    __m256d neg_temps = _mm256_sub_pd(_mm256_setzero_pd(), temps);
    ys = _mm256_blendv_pd(ys, neg_temps, swap_mask);

    // Compute ratio and powers
    __m256d const ratio = _mm256_div_pd(ys, xs);
    __m256d const ratio_squared = _mm256_mul_pd(ratio, ratio);
    __m256d const ratio_cubed = _mm256_mul_pd(ratio, ratio_squared);

    // Polynomial evaluation using Horner's method
    __m256d polynomials = coeff_19;
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_18);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_17);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_16);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_15);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_14);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_13);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_12);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_11);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_10);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_9);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_8);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_7);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_6);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_5);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_4);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_3);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_2);
    polynomials = _mm256_fmadd_pd(polynomials, ratio_squared, coeff_1);

    // Compute the result using masks for quadrant adjustments
    __m256d results = _mm256_fmadd_pd(ratio_cubed, polynomials, ratio);

    // Compute quadrant value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    __m256d quadrant = _mm256_setzero_pd();
    __m256d neg_two = _mm256_set1_pd(-2.0);
    quadrant = _mm256_blendv_pd(quadrant, neg_two, xs_negative_mask);
    __m256d one = _mm256_set1_pd(1.0);
    __m256d quadrant_incremented = _mm256_add_pd(quadrant, one);
    quadrant = _mm256_blendv_pd(quadrant, quadrant_incremented, swap_mask);

    // Adjust for quadrant: result += quadrant * π/2
    __m256d pi_half = _mm256_set1_pd(1.5707963267948966);
    results = _mm256_fmadd_pd(quadrant, pi_half, results);

    // Transfer sign from x (XOR with sign bit of x_input)
    __m256d xs_sign_bits = _mm256_and_pd(xs_inputs, sign_mask);
    results = _mm256_xor_pd(results, xs_sign_bits);

    // Transfer sign from y (XOR with sign bit of y_input)
    __m256d ys_sign_bits = _mm256_and_pd(ys_inputs, sign_mask);
    results = _mm256_xor_pd(results, ys_sign_bits);

    return results;
}

NK_PUBLIC void nk_each_sin_f32_haswell(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 angles = _mm256_loadu_ps(ins + i);
        __m256 results = nk_sin_f32x8_haswell_(angles);
        _mm256_storeu_ps(outs + i, results);
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
        __m256 angles = _mm256_loadu_ps(ins + i);
        __m256 results = nk_cos_f32x8_haswell_(angles);
        _mm256_storeu_ps(outs + i, results);
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
        __m256 values = _mm256_loadu_ps(ins + i);
        __m256 results = nk_atan_f32x8_haswell_(values);
        _mm256_storeu_ps(outs + i, results);
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
        __m256d angles = _mm256_loadu_pd(ins + i);
        __m256d results = nk_sin_f64x4_haswell_(angles);
        _mm256_storeu_pd(outs + i, results);
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
        __m256d angles = _mm256_loadu_pd(ins + i);
        __m256d results = nk_cos_f64x4_haswell_(angles);
        _mm256_storeu_pd(outs + i, results);
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
        __m256d values = _mm256_loadu_pd(ins + i);
        __m256d results = nk_atan_f64x4_haswell_(values);
        _mm256_storeu_pd(outs + i, results);
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

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_
#endif // NK_TRIGONOMETRY_HASWELL_H
