/**
 *  @brief SIMD-accelerated Trigonometric Functions for Skylake.
 *  @file include/numkong/trigonometry/skylake.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/trigonometry.h
 *  @see https://sleef.org
 *
 *  @section skylake_trig_instructions Key AVX-512 Trigonometry Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      _mm512_fmadd_ps             VFMADD132PS (ZMM, ZMM, ZMM)     4cy         0.5/cy      p05
 *      _mm512_mul_ps               VMULPS (ZMM, ZMM, ZMM)          4cy         0.5/cy      p05
 *      _mm512_and_ps               VANDPS (ZMM, ZMM, ZMM)          1cy         0.33/cy     p015
 *      _mm512_cmp_ps_mask          VCMPPS (K, ZMM, ZMM, I8)        3cy         1/cy        p01
 *      _mm512_roundscale_ps        VRNDSCALEPS (ZMM, ZMM, I8)      8cy         0.5/cy      p01
 *
 *  Trigonometric functions use polynomial approximations evaluated via Horner's method with FMA chains.
 *  AVX-512 mask registers enable branchless range reduction and sign handling without blend overhead.
 *  Skylake-X's dual FMA units achieve 0.5cy throughput, processing 32 f32 sin/cos values per 8 cycles.
 */
#ifndef NK_TRIGONOMETRY_SKYLAKE_H
#define NK_TRIGONOMETRY_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

NK_INTERNAL __m512 nk_sin_f32x16_skylake_(__m512 const angles_radians) {
    // Cody-Waite constants for argument reduction
    __m512 const pi_hi_f32x16 = _mm512_set1_ps(3.1415927f);
    __m512 const pi_lo_f32x16 = _mm512_set1_ps(-8.742278e-8f);
    __m512 const pi_reciprocal = _mm512_set1_ps(0.31830988618379067154f); // 1/π
    // Degree-9 minimax coefficients
    __m512 const coeff_9 = _mm512_set1_ps(+2.7557319224e-6f);
    __m512 const coeff_7 = _mm512_set1_ps(-1.9841269841e-4f);
    __m512 const coeff_5 = _mm512_set1_ps(+8.3333293855e-3f);
    __m512 const coeff_3 = _mm512_set1_ps(-1.6666666641e-1f);

    // Compute (multiples_of_pi) = round(angle / π)
    __m512 quotients = _mm512_mul_ps(angles_radians, pi_reciprocal);
    __m512 rounded_quotients = _mm512_roundscale_ps(quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // Use explicit rounding to match roundscale (MXCSR-independent)
    __m512i multiples_of_pi = _mm512_cvt_roundps_epi32(rounded_quotients,
                                                       _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Cody-Waite range reduction
    __m512 angles = _mm512_fnmadd_ps(rounded_quotients, pi_hi_f32x16, angles_radians);
    angles = _mm512_fnmadd_ps(rounded_quotients, pi_lo_f32x16, angles);
    __m512 const angles_squared = _mm512_mul_ps(angles, angles);
    __m512 const angles_cubed = _mm512_mul_ps(angles, angles_squared);

    // Degree-9 polynomial via Horner's method
    __m512 polynomials = coeff_9;
    polynomials = _mm512_fmadd_ps(polynomials, angles_squared, coeff_7);
    polynomials = _mm512_fmadd_ps(polynomials, angles_squared, coeff_5);
    polynomials = _mm512_fmadd_ps(polynomials, angles_squared, coeff_3);

    // If multiples_of_pi is odd, flip the sign of the results
    __mmask16 odd_mask = _mm512_test_epi32_mask(multiples_of_pi, _mm512_set1_epi32(1));
    __m512 results = _mm512_fmadd_ps(angles_cubed, polynomials, angles);
    results = _mm512_mask_sub_ps(results, odd_mask, _mm512_setzero_ps(), results);
    return results;
}

NK_INTERNAL __m512 nk_cos_f32x16_skylake_(__m512 const angles_radians) {
    // Cody-Waite constants for argument reduction
    __m512 const pi_hi_f32x16 = _mm512_set1_ps(3.1415927f);
    __m512 const pi_lo_f32x16 = _mm512_set1_ps(-8.742278e-8f);
    __m512 const pi_half = _mm512_set1_ps(1.57079632679489661923f);       // π/2
    __m512 const pi_reciprocal = _mm512_set1_ps(0.31830988618379067154f); // 1/π
    // Degree-9 minimax coefficients
    __m512 const coeff_9 = _mm512_set1_ps(+2.7557319224e-6f);
    __m512 const coeff_7 = _mm512_set1_ps(-1.9841269841e-4f);
    __m512 const coeff_5 = _mm512_set1_ps(+8.3333293855e-3f);
    __m512 const coeff_3 = _mm512_set1_ps(-1.6666666641e-1f);

    // Compute (multiples_of_pi) = round((angle / π) - 0.5)
    __m512 quotients = _mm512_fmsub_ps(angles_radians, pi_reciprocal, _mm512_set1_ps(0.5f));
    __m512 rounded_quotients = _mm512_roundscale_ps(quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // Use explicit rounding to match roundscale (MXCSR-independent)
    __m512i multiples_of_pi = _mm512_cvt_roundps_epi32(rounded_quotients,
                                                       _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Cody-Waite range reduction: angle = angle_radians - (multiples * pi + pi/2)
    __m512 const offset = _mm512_fmadd_ps(rounded_quotients, pi_hi_f32x16, pi_half);
    __m512 angles = _mm512_sub_ps(angles_radians, offset);
    angles = _mm512_fnmadd_ps(rounded_quotients, pi_lo_f32x16, angles);
    __m512 const angles_squared = _mm512_mul_ps(angles, angles);
    __m512 const angles_cubed = _mm512_mul_ps(angles, angles_squared);

    // Degree-9 polynomial via Horner's method
    __m512 polynomials = coeff_9;
    polynomials = _mm512_fmadd_ps(polynomials, angles_squared, coeff_7);
    polynomials = _mm512_fmadd_ps(polynomials, angles_squared, coeff_5);
    polynomials = _mm512_fmadd_ps(polynomials, angles_squared, coeff_3);
    __m512 results = _mm512_fmadd_ps(angles_cubed, polynomials, angles);

    // If multiples_of_pi is even, flip the sign of the results
    __mmask16 even_mask = _mm512_testn_epi32_mask(multiples_of_pi, _mm512_set1_epi32(1));
    results = _mm512_mask_sub_ps(results, even_mask, _mm512_setzero_ps(), results);
    return results;
}

NK_INTERNAL __m512 nk_atan_f32x16_skylake_(__m512 const inputs) {
    // Polynomial coefficients
    __m512 const coeff_8 = _mm512_set1_ps(-0.333331018686294555664062f);
    __m512 const coeff_7 = _mm512_set1_ps(+0.199926957488059997558594f);
    __m512 const coeff_6 = _mm512_set1_ps(-0.142027363181114196777344f);
    __m512 const coeff_5 = _mm512_set1_ps(+0.106347933411598205566406f);
    __m512 const coeff_4 = _mm512_set1_ps(-0.0748900920152664184570312f);
    __m512 const coeff_3 = _mm512_set1_ps(+0.0425049886107444763183594f);
    __m512 const coeff_2 = _mm512_set1_ps(-0.0159569028764963150024414f);
    __m512 const coeff_1 = _mm512_set1_ps(+0.00282363896258175373077393f);

    // Adjust for quadrant
    __m512 values = inputs;
    __mmask16 const negative_mask = _mm512_fpclass_ps_mask(values, 0x40);
    values = _mm512_abs_ps(values);
    __mmask16 const reciprocal_mask = _mm512_cmp_ps_mask(values, _mm512_set1_ps(1.0f), _CMP_GT_OS);
    values = _mm512_mask_div_ps(values, reciprocal_mask, _mm512_set1_ps(1.0f), values);

    // Argument reduction
    __m512 const values_squared = _mm512_mul_ps(values, values);
    __m512 const values_cubed = _mm512_mul_ps(values, values_squared);

    // Polynomial evaluation
    __m512 polynomials = coeff_1;
    polynomials = _mm512_fmadd_ps(polynomials, values_squared, coeff_2);
    polynomials = _mm512_fmadd_ps(polynomials, values_squared, coeff_3);
    polynomials = _mm512_fmadd_ps(polynomials, values_squared, coeff_4);
    polynomials = _mm512_fmadd_ps(polynomials, values_squared, coeff_5);
    polynomials = _mm512_fmadd_ps(polynomials, values_squared, coeff_6);
    polynomials = _mm512_fmadd_ps(polynomials, values_squared, coeff_7);
    polynomials = _mm512_fmadd_ps(polynomials, values_squared, coeff_8);

    // Adjust result for quadrants
    __m512 result = _mm512_fmadd_ps(values_cubed, polynomials, values);
    result = _mm512_mask_sub_ps(result, reciprocal_mask, _mm512_set1_ps(1.5707963267948966f), result);
    result = _mm512_mask_sub_ps(result, negative_mask, _mm512_setzero_ps(), result);
    return result;
}

NK_INTERNAL __m512 nk_atan2_f32x16_skylake_(__m512 const ys_inputs, __m512 const xs_inputs) {
    // Polynomial coefficients
    __m512 const coeff_8 = _mm512_set1_ps(-0.333331018686294555664062f);
    __m512 const coeff_7 = _mm512_set1_ps(+0.199926957488059997558594f);
    __m512 const coeff_6 = _mm512_set1_ps(-0.142027363181114196777344f);
    __m512 const coeff_5 = _mm512_set1_ps(+0.106347933411598205566406f);
    __m512 const coeff_4 = _mm512_set1_ps(-0.0748900920152664184570312f);
    __m512 const coeff_3 = _mm512_set1_ps(+0.0425049886107444763183594f);
    __m512 const coeff_2 = _mm512_set1_ps(-0.0159569028764963150024414f);
    __m512 const coeff_1 = _mm512_set1_ps(+0.00282363896258175373077393f);

    // Quadrant adjustments normalizing to absolute values of x and y
    __mmask16 const xs_negative_mask = _mm512_fpclass_ps_mask(xs_inputs, 0x40);
    __m512 xs = _mm512_abs_ps(xs_inputs);
    __m512 ys = _mm512_abs_ps(ys_inputs);
    // Ensure proper fraction where the numerator is smaller than the denominator
    __mmask16 const swap_mask = _mm512_cmp_ps_mask(ys, xs, _CMP_GT_OS);
    __m512 temps = xs;
    xs = _mm512_mask_blend_ps(swap_mask, xs, ys);
    ys = _mm512_mask_sub_ps(ys, swap_mask, _mm512_setzero_ps(), temps);

    // Compute ratio and ratio²
    __m512 const ratio = _mm512_div_ps(ys, xs);
    __m512 const ratio_squared = _mm512_mul_ps(ratio, ratio);
    __m512 const ratio_cubed = _mm512_mul_ps(ratio, ratio_squared);

    // Polynomial evaluation
    __m512 polynomials = coeff_1;
    polynomials = _mm512_fmadd_ps(polynomials, ratio_squared, coeff_2);
    polynomials = _mm512_fmadd_ps(polynomials, ratio_squared, coeff_3);
    polynomials = _mm512_fmadd_ps(polynomials, ratio_squared, coeff_4);
    polynomials = _mm512_fmadd_ps(polynomials, ratio_squared, coeff_5);
    polynomials = _mm512_fmadd_ps(polynomials, ratio_squared, coeff_6);
    polynomials = _mm512_fmadd_ps(polynomials, ratio_squared, coeff_7);
    polynomials = _mm512_fmadd_ps(polynomials, ratio_squared, coeff_8);

    // Compute quadrant value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    __m512 results = _mm512_fmadd_ps(ratio_cubed, polynomials, ratio);
    __m512 quadrant = _mm512_setzero_ps();
    __m512 neg_two = _mm512_set1_ps(-2.0f);
    quadrant = _mm512_mask_blend_ps(xs_negative_mask, quadrant, neg_two);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 quadrant_incremented = _mm512_add_ps(quadrant, one);
    quadrant = _mm512_mask_blend_ps(swap_mask, quadrant, quadrant_incremented);

    // Adjust for quadrant: result += quadrant * π/2
    __m512 pi_half = _mm512_set1_ps(1.5707963267948966f);
    results = _mm512_fmadd_ps(quadrant, pi_half, results);

    // Transfer sign from x (XOR with sign bit of x_input)
    __m512 xs_sign_bits = _mm512_and_ps(xs_inputs, _mm512_set1_ps(-0.0f));
    results = _mm512_xor_ps(results, xs_sign_bits);

    // Transfer sign from y (XOR with sign bit of y_input)
    __m512 ys_sign_bits = _mm512_and_ps(ys_inputs, _mm512_set1_ps(-0.0f));
    results = _mm512_xor_ps(results, ys_sign_bits);

    return results;
}

NK_PUBLIC void nk_each_sin_f32_skylake(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 angles = _mm512_loadu_ps(ins + i);
        __m512 results = nk_sin_f32x16_skylake_(angles);
        _mm512_storeu_ps(outs + i, results);
    }
    if (i < n) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m512 angles = _mm512_maskz_loadu_ps(mask, ins + i);
        __m512 results = nk_sin_f32x16_skylake_(angles);
        _mm512_mask_storeu_ps(outs + i, mask, results);
    }
}
NK_PUBLIC void nk_each_cos_f32_skylake(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 angles = _mm512_loadu_ps(ins + i);
        __m512 results = nk_cos_f32x16_skylake_(angles);
        _mm512_storeu_ps(outs + i, results);
    }
    if (i < n) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m512 angles = _mm512_maskz_loadu_ps(mask, ins + i);
        __m512 results = nk_cos_f32x16_skylake_(angles);
        _mm512_mask_storeu_ps(outs + i, mask, results);
    }
}
NK_PUBLIC void nk_each_atan_f32_skylake(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 angles = _mm512_loadu_ps(ins + i);
        __m512 results = nk_atan_f32x16_skylake_(angles);
        _mm512_storeu_ps(outs + i, results);
    }
    if (i < n) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m512 angles = _mm512_maskz_loadu_ps(mask, ins + i);
        __m512 results = nk_atan_f32x16_skylake_(angles);
        _mm512_mask_storeu_ps(outs + i, mask, results);
    }
}

NK_INTERNAL __m512d nk_sin_f64x8_skylake_(__m512d const angles_radians) {
    // Constants for argument reduction
    __m512d const pi_high = _mm512_set1_pd(3.141592653589793116);         // High-digits part of π
    __m512d const pi_low = _mm512_set1_pd(1.2246467991473532072e-16);     // Low-digits part of π
    __m512d const pi_reciprocal = _mm512_set1_pd(0.31830988618379067154); // 1/π

    // Polynomial coefficients for sine/cosine approximation (minimax polynomial)
    __m512d const coeff_0 = _mm512_set1_pd(+0.00833333333333332974823815);
    __m512d const coeff_1 = _mm512_set1_pd(-0.000198412698412696162806809);
    __m512d const coeff_2 = _mm512_set1_pd(+2.75573192239198747630416e-06);
    __m512d const coeff_3 = _mm512_set1_pd(-2.50521083763502045810755e-08);
    __m512d const coeff_4 = _mm512_set1_pd(+1.60590430605664501629054e-10);
    __m512d const coeff_5 = _mm512_set1_pd(-7.64712219118158833288484e-13);
    __m512d const coeff_6 = _mm512_set1_pd(+2.81009972710863200091251e-15);
    __m512d const coeff_7 = _mm512_set1_pd(-7.97255955009037868891952e-18);
    __m512d const coeff_8 = _mm512_set1_pd(-0.166666666666666657414808);

    // Compute (rounded_quotients) = round(angle / π)
    __m512d const quotients = _mm512_mul_pd(angles_radians, pi_reciprocal);
    __m512d const rounded_quotients = _mm512_roundscale_pd(quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Reduce the angle to: angle - (rounded_quotients * π_high + rounded_quotients * π_low)
    __m512d angles = angles_radians;
    angles = _mm512_fnmadd_pd(rounded_quotients, pi_high, angles);
    angles = _mm512_fnmadd_pd(rounded_quotients, pi_low, angles);

    // If rounded_quotients is odd (bit 0 set), negate the angle
    // Use explicit rounding to match roundscale (MXCSR-independent)
    __mmask8 const sign_flip_mask = _mm256_test_epi32_mask(
        _mm512_cvt_roundpd_epi32(rounded_quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
        _mm256_set1_epi32(1));
    angles = _mm512_mask_sub_pd(angles, sign_flip_mask, _mm512_setzero_pd(), angles);

    __m512d const angles_squared = _mm512_mul_pd(angles, angles);
    __m512d const angles_cubed = _mm512_mul_pd(angles, angles_squared);
    __m512d const angles_quadratic = _mm512_mul_pd(angles_squared, angles_squared);
    __m512d const angles_octic = _mm512_mul_pd(angles_quadratic, angles_quadratic);

    // Compute higher-degree polynomial terms
    __m512d const poly_67 = _mm512_fmadd_pd(angles_squared, coeff_7, coeff_6);
    __m512d const poly_45 = _mm512_fmadd_pd(angles_squared, coeff_5, coeff_4);
    __m512d const poly_4567 = _mm512_fmadd_pd(angles_quadratic, poly_67, poly_45);

    // Compute lower-degree polynomial terms
    __m512d const poly_23 = _mm512_fmadd_pd(angles_squared, coeff_3, coeff_2);
    __m512d const poly_01 = _mm512_fmadd_pd(angles_squared, coeff_1, coeff_0);
    __m512d const poly_0123 = _mm512_fmadd_pd(angles_quadratic, poly_23, poly_01);

    // Combine polynomial terms
    __m512d results = _mm512_fmadd_pd(angles_octic, poly_4567, poly_0123);
    results = _mm512_fmadd_pd(results, angles_squared, coeff_8);
    results = _mm512_fmadd_pd(results, angles_cubed, angles);

    // Handle the special case of negative zero input
    __mmask8 const non_zero_mask = _mm512_cmpneq_pd_mask(angles_radians, _mm512_setzero_pd());
    results = _mm512_maskz_mov_pd(non_zero_mask, results);
    return results;
}

NK_INTERNAL __m512d nk_cos_f64x8_skylake_(__m512d const angles_radians) {
    // Constants for argument reduction
    __m512d const pi_high_half = _mm512_set1_pd(3.141592653589793116 * 0.5);     // High-digits part of π
    __m512d const pi_low_half = _mm512_set1_pd(1.2246467991473532072e-16 * 0.5); // Low-digits part of π
    __m512d const pi_reciprocal = _mm512_set1_pd(0.31830988618379067154);        // 1/π

    // Polynomial coefficients for sine/cosine approximation (minimax polynomial)
    __m512d const coeff_0 = _mm512_set1_pd(+0.00833333333333332974823815);
    __m512d const coeff_1 = _mm512_set1_pd(-0.000198412698412696162806809);
    __m512d const coeff_2 = _mm512_set1_pd(+2.75573192239198747630416e-06);
    __m512d const coeff_3 = _mm512_set1_pd(-2.50521083763502045810755e-08);
    __m512d const coeff_4 = _mm512_set1_pd(+1.60590430605664501629054e-10);
    __m512d const coeff_5 = _mm512_set1_pd(-7.64712219118158833288484e-13);
    __m512d const coeff_6 = _mm512_set1_pd(+2.81009972710863200091251e-15);
    __m512d const coeff_7 = _mm512_set1_pd(-7.97255955009037868891952e-18);
    __m512d const coeff_8 = _mm512_set1_pd(-0.166666666666666657414808);

    // Compute (rounded_quotients) = 2 * round(angle / π - 0.5) + 1
    // Use fmsub: a*b - c = angles * (1/π) - 0.5
    __m512d const quotients = _mm512_fmsub_pd(angles_radians, pi_reciprocal, _mm512_set1_pd(0.5));
    __m512d const rounded_quotients = _mm512_fmadd_pd(                                  //
        _mm512_set1_pd(2),                                                              //
        _mm512_roundscale_pd(quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), //
        _mm512_set1_pd(1));

    // Reduce the angle to: angle - (rounded_quotients * π_high + rounded_quotients * π_low)
    __m512d angles = angles_radians;
    angles = _mm512_fnmadd_pd(rounded_quotients, pi_high_half, angles);
    angles = _mm512_fnmadd_pd(rounded_quotients, pi_low_half, angles);
    // Use explicit rounding to match roundscale (MXCSR-independent)
    __mmask8 const sign_flip_mask = _mm256_testn_epi32_mask(
        _mm512_cvt_roundpd_epi32(rounded_quotients, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
        _mm256_set1_epi32(2));
    angles = _mm512_mask_sub_pd(angles, sign_flip_mask, _mm512_setzero_pd(), angles);
    __m512d const angles_squared = _mm512_mul_pd(angles, angles);
    __m512d const angles_cubed = _mm512_mul_pd(angles, angles_squared);
    __m512d const angles_quadratic = _mm512_mul_pd(angles_squared, angles_squared);
    __m512d const angles_octic = _mm512_mul_pd(angles_quadratic, angles_quadratic);

    // Compute higher-degree polynomial terms
    __m512d const poly_67 = _mm512_fmadd_pd(angles_squared, coeff_7, coeff_6);
    __m512d const poly_45 = _mm512_fmadd_pd(angles_squared, coeff_5, coeff_4);
    __m512d const poly_4567 = _mm512_fmadd_pd(angles_quadratic, poly_67, poly_45);

    // Compute lower-degree polynomial terms
    __m512d const poly_23 = _mm512_fmadd_pd(angles_squared, coeff_3, coeff_2);
    __m512d const poly_01 = _mm512_fmadd_pd(angles_squared, coeff_1, coeff_0);
    __m512d const poly_0123 = _mm512_fmadd_pd(angles_quadratic, poly_23, poly_01);

    // Combine polynomial terms
    __m512d results = _mm512_fmadd_pd(angles_octic, poly_4567, poly_0123);
    results = _mm512_fmadd_pd(results, angles_squared, coeff_8);
    results = _mm512_fmadd_pd(results, angles_cubed, angles);
    return results;
}

NK_INTERNAL __m512d nk_atan_f64x8_skylake_(__m512d const inputs) {
    // Polynomial coefficients for atan approximation
    __m512d const coeff_19 = _mm512_set1_pd(-1.88796008463073496563746e-05);
    __m512d const coeff_18 = _mm512_set1_pd(+0.000209850076645816976906797);
    __m512d const coeff_17 = _mm512_set1_pd(-0.00110611831486672482563471);
    __m512d const coeff_16 = _mm512_set1_pd(+0.00370026744188713119232403);
    __m512d const coeff_15 = _mm512_set1_pd(-0.00889896195887655491740809);
    __m512d const coeff_14 = _mm512_set1_pd(+0.016599329773529201970117);
    __m512d const coeff_13 = _mm512_set1_pd(-0.0254517624932312641616861);
    __m512d const coeff_12 = _mm512_set1_pd(+0.0337852580001353069993897);
    __m512d const coeff_11 = _mm512_set1_pd(-0.0407629191276836500001934);
    __m512d const coeff_10 = _mm512_set1_pd(+0.0466667150077840625632675);
    __m512d const coeff_9 = _mm512_set1_pd(-0.0523674852303482457616113);
    __m512d const coeff_8 = _mm512_set1_pd(+0.0587666392926673580854313);
    __m512d const coeff_7 = _mm512_set1_pd(-0.0666573579361080525984562);
    __m512d const coeff_6 = _mm512_set1_pd(+0.0769219538311769618355029);
    __m512d const coeff_5 = _mm512_set1_pd(-0.090908995008245008229153);
    __m512d const coeff_4 = _mm512_set1_pd(+0.111111105648261418443745);
    __m512d const coeff_3 = _mm512_set1_pd(-0.14285714266771329383765);
    __m512d const coeff_2 = _mm512_set1_pd(+0.199999999996591265594148);
    __m512d const coeff_1 = _mm512_set1_pd(-0.333333333333311110369124);

    // Quadrant adjustments
    __mmask8 negative_mask = _mm512_cmp_pd_mask(inputs, _mm512_setzero_pd(), _CMP_LT_OS);
    __m512d values = _mm512_abs_pd(inputs);
    __mmask8 reciprocal_mask = _mm512_cmp_pd_mask(values, _mm512_set1_pd(1.0), _CMP_GT_OS);
    values = _mm512_mask_div_pd(values, reciprocal_mask, _mm512_set1_pd(1.0), values);
    __m512d const values_squared = _mm512_mul_pd(values, values);
    __m512d const values_cubed = _mm512_mul_pd(values, values_squared);

    // Polynomial evaluation (argument reduction and approximation)
    __m512d polynomials = coeff_19;
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_18);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_17);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_16);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_15);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_14);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_13);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_12);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_11);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_10);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_9);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_8);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_7);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_6);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_5);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_4);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_3);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_2);
    polynomials = _mm512_fmadd_pd(polynomials, values_squared, coeff_1);

    // Compute atan approximation
    __m512d result = _mm512_fmadd_pd(values_cubed, polynomials, values);
    result = _mm512_mask_sub_pd(result, reciprocal_mask, _mm512_set1_pd(1.5707963267948966), result);
    result = _mm512_mask_sub_pd(result, negative_mask, _mm512_setzero_pd(), result);
    return result;
}

/**
 *  @brief AVX-512 implementation of atan2(y, x) for 8 double-precision values.
 *  @see Based on the f32x16 version with appropriate precision constants.
 */
NK_INTERNAL __m512d nk_atan2_f64x8_skylake_(__m512d const ys_inputs, __m512d const xs_inputs) {
    // Polynomial coefficients for atan approximation (higher precision than f32)
    __m512d const coeff_19 = _mm512_set1_pd(-1.88796008463073496563746e-05);
    __m512d const coeff_18 = _mm512_set1_pd(+0.000209850076645816976906797);
    __m512d const coeff_17 = _mm512_set1_pd(-0.00110611831486672482563471);
    __m512d const coeff_16 = _mm512_set1_pd(+0.00370026744188713119232403);
    __m512d const coeff_15 = _mm512_set1_pd(-0.00889896195887655491740809);
    __m512d const coeff_14 = _mm512_set1_pd(+0.016599329773529201970117);
    __m512d const coeff_13 = _mm512_set1_pd(-0.0254517624932312641616861);
    __m512d const coeff_12 = _mm512_set1_pd(+0.0337852580001353069993897);
    __m512d const coeff_11 = _mm512_set1_pd(-0.0407629191276836500001934);
    __m512d const coeff_10 = _mm512_set1_pd(+0.0466667150077840625632675);
    __m512d const coeff_9 = _mm512_set1_pd(-0.0523674852303482457616113);
    __m512d const coeff_8 = _mm512_set1_pd(+0.0587666392926673580854313);
    __m512d const coeff_7 = _mm512_set1_pd(-0.0666573579361080525984562);
    __m512d const coeff_6 = _mm512_set1_pd(+0.0769219538311769618355029);
    __m512d const coeff_5 = _mm512_set1_pd(-0.090908995008245008229153);
    __m512d const coeff_4 = _mm512_set1_pd(+0.111111105648261418443745);
    __m512d const coeff_3 = _mm512_set1_pd(-0.14285714266771329383765);
    __m512d const coeff_2 = _mm512_set1_pd(+0.199999999996591265594148);
    __m512d const coeff_1 = _mm512_set1_pd(-0.333333333333311110369124);

    // Quadrant adjustments normalizing to absolute values of x and y
    __mmask8 const xs_negative_mask = _mm512_cmp_pd_mask(xs_inputs, _mm512_setzero_pd(), _CMP_LT_OS);
    __m512d xs = _mm512_abs_pd(xs_inputs);
    __m512d ys = _mm512_abs_pd(ys_inputs);
    // Ensure proper fraction where the numerator is smaller than the denominator
    __mmask8 const swap_mask = _mm512_cmp_pd_mask(ys, xs, _CMP_GT_OS);
    __m512d temps = xs;
    xs = _mm512_mask_blend_pd(swap_mask, xs, ys);
    ys = _mm512_mask_sub_pd(ys, swap_mask, _mm512_setzero_pd(), temps);

    // Compute ratio and ratio²
    __m512d const ratio = _mm512_div_pd(ys, xs);
    __m512d const ratio_squared = _mm512_mul_pd(ratio, ratio);
    __m512d const ratio_cubed = _mm512_mul_pd(ratio, ratio_squared);

    // Polynomial evaluation
    __m512d polynomials = coeff_19;
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_18);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_17);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_16);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_15);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_14);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_13);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_12);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_11);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_10);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_9);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_8);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_7);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_6);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_5);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_4);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_3);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_2);
    polynomials = _mm512_fmadd_pd(polynomials, ratio_squared, coeff_1);

    // Compute the result with quadrant adjustments
    __m512d results = _mm512_fmadd_pd(ratio_cubed, polynomials, ratio);

    // Compute quadrant value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    __m512d quadrant = _mm512_setzero_pd();
    quadrant = _mm512_mask_blend_pd(xs_negative_mask, quadrant, _mm512_set1_pd(-2.0));
    __m512d quadrant_incremented = _mm512_add_pd(quadrant, _mm512_set1_pd(1.0));
    quadrant = _mm512_mask_blend_pd(swap_mask, quadrant, quadrant_incremented);

    // Adjust for quadrant: result += quadrant * π/2
    results = _mm512_fmadd_pd(quadrant, _mm512_set1_pd(1.5707963267948966), results);

    // Transfer sign from x (XOR with sign bit of x_input)
    __m512d xs_sign = _mm512_and_pd(xs_inputs, _mm512_set1_pd(-0.0));
    results = _mm512_xor_pd(results, xs_sign);

    // Transfer sign from y (XOR with sign bit of y_input)
    __m512d ys_sign = _mm512_and_pd(ys_inputs, _mm512_set1_pd(-0.0));
    results = _mm512_xor_pd(results, ys_sign);

    return results;
}

NK_PUBLIC void nk_each_sin_f64_skylake(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d angles = _mm512_loadu_pd(ins + i);
        __m512d results = nk_sin_f64x8_skylake_(angles);
        _mm512_storeu_pd(outs + i, results);
    }
    if (i < n) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFF, n - i);
        __m512d angles = _mm512_maskz_loadu_pd(mask, ins + i);
        __m512d results = nk_sin_f64x8_skylake_(angles);
        _mm512_mask_storeu_pd(outs + i, mask, results);
    }
}
NK_PUBLIC void nk_each_cos_f64_skylake(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d angles = _mm512_loadu_pd(ins + i);
        __m512d results = nk_cos_f64x8_skylake_(angles);
        _mm512_storeu_pd(outs + i, results);
    }
    if (i < n) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFF, n - i);
        __m512d angles = _mm512_maskz_loadu_pd(mask, ins + i);
        __m512d results = nk_cos_f64x8_skylake_(angles);
        _mm512_mask_storeu_pd(outs + i, mask, results);
    }
}
NK_PUBLIC void nk_each_atan_f64_skylake(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d angles = _mm512_loadu_pd(ins + i);
        __m512d results = nk_atan_f64x8_skylake_(angles);
        _mm512_storeu_pd(outs + i, results);
    }
    if (i < n) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFF, n - i);
        __m512d angles = _mm512_maskz_loadu_pd(mask, ins + i);
        __m512d results = nk_atan_f64x8_skylake_(angles);
        _mm512_mask_storeu_pd(outs + i, mask, results);
    }
}

/**
 *  @brief Sine approximation for 16 f16 values via f32 upcasting.
 *
 *  Degree-5 polynomial with Cody-Waite range reduction in f32.
 *  Takes __m256i (f16 data), returns __m256i (f16 result).
 */
NK_INTERNAL __m256i nk_sin_f16x16_skylake_(__m256i angles_f16x16) {
    __m512 angles_f32x16 = _mm512_cvtph_ps(angles_f16x16);
    // Cody-Waite range reduction constants
    __m512 pi_hi_f32x16 = _mm512_set1_ps(3.1415927f);
    __m512 pi_lo_f32x16 = _mm512_set1_ps(-8.742278e-8f);
    __m512 pi_recip_f32x16 = _mm512_set1_ps(0.31830988618f);
    __m512 c3_f32x16 = _mm512_set1_ps(-1.6666666641e-1f);
    __m512 c5_f32x16 = _mm512_set1_ps(8.3333293855e-3f);

    __m512 quotient_f32x16 = _mm512_mul_ps(angles_f32x16, pi_recip_f32x16);
    __m512 rounded_f32x16 = _mm512_roundscale_ps(quotient_f32x16, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // Use explicit rounding to match roundscale (MXCSR-independent)
    __m512i multiple_i32x16 = _mm512_cvt_roundps_epi32(rounded_f32x16, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    angles_f32x16 = _mm512_fnmadd_ps(rounded_f32x16, pi_hi_f32x16, angles_f32x16);
    angles_f32x16 = _mm512_fnmadd_ps(rounded_f32x16, pi_lo_f32x16, angles_f32x16);

    __m512 x2_f32x16 = _mm512_mul_ps(angles_f32x16, angles_f32x16);
    __m512 poly_f32x16 = _mm512_fmadd_ps(c5_f32x16, x2_f32x16, c3_f32x16);
    poly_f32x16 = _mm512_mul_ps(poly_f32x16, x2_f32x16);
    __m512 result_f32x16 = _mm512_fmadd_ps(poly_f32x16, angles_f32x16, angles_f32x16);

    __mmask16 odd_mask = _mm512_test_epi32_mask(multiple_i32x16, _mm512_set1_epi32(1));
    result_f32x16 = _mm512_mask_sub_ps(result_f32x16, odd_mask, _mm512_setzero_ps(), result_f32x16);
    return _mm512_cvtps_ph(result_f32x16, _MM_FROUND_TO_NEAREST_INT);
}

/**
 *  @brief Cosine approximation for 16 f16 values via f32 upcasting.
 *
 *  Uses cos(x) = sin(x + pi/2) with Cody-Waite range reduction in f32.
 */
NK_INTERNAL __m256i nk_cos_f16x16_skylake_(__m256i angles_f16x16) {
    __m512 angles_f32x16 = _mm512_cvtph_ps(angles_f16x16);
    __m512 pi_hi_f32x16 = _mm512_set1_ps(3.1415927f);
    __m512 pi_lo_f32x16 = _mm512_set1_ps(-8.742278e-8f);
    __m512 pi_half_f32x16 = _mm512_set1_ps(1.5707963268f);
    __m512 pi_recip_f32x16 = _mm512_set1_ps(0.31830988618f);
    __m512 half_f32x16 = _mm512_set1_ps(0.5f);
    __m512 c3_f32x16 = _mm512_set1_ps(-1.6666666641e-1f);
    __m512 c5_f32x16 = _mm512_set1_ps(8.3333293855e-3f);

    __m512 quotient_f32x16 = _mm512_fmsub_ps(angles_f32x16, pi_recip_f32x16, half_f32x16);
    __m512 rounded_f32x16 = _mm512_roundscale_ps(quotient_f32x16, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // Use explicit rounding to match roundscale (MXCSR-independent)
    __m512i multiple_i32x16 = _mm512_cvt_roundps_epi32(rounded_f32x16, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m512 shift_f32x16 = _mm512_fmadd_ps(rounded_f32x16, pi_hi_f32x16, pi_half_f32x16);
    angles_f32x16 = _mm512_sub_ps(angles_f32x16, shift_f32x16);
    angles_f32x16 = _mm512_fnmadd_ps(rounded_f32x16, pi_lo_f32x16, angles_f32x16);

    __m512 x2_f32x16 = _mm512_mul_ps(angles_f32x16, angles_f32x16);
    __m512 poly_f32x16 = _mm512_fmadd_ps(c5_f32x16, x2_f32x16, c3_f32x16);
    poly_f32x16 = _mm512_mul_ps(poly_f32x16, x2_f32x16);
    __m512 result_f32x16 = _mm512_fmadd_ps(poly_f32x16, angles_f32x16, angles_f32x16);

    __mmask16 even_mask = _mm512_testn_epi32_mask(multiple_i32x16, _mm512_set1_epi32(1));
    result_f32x16 = _mm512_mask_sub_ps(result_f32x16, even_mask, _mm512_setzero_ps(), result_f32x16);
    return _mm512_cvtps_ph(result_f32x16, _MM_FROUND_TO_NEAREST_INT);
}

/**
 *  @brief Arctangent approximation for 16 f16 values via f32 upcasting.
 *
 *  Degree-9 polynomial in f32 with quadrant adjustments.
 */
NK_INTERNAL __m256i nk_atan_f16x16_skylake_(__m256i values_f16x16) {
    __m512 values_f32x16 = _mm512_cvtph_ps(values_f16x16);
    __m512 c3_f32x16 = _mm512_set1_ps(-0.3333333333f);
    __m512 c5_f32x16 = _mm512_set1_ps(0.2f);
    __m512 c7_f32x16 = _mm512_set1_ps(-0.1428571429f);
    __m512 c9_f32x16 = _mm512_set1_ps(0.1111111111f);
    __m512 pi_half_f32x16 = _mm512_set1_ps(1.5707963268f);
    __m512 one_f32x16 = _mm512_set1_ps(1.0f);

    __mmask16 negative_mask = _mm512_cmp_ps_mask(values_f32x16, _mm512_setzero_ps(), _CMP_LT_OS);
    values_f32x16 = _mm512_abs_ps(values_f32x16);
    __mmask16 reciprocal_mask = _mm512_cmp_ps_mask(values_f32x16, one_f32x16, _CMP_GT_OS);
    values_f32x16 = _mm512_mask_div_ps(values_f32x16, reciprocal_mask, one_f32x16, values_f32x16);

    __m512 x2_f32x16 = _mm512_mul_ps(values_f32x16, values_f32x16);
    __m512 x3_f32x16 = _mm512_mul_ps(values_f32x16, x2_f32x16);

    __m512 poly_f32x16 = c9_f32x16;
    poly_f32x16 = _mm512_fmadd_ps(poly_f32x16, x2_f32x16, c7_f32x16);
    poly_f32x16 = _mm512_fmadd_ps(poly_f32x16, x2_f32x16, c5_f32x16);
    poly_f32x16 = _mm512_fmadd_ps(poly_f32x16, x2_f32x16, c3_f32x16);

    __m512 result_f32x16 = _mm512_fmadd_ps(x3_f32x16, poly_f32x16, values_f32x16);
    result_f32x16 = _mm512_mask_sub_ps(result_f32x16, reciprocal_mask, pi_half_f32x16, result_f32x16);
    result_f32x16 = _mm512_mask_sub_ps(result_f32x16, negative_mask, _mm512_setzero_ps(), result_f32x16);
    return _mm512_cvtps_ph(result_f32x16, _MM_FROUND_TO_NEAREST_INT);
}

NK_PUBLIC void nk_each_sin_f16_skylake(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i angles_f16x16 = _mm256_loadu_si256((__m256i const *)(ins + i));
        __m256i result_f16x16 = nk_sin_f16x16_skylake_(angles_f16x16);
        _mm256_storeu_si256((__m256i *)(outs + i), result_f16x16);
    }
    if (i < n) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m256i angles_f16x16 = _mm256_maskz_loadu_epi16(mask, ins + i);
        __m256i result_f16x16 = nk_sin_f16x16_skylake_(angles_f16x16);
        _mm256_mask_storeu_epi16(outs + i, mask, result_f16x16);
    }
}

NK_PUBLIC void nk_each_cos_f16_skylake(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i angles_f16x16 = _mm256_loadu_si256((__m256i const *)(ins + i));
        __m256i result_f16x16 = nk_cos_f16x16_skylake_(angles_f16x16);
        _mm256_storeu_si256((__m256i *)(outs + i), result_f16x16);
    }
    if (i < n) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m256i angles_f16x16 = _mm256_maskz_loadu_epi16(mask, ins + i);
        __m256i result_f16x16 = nk_cos_f16x16_skylake_(angles_f16x16);
        _mm256_mask_storeu_epi16(outs + i, mask, result_f16x16);
    }
}

NK_PUBLIC void nk_each_atan_f16_skylake(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i values_f16x16 = _mm256_loadu_si256((__m256i const *)(ins + i));
        __m256i result_f16x16 = nk_atan_f16x16_skylake_(values_f16x16);
        _mm256_storeu_si256((__m256i *)(outs + i), result_f16x16);
    }
    if (i < n) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m256i values_f16x16 = _mm256_maskz_loadu_epi16(mask, ins + i);
        __m256i result_f16x16 = nk_atan_f16x16_skylake_(values_f16x16);
        _mm256_mask_storeu_epi16(outs + i, mask, result_f16x16);
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

#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_
#endif // NK_TRIGONOMETRY_SKYLAKE_H
