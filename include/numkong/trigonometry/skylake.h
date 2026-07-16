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
 *      Intrinsic             Instruction                  Skylake-X      Genoa
 *      _mm512_fmadd_ps       VFMADD132PS (ZMM, ZMM, ZMM)  4cy @ p05      4cy @ p01
 *      _mm512_mul_ps         VMULPS (ZMM, ZMM, ZMM)       4cy @ p05      3cy @ p01
 *      _mm512_and_ps         VANDPS (ZMM, ZMM, ZMM)       1cy @ p05      1cy @ p0123
 *      _mm512_cmp_ps_mask    VCMPPS (K, ZMM, ZMM, I8)     4cy @ p5       5cy @ p01
 *      _mm512_roundscale_ps  VRNDSCALEPS (ZMM, ZMM, I8)   8cy @ p05+p05  3cy @ p23
 *
 *  Trigonometric functions use polynomial approximations evaluated via Horner's method with FMA chains.
 *  AVX-512 mask registers enable branchless range reduction and sign handling without blend overhead.
 *  Skylake-X's dual FMA units achieve 0.5cy throughput, processing 32 f32 sin/cos values per 8 cycles.
 */
#ifndef NK_TRIGONOMETRY_SKYLAKE_H
#define NK_TRIGONOMETRY_SKYLAKE_H

#if NK_TARGET_X8664_
#if NK_TARGET_SKYLAKE

#include "numkong/types.h"
#include "numkong/cast/skylake.h" // `nk_load_bf16x16_to_f32x16_skylake_`

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

NK_HELPER_INLINE __m512 nk_sin_f32x16_skylake_(__m512 const angles_radians) {
    // Cody-Waite constants for argument reduction
    __m512 const pi_high_f32x16 = _mm512_set1_ps(3.1415927f);
    __m512 const pi_low_f32x16 = _mm512_set1_ps(-8.742278e-8f);
    __m512 const pi_reciprocal_f32x16 = _mm512_set1_ps(0.31830988618379067154f); // 1/π
    // Degree-9 minimax coefficients
    __m512 const coeff_9_f32x16 = _mm512_set1_ps(+2.7557319224e-6f);
    __m512 const coeff_7_f32x16 = _mm512_set1_ps(-1.9841269841e-4f);
    __m512 const coeff_5_f32x16 = _mm512_set1_ps(+8.3333293855e-3f);
    __m512 const coeff_3_f32x16 = _mm512_set1_ps(-1.6666666641e-1f);

    // Compute (multiples_of_pi_i32x16) = round(angle / π)
    __m512 quotients_f32x16 = _mm512_mul_ps(angles_radians, pi_reciprocal_f32x16);
    __m512 rounded_quotients_f32x16 = _mm512_roundscale_ps(quotients_f32x16,
                                                           _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // Use explicit rounding to match roundscale (MXCSR-independent)
    __m512i multiples_of_pi_i32x16 = _mm512_cvt_roundps_epi32(rounded_quotients_f32x16,
                                                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Cody-Waite range reduction
    __m512 angles_f32x16 = _mm512_fnmadd_ps(rounded_quotients_f32x16, pi_high_f32x16, angles_radians);
    angles_f32x16 = _mm512_fnmadd_ps(rounded_quotients_f32x16, pi_low_f32x16, angles_f32x16);
    __m512 const angles_squared_f32x16 = _mm512_mul_ps(angles_f32x16, angles_f32x16);
    __m512 const angles_cubed_f32x16 = _mm512_mul_ps(angles_f32x16, angles_squared_f32x16);

    // Degree-9 polynomial via Horner's method
    __m512 polynomials_f32x16 = coeff_9_f32x16;
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, angles_squared_f32x16, coeff_7_f32x16);
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, angles_squared_f32x16, coeff_5_f32x16);
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, angles_squared_f32x16, coeff_3_f32x16);

    // If multiples_of_pi_i32x16 is odd, flip the sign of the results_f32x16
    __mmask16 odd_m16 = _mm512_test_epi32_mask(multiples_of_pi_i32x16, _mm512_set1_epi32(1));
    __m512 results_f32x16 = _mm512_fmadd_ps(angles_cubed_f32x16, polynomials_f32x16, angles_f32x16);
    results_f32x16 = _mm512_mask_sub_ps(results_f32x16, odd_m16, _mm512_setzero_ps(), results_f32x16);
    return results_f32x16;
}

NK_HELPER_INLINE __m512 nk_cos_f32x16_skylake_(__m512 const angles_radians) {
    // Cody-Waite constants for argument reduction
    __m512 const pi_high_f32x16 = _mm512_set1_ps(3.1415927f);
    __m512 const pi_low_f32x16 = _mm512_set1_ps(-8.742278e-8f);
    __m512 const pi_half_f32x16 = _mm512_set1_ps(1.57079632679489661923f);       // π/2
    __m512 const pi_reciprocal_f32x16 = _mm512_set1_ps(0.31830988618379067154f); // 1/π
    // Degree-9 minimax coefficients
    __m512 const coeff_9_f32x16 = _mm512_set1_ps(+2.7557319224e-6f);
    __m512 const coeff_7_f32x16 = _mm512_set1_ps(-1.9841269841e-4f);
    __m512 const coeff_5_f32x16 = _mm512_set1_ps(+8.3333293855e-3f);
    __m512 const coeff_3_f32x16 = _mm512_set1_ps(-1.6666666641e-1f);

    // Compute (multiples_of_pi_i32x16) = round((angle / π) - 0.5)
    __m512 quotients_f32x16 = _mm512_fmsub_ps(angles_radians, pi_reciprocal_f32x16, _mm512_set1_ps(0.5f));
    __m512 rounded_quotients_f32x16 = _mm512_roundscale_ps(quotients_f32x16,
                                                           _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // Use explicit rounding to match roundscale (MXCSR-independent)
    __m512i multiples_of_pi_i32x16 = _mm512_cvt_roundps_epi32(rounded_quotients_f32x16,
                                                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Cody-Waite range reduction: angle = angle_radians - (multiples * pi + pi/2)
    __m512 const offset_f32x16 = _mm512_fmadd_ps(rounded_quotients_f32x16, pi_high_f32x16, pi_half_f32x16);
    __m512 angles_f32x16 = _mm512_sub_ps(angles_radians, offset_f32x16);
    angles_f32x16 = _mm512_fnmadd_ps(rounded_quotients_f32x16, pi_low_f32x16, angles_f32x16);
    __m512 const angles_squared_f32x16 = _mm512_mul_ps(angles_f32x16, angles_f32x16);
    __m512 const angles_cubed_f32x16 = _mm512_mul_ps(angles_f32x16, angles_squared_f32x16);

    // Degree-9 polynomial via Horner's method
    __m512 polynomials_f32x16 = coeff_9_f32x16;
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, angles_squared_f32x16, coeff_7_f32x16);
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, angles_squared_f32x16, coeff_5_f32x16);
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, angles_squared_f32x16, coeff_3_f32x16);
    __m512 results_f32x16 = _mm512_fmadd_ps(angles_cubed_f32x16, polynomials_f32x16, angles_f32x16);

    // If multiples_of_pi_i32x16 is even, flip the sign of the results_f32x16
    __mmask16 even_m16 = _mm512_testn_epi32_mask(multiples_of_pi_i32x16, _mm512_set1_epi32(1));
    results_f32x16 = _mm512_mask_sub_ps(results_f32x16, even_m16, _mm512_setzero_ps(), results_f32x16);
    return results_f32x16;
}

NK_HELPER_INLINE __m512 nk_atan_f32x16_skylake_(__m512 const inputs) {
    // Polynomial coefficients
    __m512 const coeff_8_f32x16 = _mm512_set1_ps(-0.333331018686294555664062f);
    __m512 const coeff_7_f32x16 = _mm512_set1_ps(+0.199926957488059997558594f);
    __m512 const coeff_6_f32x16 = _mm512_set1_ps(-0.142027363181114196777344f);
    __m512 const coeff_5_f32x16 = _mm512_set1_ps(+0.106347933411598205566406f);
    __m512 const coeff_4_f32x16 = _mm512_set1_ps(-0.0748900920152664184570312f);
    __m512 const coeff_3_f32x16 = _mm512_set1_ps(+0.0425049886107444763183594f);
    __m512 const coeff_2_f32x16 = _mm512_set1_ps(-0.0159569028764963150024414f);
    __m512 const coeff_1_f32x16 = _mm512_set1_ps(+0.00282363896258175373077393f);

    // Adjust for quadrant
    __m512 values_f32x16 = inputs;
    __mmask16 const negative_m16 = _mm512_fpclass_ps_mask(values_f32x16, 0x40);
    values_f32x16 = _mm512_abs_ps(values_f32x16);
    __mmask16 const reciprocal_m16 = _mm512_cmp_ps_mask(values_f32x16, _mm512_set1_ps(1.0f), _CMP_GT_OS);
    values_f32x16 = _mm512_mask_div_ps(values_f32x16, reciprocal_m16, _mm512_set1_ps(1.0f), values_f32x16);

    // Argument reduction
    __m512 const values_squared_f32x16 = _mm512_mul_ps(values_f32x16, values_f32x16);
    __m512 const values_cubed_f32x16 = _mm512_mul_ps(values_f32x16, values_squared_f32x16);

    // Polynomial evaluation
    __m512 polynomials_f32x16 = coeff_1_f32x16;
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, values_squared_f32x16, coeff_2_f32x16);
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, values_squared_f32x16, coeff_3_f32x16);
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, values_squared_f32x16, coeff_4_f32x16);
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, values_squared_f32x16, coeff_5_f32x16);
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, values_squared_f32x16, coeff_6_f32x16);
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, values_squared_f32x16, coeff_7_f32x16);
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, values_squared_f32x16, coeff_8_f32x16);

    // Adjust result_f32x16 for quadrants
    __m512 result_f32x16 = _mm512_fmadd_ps(values_cubed_f32x16, polynomials_f32x16, values_f32x16);
    result_f32x16 = _mm512_mask_sub_ps(result_f32x16, reciprocal_m16, _mm512_set1_ps(1.5707963267948966f),
                                       result_f32x16);
    result_f32x16 = _mm512_mask_sub_ps(result_f32x16, negative_m16, _mm512_setzero_ps(), result_f32x16);
    return result_f32x16;
}

NK_HELPER_INLINE __m512 nk_atan2_f32x16_skylake_(__m512 const ys_inputs, __m512 const xs_inputs) {
    // Polynomial coefficients
    __m512 const coeff_8_f32x16 = _mm512_set1_ps(-0.333331018686294555664062f);
    __m512 const coeff_7_f32x16 = _mm512_set1_ps(+0.199926957488059997558594f);
    __m512 const coeff_6_f32x16 = _mm512_set1_ps(-0.142027363181114196777344f);
    __m512 const coeff_5_f32x16 = _mm512_set1_ps(+0.106347933411598205566406f);
    __m512 const coeff_4_f32x16 = _mm512_set1_ps(-0.0748900920152664184570312f);
    __m512 const coeff_3_f32x16 = _mm512_set1_ps(+0.0425049886107444763183594f);
    __m512 const coeff_2_f32x16 = _mm512_set1_ps(-0.0159569028764963150024414f);
    __m512 const coeff_1_f32x16 = _mm512_set1_ps(+0.00282363896258175373077393f);

    // Quadrant adjustments normalizing to absolute values of x and y
    __mmask16 const xs_negative_m16 = _mm512_fpclass_ps_mask(xs_inputs, 0x40);
    __m512 xs_f32x16 = _mm512_abs_ps(xs_inputs);
    __m512 ys_f32x16 = _mm512_abs_ps(ys_inputs);
    // Ensure proper fraction where the numerator is smaller than the denominator
    __mmask16 const swap_m16 = _mm512_cmp_ps_mask(ys_f32x16, xs_f32x16, _CMP_GT_OS);
    __m512 temps_f32x16 = xs_f32x16;
    xs_f32x16 = _mm512_mask_blend_ps(swap_m16, xs_f32x16, ys_f32x16);
    ys_f32x16 = _mm512_mask_sub_ps(ys_f32x16, swap_m16, _mm512_setzero_ps(), temps_f32x16);

    // Compute ratio_f32x16 and ratio²
    __m512 const ratio_f32x16 = _mm512_div_ps(ys_f32x16, xs_f32x16);
    __m512 const ratio_squared_f32x16 = _mm512_mul_ps(ratio_f32x16, ratio_f32x16);
    __m512 const ratio_cubed_f32x16 = _mm512_mul_ps(ratio_f32x16, ratio_squared_f32x16);

    // Polynomial evaluation
    __m512 polynomials_f32x16 = coeff_1_f32x16;
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, ratio_squared_f32x16, coeff_2_f32x16);
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, ratio_squared_f32x16, coeff_3_f32x16);
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, ratio_squared_f32x16, coeff_4_f32x16);
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, ratio_squared_f32x16, coeff_5_f32x16);
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, ratio_squared_f32x16, coeff_6_f32x16);
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, ratio_squared_f32x16, coeff_7_f32x16);
    polynomials_f32x16 = _mm512_fmadd_ps(polynomials_f32x16, ratio_squared_f32x16, coeff_8_f32x16);

    // Compute quadrant_f32x16 value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    __m512 results_f32x16 = _mm512_fmadd_ps(ratio_cubed_f32x16, polynomials_f32x16, ratio_f32x16);
    __m512 quadrant_f32x16 = _mm512_setzero_ps();
    __m512 neg_two_f32x16 = _mm512_set1_ps(-2.0f);
    quadrant_f32x16 = _mm512_mask_blend_ps(xs_negative_m16, quadrant_f32x16, neg_two_f32x16);
    __m512 one_f32x16 = _mm512_set1_ps(1.0f);
    __m512 quadrant_incremented_f32x16 = _mm512_add_ps(quadrant_f32x16, one_f32x16);
    quadrant_f32x16 = _mm512_mask_blend_ps(swap_m16, quadrant_f32x16, quadrant_incremented_f32x16);

    // Adjust for quadrant_f32x16: result += quadrant_f32x16 * π/2
    __m512 pi_half_f32x16 = _mm512_set1_ps(1.5707963267948966f);
    results_f32x16 = _mm512_fmadd_ps(quadrant_f32x16, pi_half_f32x16, results_f32x16);

    // Transfer sign from x (XOR with sign bit of x_input)
    __m512 xs_sign_f32x16 = _mm512_and_ps(xs_inputs, _mm512_set1_ps(-0.0f));
    results_f32x16 = _mm512_xor_ps(results_f32x16, xs_sign_f32x16);

    // Transfer sign from y (XOR with sign bit of y_input)
    __m512 ys_sign_f32x16 = _mm512_and_ps(ys_inputs, _mm512_set1_ps(-0.0f));
    results_f32x16 = _mm512_xor_ps(results_f32x16, ys_sign_f32x16);

    return results_f32x16;
}

NK_API_COMPTIME void nk_trig_sin_f32_skylake(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 angles_f32x16 = _mm512_loadu_ps(ins + i);
        __m512 results_f32x16 = nk_sin_f32x16_skylake_(angles_f32x16);
        _mm512_storeu_ps(outs + i, results_f32x16);
    }
    if (i < n) {
        __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m512 angles_f32x16 = _mm512_maskz_loadu_ps(mask_m16, ins + i);
        __m512 results_f32x16 = nk_sin_f32x16_skylake_(angles_f32x16);
        _mm512_mask_storeu_ps(outs + i, mask_m16, results_f32x16);
    }
}
NK_API_COMPTIME void nk_trig_cos_f32_skylake(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 angles_f32x16 = _mm512_loadu_ps(ins + i);
        __m512 results_f32x16 = nk_cos_f32x16_skylake_(angles_f32x16);
        _mm512_storeu_ps(outs + i, results_f32x16);
    }
    if (i < n) {
        __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m512 angles_f32x16 = _mm512_maskz_loadu_ps(mask_m16, ins + i);
        __m512 results_f32x16 = nk_cos_f32x16_skylake_(angles_f32x16);
        _mm512_mask_storeu_ps(outs + i, mask_m16, results_f32x16);
    }
}
NK_API_COMPTIME void nk_trig_atan_f32_skylake(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 angles_f32x16 = _mm512_loadu_ps(ins + i);
        __m512 results_f32x16 = nk_atan_f32x16_skylake_(angles_f32x16);
        _mm512_storeu_ps(outs + i, results_f32x16);
    }
    if (i < n) {
        __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m512 angles_f32x16 = _mm512_maskz_loadu_ps(mask_m16, ins + i);
        __m512 results_f32x16 = nk_atan_f32x16_skylake_(angles_f32x16);
        _mm512_mask_storeu_ps(outs + i, mask_m16, results_f32x16);
    }
}

NK_HELPER_INLINE __m512d nk_sin_f64x8_skylake_(__m512d const angles_radians) {
    // Constants for argument reduction
    __m512d const pi_high_f64x8 = _mm512_set1_pd(3.141592653589793116);         // High-digits part of π
    __m512d const pi_low_f64x8 = _mm512_set1_pd(1.2246467991473532072e-16);     // Low-digits part of π
    __m512d const pi_reciprocal_f64x8 = _mm512_set1_pd(0.31830988618379067154); // 1/π

    // Polynomial coefficients for sine/cosine approximation (minimax polynomial)
    __m512d const coeff_0_f64x8 = _mm512_set1_pd(+0.00833333333333332974823815);
    __m512d const coeff_1_f64x8 = _mm512_set1_pd(-0.000198412698412696162806809);
    __m512d const coeff_2_f64x8 = _mm512_set1_pd(+2.75573192239198747630416e-06);
    __m512d const coeff_3_f64x8 = _mm512_set1_pd(-2.50521083763502045810755e-08);
    __m512d const coeff_4_f64x8 = _mm512_set1_pd(+1.60590430605664501629054e-10);
    __m512d const coeff_5_f64x8 = _mm512_set1_pd(-7.64712219118158833288484e-13);
    __m512d const coeff_6_f64x8 = _mm512_set1_pd(+2.81009972710863200091251e-15);
    __m512d const coeff_7_f64x8 = _mm512_set1_pd(-7.97255955009037868891952e-18);
    __m512d const coeff_8_f64x8 = _mm512_set1_pd(-0.166666666666666657414808);

    // Compute (rounded_quotients_f64x8) = round(angle / π)
    __m512d const quotients_f64x8 = _mm512_mul_pd(angles_radians, pi_reciprocal_f64x8);
    __m512d const rounded_quotients_f64x8 = _mm512_roundscale_pd(quotients_f64x8,
                                                                 _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Reduce the angle to: angle - (rounded_quotients_f64x8 * π_high + rounded_quotients_f64x8 * π_low)
    __m512d angles_f64x8 = angles_radians;
    angles_f64x8 = _mm512_fnmadd_pd(rounded_quotients_f64x8, pi_high_f64x8, angles_f64x8);
    angles_f64x8 = _mm512_fnmadd_pd(rounded_quotients_f64x8, pi_low_f64x8, angles_f64x8);

    // If rounded_quotients_f64x8 is odd (bit 0 set), negate the angle
    // Use explicit rounding to match roundscale (MXCSR-independent)
    __mmask8 const sign_flip_m8 = _mm256_test_epi32_mask(
        _mm512_cvt_roundpd_epi32(rounded_quotients_f64x8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
        _mm256_set1_epi32(1));
    angles_f64x8 = _mm512_mask_sub_pd(angles_f64x8, sign_flip_m8, _mm512_setzero_pd(), angles_f64x8);

    __m512d const angles_squared_f64x8 = _mm512_mul_pd(angles_f64x8, angles_f64x8);
    __m512d const angles_cubed_f64x8 = _mm512_mul_pd(angles_f64x8, angles_squared_f64x8);
    __m512d const angles_quadratic_f64x8 = _mm512_mul_pd(angles_squared_f64x8, angles_squared_f64x8);
    __m512d const angles_octic_f64x8 = _mm512_mul_pd(angles_quadratic_f64x8, angles_quadratic_f64x8);

    // Compute higher-degree polynomial terms
    __m512d const poly_67_f64x8 = _mm512_fmadd_pd(angles_squared_f64x8, coeff_7_f64x8, coeff_6_f64x8);
    __m512d const poly_45_f64x8 = _mm512_fmadd_pd(angles_squared_f64x8, coeff_5_f64x8, coeff_4_f64x8);
    __m512d const poly_4567_f64x8 = _mm512_fmadd_pd(angles_quadratic_f64x8, poly_67_f64x8, poly_45_f64x8);

    // Compute lower-degree polynomial terms
    __m512d const poly_23_f64x8 = _mm512_fmadd_pd(angles_squared_f64x8, coeff_3_f64x8, coeff_2_f64x8);
    __m512d const poly_01_f64x8 = _mm512_fmadd_pd(angles_squared_f64x8, coeff_1_f64x8, coeff_0_f64x8);
    __m512d const poly_0123_f64x8 = _mm512_fmadd_pd(angles_quadratic_f64x8, poly_23_f64x8, poly_01_f64x8);

    // Combine polynomial terms
    __m512d results_f64x8 = _mm512_fmadd_pd(angles_octic_f64x8, poly_4567_f64x8, poly_0123_f64x8);
    results_f64x8 = _mm512_fmadd_pd(results_f64x8, angles_squared_f64x8, coeff_8_f64x8);
    results_f64x8 = _mm512_fmadd_pd(results_f64x8, angles_cubed_f64x8, angles_f64x8);

    // Handle the special case of negative zero input
    __mmask8 const non_zero_m8 = _mm512_cmpneq_pd_mask(angles_radians, _mm512_setzero_pd());
    results_f64x8 = _mm512_maskz_mov_pd(non_zero_m8, results_f64x8);
    return results_f64x8;
}

NK_HELPER_INLINE __m512d nk_cos_f64x8_skylake_(__m512d const angles_radians) {
    // Constants for argument reduction
    __m512d const pi_high_half_f64x8 = _mm512_set1_pd(3.141592653589793116 * 0.5);     // High-digits part of π
    __m512d const pi_low_half_f64x8 = _mm512_set1_pd(1.2246467991473532072e-16 * 0.5); // Low-digits part of π
    __m512d const pi_reciprocal_f64x8 = _mm512_set1_pd(0.31830988618379067154);        // 1/π

    // Polynomial coefficients for sine/cosine approximation (minimax polynomial)
    __m512d const coeff_0_f64x8 = _mm512_set1_pd(+0.00833333333333332974823815);
    __m512d const coeff_1_f64x8 = _mm512_set1_pd(-0.000198412698412696162806809);
    __m512d const coeff_2_f64x8 = _mm512_set1_pd(+2.75573192239198747630416e-06);
    __m512d const coeff_3_f64x8 = _mm512_set1_pd(-2.50521083763502045810755e-08);
    __m512d const coeff_4_f64x8 = _mm512_set1_pd(+1.60590430605664501629054e-10);
    __m512d const coeff_5_f64x8 = _mm512_set1_pd(-7.64712219118158833288484e-13);
    __m512d const coeff_6_f64x8 = _mm512_set1_pd(+2.81009972710863200091251e-15);
    __m512d const coeff_7_f64x8 = _mm512_set1_pd(-7.97255955009037868891952e-18);
    __m512d const coeff_8_f64x8 = _mm512_set1_pd(-0.166666666666666657414808);

    // Compute (rounded_quotients_f64x8) = 2 * round(angle / π - 0.5) + 1
    // Use fmsub: a*b - c = angles_f64x8 * (1/π) - 0.5
    __m512d const quotients_f64x8 = _mm512_fmsub_pd(angles_radians, pi_reciprocal_f64x8, _mm512_set1_pd(0.5));
    __m512d const rounded_quotients_f64x8 = _mm512_fmadd_pd(                                  //
        _mm512_set1_pd(2),                                                                    //
        _mm512_roundscale_pd(quotients_f64x8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), //
        _mm512_set1_pd(1));

    // Reduce the angle to: angle - (rounded_quotients_f64x8 * π_high + rounded_quotients_f64x8 * π_low)
    __m512d angles_f64x8 = angles_radians;
    angles_f64x8 = _mm512_fnmadd_pd(rounded_quotients_f64x8, pi_high_half_f64x8, angles_f64x8);
    angles_f64x8 = _mm512_fnmadd_pd(rounded_quotients_f64x8, pi_low_half_f64x8, angles_f64x8);
    // Use explicit rounding to match roundscale (MXCSR-independent)
    __mmask8 const sign_flip_m8 = _mm256_testn_epi32_mask(
        _mm512_cvt_roundpd_epi32(rounded_quotients_f64x8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
        _mm256_set1_epi32(2));
    angles_f64x8 = _mm512_mask_sub_pd(angles_f64x8, sign_flip_m8, _mm512_setzero_pd(), angles_f64x8);
    __m512d const angles_squared_f64x8 = _mm512_mul_pd(angles_f64x8, angles_f64x8);
    __m512d const angles_cubed_f64x8 = _mm512_mul_pd(angles_f64x8, angles_squared_f64x8);
    __m512d const angles_quadratic_f64x8 = _mm512_mul_pd(angles_squared_f64x8, angles_squared_f64x8);
    __m512d const angles_octic_f64x8 = _mm512_mul_pd(angles_quadratic_f64x8, angles_quadratic_f64x8);

    // Compute higher-degree polynomial terms
    __m512d const poly_67_f64x8 = _mm512_fmadd_pd(angles_squared_f64x8, coeff_7_f64x8, coeff_6_f64x8);
    __m512d const poly_45_f64x8 = _mm512_fmadd_pd(angles_squared_f64x8, coeff_5_f64x8, coeff_4_f64x8);
    __m512d const poly_4567_f64x8 = _mm512_fmadd_pd(angles_quadratic_f64x8, poly_67_f64x8, poly_45_f64x8);

    // Compute lower-degree polynomial terms
    __m512d const poly_23_f64x8 = _mm512_fmadd_pd(angles_squared_f64x8, coeff_3_f64x8, coeff_2_f64x8);
    __m512d const poly_01_f64x8 = _mm512_fmadd_pd(angles_squared_f64x8, coeff_1_f64x8, coeff_0_f64x8);
    __m512d const poly_0123_f64x8 = _mm512_fmadd_pd(angles_quadratic_f64x8, poly_23_f64x8, poly_01_f64x8);

    // Combine polynomial terms
    __m512d results_f64x8 = _mm512_fmadd_pd(angles_octic_f64x8, poly_4567_f64x8, poly_0123_f64x8);
    results_f64x8 = _mm512_fmadd_pd(results_f64x8, angles_squared_f64x8, coeff_8_f64x8);
    results_f64x8 = _mm512_fmadd_pd(results_f64x8, angles_cubed_f64x8, angles_f64x8);
    return results_f64x8;
}

NK_HELPER_INLINE __m512d nk_atan_f64x8_skylake_(__m512d const inputs) {
    // Polynomial coefficients for atan approximation
    __m512d const coeff_19_f64x8 = _mm512_set1_pd(-1.88796008463073496563746e-05);
    __m512d const coeff_18_f64x8 = _mm512_set1_pd(+0.000209850076645816976906797);
    __m512d const coeff_17_f64x8 = _mm512_set1_pd(-0.00110611831486672482563471);
    __m512d const coeff_16_f64x8 = _mm512_set1_pd(+0.00370026744188713119232403);
    __m512d const coeff_15_f64x8 = _mm512_set1_pd(-0.00889896195887655491740809);
    __m512d const coeff_14_f64x8 = _mm512_set1_pd(+0.016599329773529201970117);
    __m512d const coeff_13_f64x8 = _mm512_set1_pd(-0.0254517624932312641616861);
    __m512d const coeff_12_f64x8 = _mm512_set1_pd(+0.0337852580001353069993897);
    __m512d const coeff_11_f64x8 = _mm512_set1_pd(-0.0407629191276836500001934);
    __m512d const coeff_10_f64x8 = _mm512_set1_pd(+0.0466667150077840625632675);
    __m512d const coeff_9_f64x8 = _mm512_set1_pd(-0.0523674852303482457616113);
    __m512d const coeff_8_f64x8 = _mm512_set1_pd(+0.0587666392926673580854313);
    __m512d const coeff_7_f64x8 = _mm512_set1_pd(-0.0666573579361080525984562);
    __m512d const coeff_6_f64x8 = _mm512_set1_pd(+0.0769219538311769618355029);
    __m512d const coeff_5_f64x8 = _mm512_set1_pd(-0.090908995008245008229153);
    __m512d const coeff_4_f64x8 = _mm512_set1_pd(+0.111111105648261418443745);
    __m512d const coeff_3_f64x8 = _mm512_set1_pd(-0.14285714266771329383765);
    __m512d const coeff_2_f64x8 = _mm512_set1_pd(+0.199999999996591265594148);
    __m512d const coeff_1_f64x8 = _mm512_set1_pd(-0.333333333333311110369124);

    // Quadrant adjustments
    __mmask8 negative_m8 = _mm512_cmp_pd_mask(inputs, _mm512_setzero_pd(), _CMP_LT_OS);
    __m512d values_f64x8 = _mm512_abs_pd(inputs);
    __mmask8 reciprocal_m8 = _mm512_cmp_pd_mask(values_f64x8, _mm512_set1_pd(1.0), _CMP_GT_OS);
    values_f64x8 = _mm512_mask_div_pd(values_f64x8, reciprocal_m8, _mm512_set1_pd(1.0), values_f64x8);
    __m512d const values_squared_f64x8 = _mm512_mul_pd(values_f64x8, values_f64x8);
    __m512d const values_cubed_f64x8 = _mm512_mul_pd(values_f64x8, values_squared_f64x8);

    // Polynomial evaluation (argument reduction and approximation)
    __m512d polynomials_f64x8 = coeff_19_f64x8;
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_18_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_17_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_16_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_15_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_14_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_13_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_12_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_11_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_10_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_9_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_8_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_7_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_6_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_5_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_4_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_3_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_2_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, values_squared_f64x8, coeff_1_f64x8);

    // Compute atan approximation
    __m512d result_f64x8 = _mm512_fmadd_pd(values_cubed_f64x8, polynomials_f64x8, values_f64x8);
    result_f64x8 = _mm512_mask_sub_pd(result_f64x8, reciprocal_m8, _mm512_set1_pd(1.5707963267948966), result_f64x8);
    result_f64x8 = _mm512_mask_sub_pd(result_f64x8, negative_m8, _mm512_setzero_pd(), result_f64x8);
    return result_f64x8;
}

/**
 *  @brief AVX-512 implementation of atan2(y, x) for 8 double-precision values.
 *  @see Based on the f32x16 version with appropriate precision constants.
 */
NK_HELPER_INLINE __m512d nk_atan2_f64x8_skylake_(__m512d const ys_inputs, __m512d const xs_inputs) {
    // Polynomial coefficients for atan approximation (higher precision than f32)
    __m512d const coeff_19_f64x8 = _mm512_set1_pd(-1.88796008463073496563746e-05);
    __m512d const coeff_18_f64x8 = _mm512_set1_pd(+0.000209850076645816976906797);
    __m512d const coeff_17_f64x8 = _mm512_set1_pd(-0.00110611831486672482563471);
    __m512d const coeff_16_f64x8 = _mm512_set1_pd(+0.00370026744188713119232403);
    __m512d const coeff_15_f64x8 = _mm512_set1_pd(-0.00889896195887655491740809);
    __m512d const coeff_14_f64x8 = _mm512_set1_pd(+0.016599329773529201970117);
    __m512d const coeff_13_f64x8 = _mm512_set1_pd(-0.0254517624932312641616861);
    __m512d const coeff_12_f64x8 = _mm512_set1_pd(+0.0337852580001353069993897);
    __m512d const coeff_11_f64x8 = _mm512_set1_pd(-0.0407629191276836500001934);
    __m512d const coeff_10_f64x8 = _mm512_set1_pd(+0.0466667150077840625632675);
    __m512d const coeff_9_f64x8 = _mm512_set1_pd(-0.0523674852303482457616113);
    __m512d const coeff_8_f64x8 = _mm512_set1_pd(+0.0587666392926673580854313);
    __m512d const coeff_7_f64x8 = _mm512_set1_pd(-0.0666573579361080525984562);
    __m512d const coeff_6_f64x8 = _mm512_set1_pd(+0.0769219538311769618355029);
    __m512d const coeff_5_f64x8 = _mm512_set1_pd(-0.090908995008245008229153);
    __m512d const coeff_4_f64x8 = _mm512_set1_pd(+0.111111105648261418443745);
    __m512d const coeff_3_f64x8 = _mm512_set1_pd(-0.14285714266771329383765);
    __m512d const coeff_2_f64x8 = _mm512_set1_pd(+0.199999999996591265594148);
    __m512d const coeff_1_f64x8 = _mm512_set1_pd(-0.333333333333311110369124);

    // Quadrant adjustments normalizing to absolute values of x and y
    __mmask8 const xs_negative_m8 = _mm512_cmp_pd_mask(xs_inputs, _mm512_setzero_pd(), _CMP_LT_OS);
    __m512d xs_f64x8 = _mm512_abs_pd(xs_inputs);
    __m512d ys_f64x8 = _mm512_abs_pd(ys_inputs);
    // Ensure proper fraction where the numerator is smaller than the denominator
    __mmask8 const swap_m8 = _mm512_cmp_pd_mask(ys_f64x8, xs_f64x8, _CMP_GT_OS);
    __m512d temps_f64x8 = xs_f64x8;
    xs_f64x8 = _mm512_mask_blend_pd(swap_m8, xs_f64x8, ys_f64x8);
    ys_f64x8 = _mm512_mask_sub_pd(ys_f64x8, swap_m8, _mm512_setzero_pd(), temps_f64x8);

    // Compute ratio_f64x8 and ratio²
    __m512d const ratio_f64x8 = _mm512_div_pd(ys_f64x8, xs_f64x8);
    __m512d const ratio_squared_f64x8 = _mm512_mul_pd(ratio_f64x8, ratio_f64x8);
    __m512d const ratio_cubed_f64x8 = _mm512_mul_pd(ratio_f64x8, ratio_squared_f64x8);

    // Polynomial evaluation
    __m512d polynomials_f64x8 = coeff_19_f64x8;
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_18_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_17_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_16_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_15_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_14_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_13_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_12_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_11_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_10_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_9_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_8_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_7_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_6_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_5_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_4_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_3_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_2_f64x8);
    polynomials_f64x8 = _mm512_fmadd_pd(polynomials_f64x8, ratio_squared_f64x8, coeff_1_f64x8);

    // Compute the result with quadrant_f64x8 adjustments
    __m512d results_f64x8 = _mm512_fmadd_pd(ratio_cubed_f64x8, polynomials_f64x8, ratio_f64x8);

    // Compute quadrant_f64x8 value: 0 for x>=0 && !swap, 1 for x>=0 && swap,
    //                        -2 for x<0 && !swap, -1 for x<0 && swap
    __m512d quadrant_f64x8 = _mm512_setzero_pd();
    quadrant_f64x8 = _mm512_mask_blend_pd(xs_negative_m8, quadrant_f64x8, _mm512_set1_pd(-2.0));
    __m512d quadrant_incremented_f64x8 = _mm512_add_pd(quadrant_f64x8, _mm512_set1_pd(1.0));
    quadrant_f64x8 = _mm512_mask_blend_pd(swap_m8, quadrant_f64x8, quadrant_incremented_f64x8);

    // Adjust for quadrant_f64x8: result += quadrant_f64x8 * π/2
    results_f64x8 = _mm512_fmadd_pd(quadrant_f64x8, _mm512_set1_pd(1.5707963267948966), results_f64x8);

    // Transfer sign from x (XOR with sign bit of x_input)
    __m512d xs_sign_f64x8 = _mm512_and_pd(xs_inputs, _mm512_set1_pd(-0.0));
    results_f64x8 = _mm512_xor_pd(results_f64x8, xs_sign_f64x8);

    // Transfer sign from y (XOR with sign bit of y_input)
    __m512d ys_sign_f64x8 = _mm512_and_pd(ys_inputs, _mm512_set1_pd(-0.0));
    results_f64x8 = _mm512_xor_pd(results_f64x8, ys_sign_f64x8);

    return results_f64x8;
}

NK_API_COMPTIME void nk_trig_sin_f64_skylake(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d angles_f64x8 = _mm512_loadu_pd(ins + i);
        __m512d results_f64x8 = nk_sin_f64x8_skylake_(angles_f64x8);
        _mm512_storeu_pd(outs + i, results_f64x8);
    }
    if (i < n) {
        __mmask8 mask_m8 = (__mmask8)_bzhi_u32(0xFFFF, n - i);
        __m512d angles_f64x8 = _mm512_maskz_loadu_pd(mask_m8, ins + i);
        __m512d results_f64x8 = nk_sin_f64x8_skylake_(angles_f64x8);
        _mm512_mask_storeu_pd(outs + i, mask_m8, results_f64x8);
    }
}
NK_API_COMPTIME void nk_trig_cos_f64_skylake(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d angles_f64x8 = _mm512_loadu_pd(ins + i);
        __m512d results_f64x8 = nk_cos_f64x8_skylake_(angles_f64x8);
        _mm512_storeu_pd(outs + i, results_f64x8);
    }
    if (i < n) {
        __mmask8 mask_m8 = (__mmask8)_bzhi_u32(0xFFFF, n - i);
        __m512d angles_f64x8 = _mm512_maskz_loadu_pd(mask_m8, ins + i);
        __m512d results_f64x8 = nk_cos_f64x8_skylake_(angles_f64x8);
        _mm512_mask_storeu_pd(outs + i, mask_m8, results_f64x8);
    }
}
NK_API_COMPTIME void nk_trig_atan_f64_skylake(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d angles_f64x8 = _mm512_loadu_pd(ins + i);
        __m512d results_f64x8 = nk_atan_f64x8_skylake_(angles_f64x8);
        _mm512_storeu_pd(outs + i, results_f64x8);
    }
    if (i < n) {
        __mmask8 mask_m8 = (__mmask8)_bzhi_u32(0xFFFF, n - i);
        __m512d angles_f64x8 = _mm512_maskz_loadu_pd(mask_m8, ins + i);
        __m512d results_f64x8 = nk_atan_f64x8_skylake_(angles_f64x8);
        _mm512_mask_storeu_pd(outs + i, mask_m8, results_f64x8);
    }
}

/**
 *  @brief Sine approximation for 16 f16 values via f32 upcasting.
 *
 *  Degree-5 polynomial with Cody-Waite range reduction in f32.
 *  Takes __m256i (f16 data), returns __m256i (f16 result).
 */
NK_HELPER_INLINE __m256i nk_sin_f16x16_skylake_(__m256i angles_f16x16) {
    __m512 angles_f32x16 = _mm512_cvtph_ps(angles_f16x16);
    // Cody-Waite range reduction constants
    __m512 pi_high_f32x16 = _mm512_set1_ps(3.1415927f);
    __m512 pi_low_f32x16 = _mm512_set1_ps(-8.742278e-8f);
    __m512 pi_recip_f32x16 = _mm512_set1_ps(0.31830988618f);
    __m512 c3_f32x16 = _mm512_set1_ps(-1.6666666641e-1f);
    __m512 c5_f32x16 = _mm512_set1_ps(8.3333293855e-3f);

    __m512 quotient_f32x16 = _mm512_mul_ps(angles_f32x16, pi_recip_f32x16);
    __m512 rounded_f32x16 = _mm512_roundscale_ps(quotient_f32x16, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // Use explicit rounding to match roundscale (MXCSR-independent)
    __m512i multiple_i32x16 = _mm512_cvt_roundps_epi32(rounded_f32x16, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    angles_f32x16 = _mm512_fnmadd_ps(rounded_f32x16, pi_high_f32x16, angles_f32x16);
    angles_f32x16 = _mm512_fnmadd_ps(rounded_f32x16, pi_low_f32x16, angles_f32x16);

    __m512 x2_f32x16 = _mm512_mul_ps(angles_f32x16, angles_f32x16);
    __m512 poly_f32x16 = _mm512_fmadd_ps(c5_f32x16, x2_f32x16, c3_f32x16);
    poly_f32x16 = _mm512_mul_ps(poly_f32x16, x2_f32x16);
    __m512 result_f32x16 = _mm512_fmadd_ps(poly_f32x16, angles_f32x16, angles_f32x16);

    __mmask16 odd_m16 = _mm512_test_epi32_mask(multiple_i32x16, _mm512_set1_epi32(1));
    result_f32x16 = _mm512_mask_sub_ps(result_f32x16, odd_m16, _mm512_setzero_ps(), result_f32x16);
    return _mm512_cvtps_ph(result_f32x16, _MM_FROUND_TO_NEAREST_INT);
}

/**
 *  @brief Cosine approximation for 16 f16 values via f32 upcasting.
 *
 *  Uses cos(x) = sin(x + pi/2) with Cody-Waite range reduction in f32.
 */
NK_HELPER_INLINE __m256i nk_cos_f16x16_skylake_(__m256i angles_f16x16) {
    __m512 angles_f32x16 = _mm512_cvtph_ps(angles_f16x16);
    __m512 pi_high_f32x16 = _mm512_set1_ps(3.1415927f);
    __m512 pi_low_f32x16 = _mm512_set1_ps(-8.742278e-8f);
    __m512 pi_half_f32x16 = _mm512_set1_ps(1.5707963268f);
    __m512 pi_recip_f32x16 = _mm512_set1_ps(0.31830988618f);
    __m512 half_f32x16 = _mm512_set1_ps(0.5f);
    __m512 c3_f32x16 = _mm512_set1_ps(-1.6666666641e-1f);
    __m512 c5_f32x16 = _mm512_set1_ps(8.3333293855e-3f);

    __m512 quotient_f32x16 = _mm512_fmsub_ps(angles_f32x16, pi_recip_f32x16, half_f32x16);
    __m512 rounded_f32x16 = _mm512_roundscale_ps(quotient_f32x16, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // Use explicit rounding to match roundscale (MXCSR-independent)
    __m512i multiple_i32x16 = _mm512_cvt_roundps_epi32(rounded_f32x16, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m512 shift_f32x16 = _mm512_fmadd_ps(rounded_f32x16, pi_high_f32x16, pi_half_f32x16);
    angles_f32x16 = _mm512_sub_ps(angles_f32x16, shift_f32x16);
    angles_f32x16 = _mm512_fnmadd_ps(rounded_f32x16, pi_low_f32x16, angles_f32x16);

    __m512 x2_f32x16 = _mm512_mul_ps(angles_f32x16, angles_f32x16);
    __m512 poly_f32x16 = _mm512_fmadd_ps(c5_f32x16, x2_f32x16, c3_f32x16);
    poly_f32x16 = _mm512_mul_ps(poly_f32x16, x2_f32x16);
    __m512 result_f32x16 = _mm512_fmadd_ps(poly_f32x16, angles_f32x16, angles_f32x16);

    __mmask16 even_m16 = _mm512_testn_epi32_mask(multiple_i32x16, _mm512_set1_epi32(1));
    result_f32x16 = _mm512_mask_sub_ps(result_f32x16, even_m16, _mm512_setzero_ps(), result_f32x16);
    return _mm512_cvtps_ph(result_f32x16, _MM_FROUND_TO_NEAREST_INT);
}

/**
 *  @brief Arctangent approximation for 16 f16 values via f32 upcasting.
 *
 *  Degree-9 polynomial in f32 with quadrant adjustments.
 */
NK_HELPER_INLINE __m256i nk_atan_f16x16_skylake_(__m256i values_f16x16) {
    __m512 values_f32x16 = _mm512_cvtph_ps(values_f16x16);
    __m512 c3_f32x16 = _mm512_set1_ps(-0.3333333333f);
    __m512 c5_f32x16 = _mm512_set1_ps(0.2f);
    __m512 c7_f32x16 = _mm512_set1_ps(-0.1428571429f);
    __m512 c9_f32x16 = _mm512_set1_ps(0.1111111111f);
    __m512 pi_half_f32x16 = _mm512_set1_ps(1.5707963268f);
    __m512 one_f32x16 = _mm512_set1_ps(1.0f);

    __mmask16 negative_m16 = _mm512_cmp_ps_mask(values_f32x16, _mm512_setzero_ps(), _CMP_LT_OS);
    values_f32x16 = _mm512_abs_ps(values_f32x16);
    __mmask16 reciprocal_m16 = _mm512_cmp_ps_mask(values_f32x16, one_f32x16, _CMP_GT_OS);
    values_f32x16 = _mm512_mask_div_ps(values_f32x16, reciprocal_m16, one_f32x16, values_f32x16);

    __m512 x2_f32x16 = _mm512_mul_ps(values_f32x16, values_f32x16);
    __m512 x3_f32x16 = _mm512_mul_ps(values_f32x16, x2_f32x16);

    __m512 poly_f32x16 = c9_f32x16;
    poly_f32x16 = _mm512_fmadd_ps(poly_f32x16, x2_f32x16, c7_f32x16);
    poly_f32x16 = _mm512_fmadd_ps(poly_f32x16, x2_f32x16, c5_f32x16);
    poly_f32x16 = _mm512_fmadd_ps(poly_f32x16, x2_f32x16, c3_f32x16);

    __m512 result_f32x16 = _mm512_fmadd_ps(x3_f32x16, poly_f32x16, values_f32x16);
    result_f32x16 = _mm512_mask_sub_ps(result_f32x16, reciprocal_m16, pi_half_f32x16, result_f32x16);
    result_f32x16 = _mm512_mask_sub_ps(result_f32x16, negative_m16, _mm512_setzero_ps(), result_f32x16);
    return _mm512_cvtps_ph(result_f32x16, _MM_FROUND_TO_NEAREST_INT);
}

NK_API_COMPTIME void nk_trig_sin_f16_skylake(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i angles_f16x16 = _mm256_loadu_si256((__m256i const *)(ins + i));
        __m256i result_f16x16 = nk_sin_f16x16_skylake_(angles_f16x16);
        _mm256_storeu_si256((__m256i *)(outs + i), result_f16x16);
    }
    if (i < n) {
        __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m256i angles_f16x16 = _mm256_maskz_loadu_epi16(mask_m16, ins + i);
        __m256i result_f16x16 = nk_sin_f16x16_skylake_(angles_f16x16);
        _mm256_mask_storeu_epi16(outs + i, mask_m16, result_f16x16);
    }
}

NK_API_COMPTIME void nk_trig_cos_f16_skylake(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i angles_f16x16 = _mm256_loadu_si256((__m256i const *)(ins + i));
        __m256i result_f16x16 = nk_cos_f16x16_skylake_(angles_f16x16);
        _mm256_storeu_si256((__m256i *)(outs + i), result_f16x16);
    }
    if (i < n) {
        __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m256i angles_f16x16 = _mm256_maskz_loadu_epi16(mask_m16, ins + i);
        __m256i result_f16x16 = nk_cos_f16x16_skylake_(angles_f16x16);
        _mm256_mask_storeu_epi16(outs + i, mask_m16, result_f16x16);
    }
}

NK_API_COMPTIME void nk_trig_atan_f16_skylake(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i values_f16x16 = _mm256_loadu_si256((__m256i const *)(ins + i));
        __m256i result_f16x16 = nk_atan_f16x16_skylake_(values_f16x16);
        _mm256_storeu_si256((__m256i *)(outs + i), result_f16x16);
    }
    if (i < n) {
        __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, n - i);
        __m256i values_f16x16 = _mm256_maskz_loadu_epi16(mask_m16, ins + i);
        __m256i result_f16x16 = nk_atan_f16x16_skylake_(values_f16x16);
        _mm256_mask_storeu_epi16(outs + i, mask_m16, result_f16x16);
    }
}

NK_API_COMPTIME void nk_trig_rope_f32_skylake(nk_f32_t const *x, nk_f32_t *y, nk_rope_angle_t const *cos,
                                              nk_rope_angle_t const *sin, nk_size_t rows, nk_size_t heads,
                                              nk_size_t half_dim, nk_size_t x_row_stride, nk_size_t y_row_stride,
                                              nk_f32_t input_scale) {
    __m512 scale_f32x16 = _mm512_set1_ps(input_scale);
    for (nk_size_t r = 0; r != rows; ++r) {
        nk_f32_t const *cos_row = cos + r * half_dim;
        nk_f32_t const *sin_row = sin + r * half_dim;
        nk_f32_t const *x_row = (nk_f32_t const *)((unsigned char const *)x + r * x_row_stride);
        nk_f32_t *y_row = (nk_f32_t *)((unsigned char *)y + r * y_row_stride);
        for (nk_size_t h = 0; h != heads; ++h) {
            nk_f32_t const *x_base = x_row + h * 2 * half_dim;
            nk_f32_t *y_base = y_row + h * 2 * half_dim;
            nk_size_t i = 0;
            for (; i + 16 <= half_dim; i += 16) {
                __m512 low_f32x16 = _mm512_mul_ps(_mm512_loadu_ps(x_base + i), scale_f32x16);
                __m512 high_f32x16 = _mm512_mul_ps(_mm512_loadu_ps(x_base + i + half_dim), scale_f32x16);
                __m512 cos_f32x16 = _mm512_loadu_ps(cos_row + i), sin_f32x16 = _mm512_loadu_ps(sin_row + i);
                _mm512_storeu_ps(y_base + i,
                                 _mm512_fmsub_ps(low_f32x16, cos_f32x16, _mm512_mul_ps(high_f32x16, sin_f32x16)));
                _mm512_storeu_ps(y_base + i + half_dim,
                                 _mm512_fmadd_ps(low_f32x16, sin_f32x16, _mm512_mul_ps(high_f32x16, cos_f32x16)));
            }
            if (i < half_dim) {
                __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFFu, (unsigned)(half_dim - i));
                __m512 low_f32x16 = _mm512_mul_ps(_mm512_maskz_loadu_ps(mask_m16, x_base + i), scale_f32x16);
                __m512 high_f32x16 = _mm512_mul_ps(_mm512_maskz_loadu_ps(mask_m16, x_base + i + half_dim),
                                                   scale_f32x16);
                __m512 cos_f32x16 = _mm512_maskz_loadu_ps(mask_m16, cos_row + i);
                __m512 sin_f32x16 = _mm512_maskz_loadu_ps(mask_m16, sin_row + i);
                _mm512_mask_storeu_ps(y_base + i, mask_m16,
                                      _mm512_fmsub_ps(low_f32x16, cos_f32x16, _mm512_mul_ps(high_f32x16, sin_f32x16)));
                _mm512_mask_storeu_ps(y_base + i + half_dim, mask_m16,
                                      _mm512_fmadd_ps(low_f32x16, sin_f32x16, _mm512_mul_ps(high_f32x16, cos_f32x16)));
            }
        }
    }
}

NK_API_COMPTIME void nk_trig_rope_bf16_skylake(nk_bf16_t const *x, nk_bf16_t *y, nk_rope_angle_t const *cos,
                                               nk_rope_angle_t const *sin, nk_size_t rows, nk_size_t heads,
                                               nk_size_t half_dim, nk_size_t x_row_stride, nk_size_t y_row_stride,
                                               nk_f32_t input_scale) {
    __m512 scale_f32x16 = _mm512_set1_ps(input_scale);
    for (nk_size_t r = 0; r != rows; ++r) {
        nk_f32_t const *cos_row = cos + r * half_dim;
        nk_f32_t const *sin_row = sin + r * half_dim;
        nk_bf16_t const *x_row = (nk_bf16_t const *)((unsigned char const *)x + r * x_row_stride);
        nk_bf16_t *y_row = (nk_bf16_t *)((unsigned char *)y + r * y_row_stride);
        for (nk_size_t h = 0; h != heads; ++h) {
            nk_bf16_t const *x_base = x_row + h * 2 * half_dim;
            nk_bf16_t *y_base = y_row + h * 2 * half_dim;
            nk_size_t i = 0;
            for (; i + 16 <= half_dim; i += 16) {
                nk_b512_vec_t low_vec, high_vec;
                nk_load_bf16x16_to_f32x16_skylake_(x_base + i, &low_vec);
                nk_load_bf16x16_to_f32x16_skylake_(x_base + i + half_dim, &high_vec);
                __m512 low_f32x16 = _mm512_mul_ps(low_vec.zmm_ps, scale_f32x16);
                __m512 high_f32x16 = _mm512_mul_ps(high_vec.zmm_ps, scale_f32x16);
                __m512 cos_f32x16 = _mm512_loadu_ps(cos_row + i), sin_f32x16 = _mm512_loadu_ps(sin_row + i);
                _mm256_storeu_si256((__m256i *)(y_base + i),
                                    nk_f32x16_to_bf16x16_skylake_(_mm512_fmsub_ps(
                                        low_f32x16, cos_f32x16, _mm512_mul_ps(high_f32x16, sin_f32x16))));
                _mm256_storeu_si256((__m256i *)(y_base + i + half_dim),
                                    nk_f32x16_to_bf16x16_skylake_(_mm512_fmadd_ps(
                                        low_f32x16, sin_f32x16, _mm512_mul_ps(high_f32x16, cos_f32x16))));
            }
            if (i < half_dim) {
                nk_size_t remaining = half_dim - i;
                __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFFu, (unsigned)remaining);
                nk_b512_vec_t low_vec, high_vec;
                nk_partial_load_bf16x16_to_f32x16_skylake_(x_base + i, &low_vec, remaining);
                nk_partial_load_bf16x16_to_f32x16_skylake_(x_base + i + half_dim, &high_vec, remaining);
                __m512 low_f32x16 = _mm512_mul_ps(low_vec.zmm_ps, scale_f32x16);
                __m512 high_f32x16 = _mm512_mul_ps(high_vec.zmm_ps, scale_f32x16);
                __m512 cos_f32x16 = _mm512_maskz_loadu_ps(mask_m16, cos_row + i);
                __m512 sin_f32x16 = _mm512_maskz_loadu_ps(mask_m16, sin_row + i);
                _mm256_mask_storeu_epi16((void *)(y_base + i), mask_m16,
                                         nk_f32x16_to_bf16x16_skylake_(_mm512_fmsub_ps(
                                             low_f32x16, cos_f32x16, _mm512_mul_ps(high_f32x16, sin_f32x16))));
                _mm256_mask_storeu_epi16((void *)(y_base + i + half_dim), mask_m16,
                                         nk_f32x16_to_bf16x16_skylake_(_mm512_fmadd_ps(
                                             low_f32x16, sin_f32x16, _mm512_mul_ps(high_f32x16, cos_f32x16))));
            }
        }
    }
}

NK_API_COMPTIME void nk_trig_rope_e4m3_skylake(nk_e4m3_t const *x, nk_e4m3_t *y, nk_rope_angle_t const *cos,
                                               nk_rope_angle_t const *sin, nk_size_t rows, nk_size_t heads,
                                               nk_size_t half_dim, nk_size_t x_row_stride, nk_size_t y_row_stride,
                                               nk_f32_t input_scale) {
    __m512 scale_f32x16 = _mm512_set1_ps(input_scale);
    for (nk_size_t r = 0; r != rows; ++r) {
        nk_f32_t const *cos_row = cos + r * half_dim;
        nk_f32_t const *sin_row = sin + r * half_dim;
        nk_e4m3_t const *x_row = (nk_e4m3_t const *)((unsigned char const *)x + r * x_row_stride);
        nk_e4m3_t *y_row = (nk_e4m3_t *)((unsigned char *)y + r * y_row_stride);
        for (nk_size_t h = 0; h != heads; ++h) {
            nk_e4m3_t const *x_base = x_row + h * 2 * half_dim;
            nk_e4m3_t *y_base = y_row + h * 2 * half_dim;
            nk_size_t i = 0;
            for (; i + 16 <= half_dim; i += 16) {
                nk_b512_vec_t low_vec, high_vec;
                nk_load_e4m3x16_to_f32x16_skylake_(x_base + i, &low_vec);
                nk_load_e4m3x16_to_f32x16_skylake_(x_base + i + half_dim, &high_vec);
                __m512 low_f32x16 = _mm512_mul_ps(low_vec.zmm_ps, scale_f32x16);
                __m512 high_f32x16 = _mm512_mul_ps(high_vec.zmm_ps, scale_f32x16);
                __m512 cos_f32x16 = _mm512_loadu_ps(cos_row + i), sin_f32x16 = _mm512_loadu_ps(sin_row + i);
                _mm_storeu_si128((__m128i *)(y_base + i),
                                 nk_f32x16_to_e4m3x16_skylake_(
                                     _mm512_fmsub_ps(low_f32x16, cos_f32x16, _mm512_mul_ps(high_f32x16, sin_f32x16))));
                _mm_storeu_si128((__m128i *)(y_base + i + half_dim),
                                 nk_f32x16_to_e4m3x16_skylake_(
                                     _mm512_fmadd_ps(low_f32x16, sin_f32x16, _mm512_mul_ps(high_f32x16, cos_f32x16))));
            }
            if (i < half_dim) {
                nk_size_t remaining = half_dim - i;
                __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFFu, (unsigned)remaining);
                nk_b512_vec_t low_vec, high_vec;
                nk_partial_load_e4m3x16_to_f32x16_skylake_(x_base + i, &low_vec, remaining);
                nk_partial_load_e4m3x16_to_f32x16_skylake_(x_base + i + half_dim, &high_vec, remaining);
                __m512 low_f32x16 = _mm512_mul_ps(low_vec.zmm_ps, scale_f32x16);
                __m512 high_f32x16 = _mm512_mul_ps(high_vec.zmm_ps, scale_f32x16);
                __m512 cos_f32x16 = _mm512_maskz_loadu_ps(mask_m16, cos_row + i);
                __m512 sin_f32x16 = _mm512_maskz_loadu_ps(mask_m16, sin_row + i);
                _mm_mask_storeu_epi8((void *)(y_base + i), mask_m16,
                                     nk_f32x16_to_e4m3x16_skylake_(_mm512_fmsub_ps(
                                         low_f32x16, cos_f32x16, _mm512_mul_ps(high_f32x16, sin_f32x16))));
                _mm_mask_storeu_epi8((void *)(y_base + i + half_dim), mask_m16,
                                     nk_f32x16_to_e4m3x16_skylake_(_mm512_fmadd_ps(
                                         low_f32x16, sin_f32x16, _mm512_mul_ps(high_f32x16, cos_f32x16))));
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

#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X8664_
#endif // NK_TRIGONOMETRY_SKYLAKE_H
