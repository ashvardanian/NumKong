/**
 *  @brief SWAR-accelerated Trigonometric Functions for SIMD-free CPUs.
 *  @file include/numkong/trigonometry/serial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/trigonometry.h
 *  @see https://sleef.org
 */
#ifndef NK_TRIGONOMETRY_SERIAL_H
#define NK_TRIGONOMETRY_SERIAL_H

#include "numkong/types.h"
#include "numkong/cast/serial.h"   // `nk_f16_to_f32_serial`
#include "numkong/scalar/serial.h" // `nk_f32_fma_serial`, `nk_f64_fma_serial`

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Computes an approximate sine of the given angle in radians with @b 3-ULP error bound for [-2π, 2π].
 *  @see Based on @b `xfastsinf_u3500` in SLEEF library.
 *  @param[in] angle The input angle in radians.
 *  @return The approximate sine of the input angle in [-1, 1] range.
 */
NK_PUBLIC nk_f32_t nk_f32_sin(nk_f32_t const angle_radians) {

    // Cody-Waite constants for argument reduction (pi split into hi + lo)
    nk_f32_t const pi_hi = 3.1415927f;
    nk_f32_t const pi_lo = -8.742278e-8f;
    nk_f32_t const pi_reciprocal = 0.31830988618379067154f; /// 1/π

    // Degree-9 minimax coefficients: sin(x) ≈ x + c3*x³ + c5*x⁵ + c7*x⁷ + c9*x⁹
    nk_f32_t const coeff_9 = +2.7557319224e-6f;
    nk_f32_t const coeff_7 = -1.9841269841e-4f;
    nk_f32_t const coeff_5 = +8.3333293855e-3f;
    nk_f32_t const coeff_3 = -1.6666666641e-1f;

    // Compute (multiple_of_pi) = round(angle / π)
    nk_f32_t const quotient = angle_radians * pi_reciprocal;
    int const multiple_of_pi = (int)(quotient < 0 ? quotient - 0.5f : quotient + 0.5f);

    // Cody-Waite range reduction: angle = angle_radians - multiple * (pi_hi + pi_lo)
    nk_f32_t angle = angle_radians - multiple_of_pi * pi_hi;
    angle -= multiple_of_pi * pi_lo;
    nk_f32_t const angle_squared = angle * angle;
    nk_f32_t const angle_cubed = angle * angle_squared;

    // Degree-9 polynomial via Horner's method
    nk_f32_t polynomial = coeff_9;
    polynomial = polynomial * angle_squared + coeff_7;
    polynomial = polynomial * angle_squared + coeff_5;
    polynomial = polynomial * angle_squared + coeff_3;
    nk_f32_t result = polynomial * angle_cubed + angle;

    // If multiple_of_pi is odd, flip the sign of the result
    if ((multiple_of_pi & 1) != 0) result = -result;
    return result;
}

/**
 *  @brief Computes an approximate cosine of the given angle in radians with @b 3-ULP error bound for [-2π, 2π].
 *  @see Based on @b `xfastcosf_u3500` in SLEEF library.
 *  @param[in] angle The input angle in radians.
 *  @return The approximate cosine of the input angle in [-1, 1] range.
 */
NK_PUBLIC nk_f32_t nk_f32_cos(nk_f32_t const angle_radians) {

    // Cody-Waite constants for argument reduction (pi split into hi + lo)
    nk_f32_t const pi_hi = 3.1415927f;
    nk_f32_t const pi_lo = -8.742278e-8f;
    nk_f32_t const pi_half = 1.57079632679489661923f;       /// π/2
    nk_f32_t const pi_reciprocal = 0.31830988618379067154f; /// 1/π

    // Degree-9 minimax coefficients: sin(x) ≈ x + c3*x³ + c5*x⁵ + c7*x⁷ + c9*x⁹
    nk_f32_t const coeff_9 = +2.7557319224e-6f;
    nk_f32_t const coeff_7 = -1.9841269841e-4f;
    nk_f32_t const coeff_5 = +8.3333293855e-3f;
    nk_f32_t const coeff_3 = -1.6666666641e-1f;

    // Compute (multiple_of_pi) = round(angle / π - 0.5)
    nk_f32_t const quotient = angle_radians * pi_reciprocal - 0.5f;
    int const multiple_of_pi = (int)(quotient < 0 ? quotient - 0.5f : quotient + 0.5f);

    // Cody-Waite range reduction: angle = angle_radians - (multiple * pi + pi/2)
    nk_f32_t const offset = pi_half + multiple_of_pi * pi_hi;
    nk_f32_t angle = angle_radians - offset;
    angle -= multiple_of_pi * pi_lo;
    nk_f32_t const angle_squared = angle * angle;
    nk_f32_t const angle_cubed = angle * angle_squared;

    // Degree-9 polynomial via Horner's method
    nk_f32_t polynomial = coeff_9;
    polynomial = polynomial * angle_squared + coeff_7;
    polynomial = polynomial * angle_squared + coeff_5;
    polynomial = polynomial * angle_squared + coeff_3;
    nk_f32_t result = polynomial * angle_cubed + angle;

    // If multiple_of_pi is even, flip the sign of the result
    if ((multiple_of_pi & 1) == 0) result = -result;
    return result;
}

/**
 *  @brief Computes the arc-tangent of a value with @b 0-ULP error bound.
 *  @see Based on @b `xatanf` in SLEEF library.
 *  @param  input The input value.
 *  @return The arc-tangent of the input value in [-π/2, π/2] radians range.
 */
NK_PUBLIC nk_f32_t nk_f32_atan(nk_f32_t const input) {
    // Polynomial coefficients for atan approximation
    nk_f32_t const coeff_8 = -0.333331018686294555664062f;
    nk_f32_t const coeff_7 = +0.199926957488059997558594f;
    nk_f32_t const coeff_6 = -0.142027363181114196777344f;
    nk_f32_t const coeff_5 = +0.106347933411598205566406f;
    nk_f32_t const coeff_4 = -0.0748900920152664184570312f;
    nk_f32_t const coeff_3 = +0.0425049886107444763183594f;
    nk_f32_t const coeff_2 = -0.0159569028764963150024414f;
    nk_f32_t const coeff_1 = +0.00282363896258175373077393f;

    // Quadrant adjustment
    int quadrant = 0;
    nk_f32_t value = input;
    if (value < 0.0f) value = -value, quadrant |= 2;
    if (value > 1.0f) value = 1.0f / value, quadrant |= 1;

    // Argument reduction
    nk_f32_t const value_squared = value * value;
    nk_f32_t const value_cubed = value * value_squared;

    // Polynomial evaluation using FMA for improved precision
    nk_f32_t polynomial = coeff_1;
    polynomial = nk_f32_fma_serial(polynomial, value_squared, coeff_2);
    polynomial = nk_f32_fma_serial(polynomial, value_squared, coeff_3);
    polynomial = nk_f32_fma_serial(polynomial, value_squared, coeff_4);
    polynomial = nk_f32_fma_serial(polynomial, value_squared, coeff_5);
    polynomial = nk_f32_fma_serial(polynomial, value_squared, coeff_6);
    polynomial = nk_f32_fma_serial(polynomial, value_squared, coeff_7);
    polynomial = nk_f32_fma_serial(polynomial, value_squared, coeff_8);

    // Adjust for quadrant
    nk_f32_t result = nk_f32_fma_serial(polynomial, value_cubed, value);
    nk_f32_t const pi_half = 1.5707963267948966f; // π/2
    if ((quadrant & 1) != 0) result = pi_half - result;
    if ((quadrant & 2) != 0) result = -result;
    return result;
}

typedef enum nk_float_class_t {
    nk_float_unknown_k = 0,
    nk_float_nan_k = 1 << 1,

    nk_float_positive_zero_k = 1 << 10,
    nk_float_positive_finite_k = 1 << 11,
    nk_float_positive_infinity_k = 1 << 12,

    nk_float_negative_zero_k = 1 << 20,
    nk_float_negative_finite_k = 1 << 21,
    nk_float_negative_infinity_k = 1 << 22,

} nk_float_class_t;

NK_PUBLIC nk_float_class_t nk_f32_classify(nk_f32_t const input) {
    // Constants for special cases
    nk_u32_t const positive_zero = 0x00000000u;     // +0
    nk_u32_t const negative_zero = 0x80000000u;     // -0
    nk_u32_t const positive_infinity = 0x7F800000u; // +∞
    nk_u32_t const negative_infinity = 0xFF800000u; // -∞
    nk_u32_t const exponent_mask = 0x7F800000u;     // Mask for exponent bits
    nk_u32_t const mantissa_mask = 0x007FFFFFu;     // Mask for mantissa bits

    nk_fui32_t bits;
    bits.f = input;
    if (bits.u == positive_zero) return nk_float_positive_zero_k;
    if (bits.u == negative_zero) return nk_float_negative_zero_k;
    if (bits.u == positive_infinity) return nk_float_positive_infinity_k;
    if (bits.u == negative_infinity) return nk_float_negative_infinity_k;

    // Check for NaN (exponent all 1s and non-zero mantissa)
    if ((bits.u & exponent_mask) == exponent_mask && (bits.u & mantissa_mask) != 0) return nk_float_nan_k;
    return input > 0.0f ? nk_float_positive_finite_k : nk_float_negative_finite_k;
}

NK_PUBLIC int nk_float_class_belongs_to(nk_float_class_t const class_, int const belongs_to) {
    return (class_ & belongs_to) != 0;
}

/**
 *  @brief Computes the arc-tangent of (y/x) with @b 0-ULP error bound.
 *  @see Based on @b `xatan2f` in SLEEF library.
 *  @param  y_input The input sine value.
 *  @param  x_input The input cosine value.
 *  @return The arc-tangent of (y_input/x_input) in [-π, π] radians range.
 */
NK_PUBLIC nk_f32_t nk_f32_atan2(nk_f32_t const y_input, nk_f32_t const x_input) {

    // Polynomial coefficients for atan2 approximation
    nk_f32_t const coeff_8 = -0.333331018686294555664062f;
    nk_f32_t const coeff_7 = +0.199926957488059997558594f;
    nk_f32_t const coeff_6 = -0.142027363181114196777344f;
    nk_f32_t const coeff_5 = +0.106347933411598205566406f;
    nk_f32_t const coeff_4 = -0.0748900920152664184570312f;
    nk_f32_t const coeff_3 = +0.0425049886107444763183594f;
    nk_f32_t const coeff_2 = -0.0159569028764963150024414f;
    nk_f32_t const coeff_1 = +0.00282363896258175373077393f;

    // Convert to bit representation
    nk_fui32_t const x_bits = *(nk_fui32_t *)&x_input;
    nk_fui32_t const y_bits = *(nk_fui32_t *)&y_input;
    nk_fui32_t x_abs, y_abs;
    y_abs.u = y_bits.u & 0x7FFFFFFFu;

    // Quadrant adjustment
    int quadrant = 0;
    if (x_input < 0.0f) { x_abs.f = -x_input, quadrant = -2; }
    else { x_abs.f = x_input; }
    // Ensure proper fraction where the numerator is smaller than the denominator
    if (y_abs.f > x_abs.f) {
        nk_f32_t temp = x_abs.f;
        x_abs.f = y_abs.f;
        y_abs.f = -temp;
        quadrant += 1;
    }

    // Argument reduction
    nk_f32_t const ratio = y_abs.f / x_abs.f;
    nk_f32_t const ratio_squared = ratio * ratio;
    nk_f32_t const ratio_cubed = ratio * ratio_squared;

    // Polynomial evaluation using FMA for improved precision
    nk_f32_t polynomial = coeff_1;
    polynomial = nk_f32_fma_serial(polynomial, ratio_squared, coeff_2);
    polynomial = nk_f32_fma_serial(polynomial, ratio_squared, coeff_3);
    polynomial = nk_f32_fma_serial(polynomial, ratio_squared, coeff_4);
    polynomial = nk_f32_fma_serial(polynomial, ratio_squared, coeff_5);
    polynomial = nk_f32_fma_serial(polynomial, ratio_squared, coeff_6);
    polynomial = nk_f32_fma_serial(polynomial, ratio_squared, coeff_7);
    polynomial = nk_f32_fma_serial(polynomial, ratio_squared, coeff_8);

    // Compute the result using FMA
    nk_f32_t const pi_half = 1.5707963267948966f; // π/2
    nk_f32_t result = nk_f32_fma_serial(polynomial, ratio_cubed, ratio);
    result = nk_f32_fma_serial((nk_f32_t)quadrant, pi_half, result); // quadrant * (π/2)

    // Adjust sign
    nk_i32_t const negative_zero = 0x80000000;
    nk_fui32_t result_bits;
    result_bits.f = result;
    result_bits.u ^= x_bits.u & negative_zero;
    result_bits.u ^= y_bits.u & negative_zero;
    return result_bits.f;
}

/**
 *  @brief Computes the sine of the given angle in radians with @b 0-ULP error bound in [-2π, 2π].
 *  @see Based on @b `xsin` in SLEEF library.
 *  @param[in] angle The input angle in radians.
 *  @return The approximate cosine of the input angle.
 */
NK_PUBLIC nk_f64_t nk_f64_sin(nk_f64_t const angle_radians) {

    // Constants for argument reduction
    nk_f64_t const pi_high = 3.141592653589793116;                         // High-digits part of π
    nk_f64_t const pi_low = 1.2246467991473532072e-16;                     // Low-digits part of π
    nk_f64_t const pi_reciprocal = 0.318309886183790671537767526745028724; // 1/π
    nk_i64_t const negative_zero = 0x8000000000000000LL;                   // Hexadecimal value of -0.0 in IEEE 754

    // Polynomial coefficients for sine/cosine approximation (minimax polynomial)
    nk_f64_t const coeff_0 = +0.00833333333333332974823815;
    nk_f64_t const coeff_1 = -0.000198412698412696162806809;
    nk_f64_t const coeff_2 = +2.75573192239198747630416e-06;
    nk_f64_t const coeff_3 = -2.50521083763502045810755e-08;
    nk_f64_t const coeff_4 = +1.60590430605664501629054e-10;
    nk_f64_t const coeff_5 = -7.64712219118158833288484e-13;
    nk_f64_t const coeff_6 = +2.81009972710863200091251e-15;
    nk_f64_t const coeff_7 = -7.97255955009037868891952e-18;
    nk_f64_t const coeff_8 = -0.166666666666666657414808;

    // Compute (multiple_of_pi) = round(angle / π)
    nk_f64_t const quotient = angle_radians * pi_reciprocal;
    int const multiple_of_pi = (int)(quotient < 0 ? quotient - 0.5 : quotient + 0.5);

    // Reduce the angle to: (angle - (multiple_of_pi * π)) ∈ [0, π]
    nk_f64_t angle = angle_radians;
    angle = angle - (multiple_of_pi * pi_high);
    angle = angle - (multiple_of_pi * pi_low);
    if ((multiple_of_pi & 1) != 0) angle = -angle;
    nk_f64_t const angle_squared = angle * angle;
    nk_f64_t const angle_cubed = angle * angle_squared;
    nk_f64_t const angle_quartic = angle_squared * angle_squared;
    nk_f64_t const angle_octic = angle_quartic * angle_quartic;

    // Compute higher-degree polynomial terms using FMA
    nk_f64_t const poly_67 = nk_f64_fma_serial(angle_squared, coeff_7, coeff_6);
    nk_f64_t const poly_45 = nk_f64_fma_serial(angle_squared, coeff_5, coeff_4);
    nk_f64_t const poly_4567 = nk_f64_fma_serial(angle_quartic, poly_67, poly_45);

    // Compute lower-degree polynomial terms using FMA
    nk_f64_t const poly_23 = nk_f64_fma_serial(angle_squared, coeff_3, coeff_2);
    nk_f64_t const poly_01 = nk_f64_fma_serial(angle_squared, coeff_1, coeff_0);
    nk_f64_t const poly_0123 = nk_f64_fma_serial(angle_quartic, poly_23, poly_01);

    // Combine polynomial terms using FMA
    nk_f64_t result = nk_f64_fma_serial(angle_octic, poly_4567, poly_0123);
    result = nk_f64_fma_serial(result, angle_squared, coeff_8);
    result = nk_f64_fma_serial(result, angle_cubed, angle);

    // Handle the special case of negative zero input
    nk_fui64_t converter;
    converter.f = angle_radians;
    if ((nk_i64_t)converter.u == negative_zero) result = angle;
    return result;
}

/**
 *  @brief Computes the cosine of the given angle in radians with @b 0-ULP error bound in [-2π, 2π].
 *  @see Based on @b `xcos` in SLEEF library.
 *  @param[in] angle The input angle in radians.
 *  @return The approximate cosine of the input angle in [-1, 1] range.
 */
NK_PUBLIC nk_f64_t nk_f64_cos(nk_f64_t const angle_radians) {

    // Constants for argument reduction
    nk_f64_t const pi_high_half = 3.141592653589793116 * 0.5;              // High-digits part of π
    nk_f64_t const pi_low_half = 1.2246467991473532072e-16 * 0.5;          // Low-digits part of π
    nk_f64_t const pi_reciprocal = 0.318309886183790671537767526745028724; // 1/π

    // Polynomial coefficients for sine/cosine approximation (minimax polynomial)
    nk_f64_t const coeff_0 = +0.00833333333333332974823815;
    nk_f64_t const coeff_1 = -0.000198412698412696162806809;
    nk_f64_t const coeff_2 = +2.75573192239198747630416e-06;
    nk_f64_t const coeff_3 = -2.50521083763502045810755e-08;
    nk_f64_t const coeff_4 = +1.60590430605664501629054e-10;
    nk_f64_t const coeff_5 = -7.64712219118158833288484e-13;
    nk_f64_t const coeff_6 = +2.81009972710863200091251e-15;
    nk_f64_t const coeff_7 = -7.97255955009037868891952e-18;
    nk_f64_t const coeff_8 = -0.166666666666666657414808;

    // Compute (multiple_of_pi) = 2 * round(angle / π - 0.5) + 1
    nk_f64_t const quotient = angle_radians * pi_reciprocal - 0.5;
    int const multiple_of_pi = 2 * (int)(quotient < 0 ? quotient - 0.5 : quotient + 0.5) + 1;

    // Reduce the angle to: (angle - (multiple_of_pi * π)) in [-π/2, π/2]
    nk_f64_t angle = angle_radians;
    angle = angle - (multiple_of_pi * pi_high_half);
    angle = angle - (multiple_of_pi * pi_low_half);
    if ((multiple_of_pi & 2) == 0) angle = -angle;
    nk_f64_t const angle_squared = angle * angle;
    nk_f64_t const angle_cubed = angle * angle_squared;
    nk_f64_t const angle_quartic = angle_squared * angle_squared;
    nk_f64_t const angle_octic = angle_quartic * angle_quartic;

    // Compute higher-degree polynomial terms using FMA
    nk_f64_t const poly_67 = nk_f64_fma_serial(angle_squared, coeff_7, coeff_6);
    nk_f64_t const poly_45 = nk_f64_fma_serial(angle_squared, coeff_5, coeff_4);
    nk_f64_t const poly_4567 = nk_f64_fma_serial(angle_quartic, poly_67, poly_45);

    // Compute lower-degree polynomial terms using FMA
    nk_f64_t const poly_23 = nk_f64_fma_serial(angle_squared, coeff_3, coeff_2);
    nk_f64_t const poly_01 = nk_f64_fma_serial(angle_squared, coeff_1, coeff_0);
    nk_f64_t const poly_0123 = nk_f64_fma_serial(angle_quartic, poly_23, poly_01);

    // Combine polynomial terms using FMA
    nk_f64_t result = nk_f64_fma_serial(angle_octic, poly_4567, poly_0123);
    result = nk_f64_fma_serial(result, angle_squared, coeff_8);
    result = nk_f64_fma_serial(result, angle_cubed, angle);
    return result;
}

/**
 *  @brief Computes the arc-tangent of a value with @b 0-ULP error bound.
 *  @see Based on @b `xatan` in SLEEF library.
 *  @param  input The input value.
 *  @return The arc-tangent of the input value in [-π/2, π/2] radians range.
 */
NK_PUBLIC nk_f64_t nk_f64_atan(nk_f64_t const input) {
    // Polynomial coefficients for atan approximation
    nk_f64_t const coeff_19 = -1.88796008463073496563746e-05;
    nk_f64_t const coeff_18 = +0.000209850076645816976906797;
    nk_f64_t const coeff_17 = -0.00110611831486672482563471;
    nk_f64_t const coeff_16 = +0.00370026744188713119232403;
    nk_f64_t const coeff_15 = -0.00889896195887655491740809;
    nk_f64_t const coeff_14 = +0.016599329773529201970117;
    nk_f64_t const coeff_13 = -0.0254517624932312641616861;
    nk_f64_t const coeff_12 = +0.0337852580001353069993897;
    nk_f64_t const coeff_11 = -0.0407629191276836500001934;
    nk_f64_t const coeff_10 = +0.0466667150077840625632675;
    nk_f64_t const coeff_9 = -0.0523674852303482457616113;
    nk_f64_t const coeff_8 = +0.0587666392926673580854313;
    nk_f64_t const coeff_7 = -0.0666573579361080525984562;
    nk_f64_t const coeff_6 = +0.0769219538311769618355029;
    nk_f64_t const coeff_5 = -0.090908995008245008229153;
    nk_f64_t const coeff_4 = +0.111111105648261418443745;
    nk_f64_t const coeff_3 = -0.14285714266771329383765;
    nk_f64_t const coeff_2 = +0.199999999996591265594148;
    nk_f64_t const coeff_1 = -0.333333333333311110369124;

    // Quadrant adjustment
    int quadrant = 0;
    nk_f64_t value = input;
    if (value < 0) value = -value, quadrant |= 2;
    if (value > 1) value = 1.0 / value, quadrant |= 1;
    nk_f64_t const value_squared = value * value;
    nk_f64_t const value_cubed = value * value_squared;

    // Polynomial evaluation using FMA for improved precision
    nk_f64_t polynomial = coeff_19;
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_18);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_17);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_16);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_15);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_14);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_13);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_12);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_11);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_10);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_9);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_8);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_7);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_6);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_5);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_4);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_3);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_2);
    polynomial = nk_f64_fma_serial(polynomial, value_squared, coeff_1);

    // Adjust for quadrant
    nk_f64_t const pi_half = 1.5707963267948966; // π/2
    nk_f64_t result = nk_f64_fma_serial(polynomial, value_cubed, value);
    if (quadrant & 1) result = pi_half - result;
    if (quadrant & 2) result = -result;

    return result;
}

/**
 *  @brief Computes the arc-tangent of (y/x) with @b 0-ULP error bound.
 *  @see Based on @b `xatan2` in SLEEF library.
 *  @param  y_input The input sine value.
 *  @param  x_input The input cosine value.
 *  @return The arc-tangent of (y_input/x_input) in [-π/2, π/2] radians range.
 */
NK_PUBLIC nk_f64_t nk_f64_atan2(nk_f64_t const y_input, nk_f64_t const x_input) {
    // Polynomial coefficients for atan2 approximation
    nk_f64_t const coeff_19 = -1.88796008463073496563746e-05;
    nk_f64_t const coeff_18 = +0.000209850076645816976906797;
    nk_f64_t const coeff_17 = -0.00110611831486672482563471;
    nk_f64_t const coeff_16 = +0.00370026744188713119232403;
    nk_f64_t const coeff_15 = -0.00889896195887655491740809;
    nk_f64_t const coeff_14 = +0.016599329773529201970117;
    nk_f64_t const coeff_13 = -0.0254517624932312641616861;
    nk_f64_t const coeff_12 = +0.0337852580001353069993897;
    nk_f64_t const coeff_11 = -0.0407629191276836500001934;
    nk_f64_t const coeff_10 = +0.0466667150077840625632675;
    nk_f64_t const coeff_9 = -0.0523674852303482457616113;
    nk_f64_t const coeff_8 = +0.0587666392926673580854313;
    nk_f64_t const coeff_7 = -0.0666573579361080525984562;
    nk_f64_t const coeff_6 = +0.0769219538311769618355029;
    nk_f64_t const coeff_5 = -0.090908995008245008229153;
    nk_f64_t const coeff_4 = +0.111111105648261418443745;
    nk_f64_t const coeff_3 = -0.14285714266771329383765;
    nk_f64_t const coeff_2 = +0.199999999996591265594148;
    nk_f64_t const coeff_1 = -0.333333333333311110369124;

    nk_fui64_t x_bits, y_bits;
    x_bits.f = x_input, y_bits.f = y_input;
    nk_fui64_t x_abs, y_abs;
    y_abs.u = y_bits.u & 0x7FFFFFFFFFFFFFFFull;

    // Quadrant adjustment
    int quadrant = 0;
    if (x_input < 0) { x_abs.f = -x_input, quadrant = -2; }
    else { x_abs.f = x_input; }
    // Now make sure its proper fraction, where the nominator is smaller than the denominator,
    // otherwise swap the absolute values that we will use down the road, but keep the `x_bits` and `y_bits`
    // as is for final qdrant re-adjustment.
    if (y_abs.f > x_abs.f) {
        nk_f64_t temp = x_abs.f;
        x_abs.f = y_abs.f;
        y_abs.f = -temp;
        quadrant += 1;
    }

    // Argument reduction
    nk_f64_t const ratio = y_abs.f / x_abs.f;
    nk_f64_t const ratio_squared = ratio * ratio;
    nk_f64_t const ratio_cubed = ratio * ratio_squared;

    // Polynomial evaluation using FMA for improved precision
    nk_f64_t polynomial = nk_f64_fma_serial(coeff_19, ratio_squared, coeff_18);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_17);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_16);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_15);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_14);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_13);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_12);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_11);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_10);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_9);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_8);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_7);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_6);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_5);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_4);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_3);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_2);
    polynomial = nk_f64_fma_serial(polynomial, ratio_squared, coeff_1);

    // Adjust for quadrant
    nk_f64_t const pi = 3.14159265358979323846;     // π
    nk_f64_t const pi_half = 1.5707963267948966;    // π/2
    nk_f64_t const pi_quarter = 0.7853981633974483; // π/4
    nk_u64_t const negative_zero = 0x8000000000000000ull;
    nk_u64_t const positive_infinity = 0x7FF0000000000000ull;
    nk_u64_t const negative_infinity = 0xFFF0000000000000ull;
    nk_f64_t result = nk_f64_fma_serial(polynomial, ratio_cubed, ratio);
    result = nk_f64_fma_serial((nk_f64_t)quadrant, pi_half, result);

    // Special cases handling using bit reinterpretation
    int const x_is_inf = (x_bits.u == positive_infinity) | (x_bits.u == negative_infinity);
    int const y_is_inf = (y_bits.u == positive_infinity) | (y_bits.u == negative_infinity);

    // Perform the sign multiplication and infer the right quadrant
    nk_fui64_t result_bits;
    result_bits.f = result;
    // Sign transfer:
    result_bits.u ^= x_bits.u & negative_zero;
    // Quadrant adjustments:
    if (x_is_inf | (x_bits.f == 0)) result_bits.f = pi_half - (x_is_inf ? (x_bits.f < 0 ? pi_half : 0) : 0);
    if (y_is_inf) result_bits.f = pi_half - (x_is_inf ? (x_bits.f < 0 ? pi_half : pi_quarter) : 0);
    if (y_bits.f == 0) result_bits.f = (x_bits.f < 0 ? pi : 0);
    if (x_is_inf | y_is_inf) result_bits.u = 0x7FF8000000000000ull;
    // Sign transfer back:
    else { result_bits.u ^= y_bits.u & negative_zero; }
    return result_bits.f;
}

/**
 *  @brief Computes an approximate tangent of the given angle in radians with @b 3-ULP error bound for [-2π, 2π].
 *  @param[in] angle_radians The input angle in radians.
 *  @return The approximate tangent of the input angle.
 */
NK_PUBLIC nk_f32_t nk_f32_tan(nk_f32_t const angle_radians) {

    // Cody-Waite constants for argument reduction
    nk_f32_t const pi_hi = 3.1415927f;
    nk_f32_t const pi_lo = -8.742278e-8f;
    nk_f32_t const pi_half = 1.57079632679489661923f;       /// π/2
    nk_f32_t const pi_quarter = 0.78539816339744830962f;    /// π/4
    nk_f32_t const pi_reciprocal = 0.31830988618379067154f; /// 1/π

    // Polynomial coefficients for tangent approximation (minimax polynomial)
    nk_f32_t const coeff_7 = +0.002443315461f; /// Coefficient for x⁷ term
    nk_f32_t const coeff_5 = +0.05338123068f;  /// Coefficient for x⁵ term
    nk_f32_t const coeff_3 = +0.3333314061f;   /// Coefficient for x³ term

    // Compute (multiple_of_pi) = round(angle / π)
    nk_f32_t const quotient = angle_radians * pi_reciprocal;
    int const multiple_of_pi = (int)(quotient < 0 ? quotient - 0.5f : quotient + 0.5f);

    // Cody-Waite range reduction
    nk_f32_t angle = angle_radians - multiple_of_pi * pi_hi;
    angle -= multiple_of_pi * pi_lo;

    // If |angle| > π/4, use tan(x) = 1/tan(π/2 - x) for better accuracy
    int reciprocal = 0;
    if (angle > pi_quarter) {
        angle = pi_half - angle;
        reciprocal = 1;
    }
    else if (angle < -pi_quarter) {
        angle = -pi_half - angle;
        reciprocal = 1;
    }

    // Compute the polynomial approximation: tan(x) ≈ x + c3 × x³ + c5 × x⁵ + c7 × x⁷
    nk_f32_t const angle_squared = angle * angle;
    nk_f32_t const angle_cubed = angle * angle_squared;

    nk_f32_t polynomial = coeff_7;
    polynomial = polynomial * angle_squared + coeff_5;
    polynomial = polynomial * angle_squared + coeff_3;
    nk_f32_t result = polynomial * angle_cubed + angle;

    // Apply reciprocal if we reduced from outer region
    if (reciprocal) result = 1.0f / result;
    return result;
}

/**
 *  @brief Computes the tangent of the given angle in radians with @b 0-ULP error bound in [-2π, 2π].
 *  @param[in] angle_radians The input angle in radians.
 *  @return The approximate tangent of the input angle.
 */
NK_PUBLIC nk_f64_t nk_f64_tan(nk_f64_t const angle_radians) {

    // Constants for argument reduction
    nk_f64_t const pi_high = 3.141592653589793116;                         /// High-digits part of π
    nk_f64_t const pi_low = 1.2246467991473532072e-16;                     /// Low-digits part of π
    nk_f64_t const pi_half = 1.5707963267948966192313216916398;            /// π/2
    nk_f64_t const pi_quarter = 0.78539816339744830961566084581988;        /// π/4
    nk_f64_t const pi_reciprocal = 0.318309886183790671537767526745028724; /// 1/π

    // Polynomial coefficients for tangent approximation (minimax polynomial)
    nk_f64_t const coeff_13 = +0.000024030521244861858; /// Coefficient for x¹³ term
    nk_f64_t const coeff_11 = +0.00035923150434482523;  /// Coefficient for x¹¹ term
    nk_f64_t const coeff_9 = +0.0058685277932046705;    /// Coefficient for x⁹ term
    nk_f64_t const coeff_7 = +0.021869488294859542;     /// Coefficient for x⁷ term
    nk_f64_t const coeff_5 = +0.053968253972902704;     /// Coefficient for x⁵ term
    nk_f64_t const coeff_3 = +0.13333333333320124;      /// Coefficient for x³ term
    nk_f64_t const coeff_1 = +0.33333333333333331;      /// Coefficient for x term

    // Compute (multiple_of_pi) = round(angle / π)
    nk_f64_t const quotient = angle_radians * pi_reciprocal;
    int const multiple_of_pi = (int)(quotient < 0 ? quotient - 0.5 : quotient + 0.5);

    // Reduce the angle using high/low precision split
    nk_f64_t angle = angle_radians;
    angle = angle - (multiple_of_pi * pi_high);
    angle = angle - (multiple_of_pi * pi_low);

    // If |angle| > π/4, use tan(x) = 1/tan(π/2 - x) for better accuracy
    int reciprocal = 0;
    if (angle > pi_quarter) {
        angle = pi_half - angle;
        reciprocal = 1;
    }
    else if (angle < -pi_quarter) {
        angle = -pi_half - angle;
        reciprocal = 1;
    }

    // Compute powers of angle
    nk_f64_t const angle_squared = angle * angle;
    nk_f64_t const angle_cubed = angle * angle_squared;

    // Compute the polynomial approximation: tan(x) ≈ x × (1 + c1 × x² + c3 × x⁴ + ...)
    nk_f64_t polynomial = coeff_13;
    polynomial = polynomial * angle_squared + coeff_11;
    polynomial = polynomial * angle_squared + coeff_9;
    polynomial = polynomial * angle_squared + coeff_7;
    polynomial = polynomial * angle_squared + coeff_5;
    polynomial = polynomial * angle_squared + coeff_3;
    polynomial = polynomial * angle_squared + coeff_1;
    nk_f64_t result = polynomial * angle_cubed + angle;

    // Apply reciprocal if we reduced from outer region
    if (reciprocal) result = 1.0 / result;
    return result;
}

NK_PUBLIC void nk_each_sin_f32_serial(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_f32_sin(ins[i]);
}
NK_PUBLIC void nk_each_cos_f32_serial(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_f32_cos(ins[i]);
}
NK_PUBLIC void nk_each_atan_f32_serial(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs) {
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_f32_atan(ins[i]);
}
NK_PUBLIC void nk_each_sin_f64_serial(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_f64_sin(ins[i]);
}
NK_PUBLIC void nk_each_cos_f64_serial(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_f64_cos(ins[i]);
}
NK_PUBLIC void nk_each_atan_f64_serial(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs) {
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_f64_atan(ins[i]);
}

NK_PUBLIC void nk_each_sin_f16_serial(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t angle_f32;
        nk_f16_to_f32_serial(&ins[i], &angle_f32);
        nk_f32_t const result_f32 = nk_f32_sin(angle_f32);
        nk_f32_to_f16_serial(&result_f32, &outs[i]);
    }
}

NK_PUBLIC void nk_each_cos_f16_serial(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t angle_f32;
        nk_f16_to_f32_serial(&ins[i], &angle_f32);
        nk_f32_t const result_f32 = nk_f32_cos(angle_f32);
        nk_f32_to_f16_serial(&result_f32, &outs[i]);
    }
}

NK_PUBLIC void nk_each_atan_f16_serial(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs) {
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t value_f32;
        nk_f16_to_f32_serial(&ins[i], &value_f32);
        nk_f32_t const result_f32 = nk_f32_atan(value_f32);
        nk_f32_to_f16_serial(&result_f32, &outs[i]);
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TRIGONOMETRY_SERIAL_H
