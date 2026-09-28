/**
 *  @file include/numkong/scalar/serial.h
 *  @author Ash Vardanian
 *  @date January 3, 2026
 *  @brief Software-emulated scalar math helpers for SIMD-free CPUs.
 *
 *  @sa include/numkong/scalar.h
 *
 *  Uses the Quake 3 fast inverse square root trick with Newton-Raphson refinement: three iterations
 *  for f32, ~34.9 correct bits, and four for f64, ~69.3 correct bits.
 */
#ifndef NUMKONG_SCALAR_SERIAL_H
#define NUMKONG_SCALAR_SERIAL_H

#include "numkong/types.h"
#include "numkong/cast/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

/*  GCC inlines a helper only into callers whose targets include its own, so serial code builds at
 *  the Armv8-A floor. */
#if defined(__GNUC__) && !defined(__clang__) && NUMKONG_ARCH_ARM64_
#pragma GCC push_options
#pragma GCC target("arch=armv8-a")
#endif

/** Reciprocal square root of @p number: a bit-trick seed and three Newton steps, ~34.9 bits. */
NUMKONG_INLINE nk_f32_t nk_f32_rsqrt_(nk_f32_t number) {
    nk_fui32_t conv;
    conv.f = number;
    conv.u = 0x5F375A86 - (conv.u >> 1);
    nk_f32_t y = conv.f;
    y = y * (1.5f - 0.5f * number * y * y);
    y = y * (1.5f - 0.5f * number * y * y);
    y = y * (1.5f - 0.5f * number * y * y);
    return y;
}

/** Square root of @p number as @p number times its reciprocal square root, zero for non-positives. */
NUMKONG_INLINE nk_f32_t nk_f32_sqrt_(nk_f32_t number) { return number > 0 ? number * nk_f32_rsqrt_(number) : 0; }

/** Reciprocal square root of @p number: a bit-trick seed and four Newton steps, ~69.3 bits. */
NUMKONG_INLINE nk_f64_t nk_f64_rsqrt_(nk_f64_t number) {
    nk_fui64_t conv;
    conv.f = number;
    conv.u = 0x5FE6EB50C7B537A9ULL - (conv.u >> 1);
    nk_f64_t y = conv.f;
    y = y * (1.5 - 0.5 * number * y * y);
    y = y * (1.5 - 0.5 * number * y * y);
    y = y * (1.5 - 0.5 * number * y * y);
    y = y * (1.5 - 0.5 * number * y * y);
    return y;
}

/** Square root of @p number as @p number times its reciprocal square root, zero for non-positives. */
NUMKONG_INLINE nk_f64_t nk_f64_sqrt_(nk_f64_t number) { return number > 0 ? number * nk_f64_rsqrt_(number) : 0; }

/** Fused multiply-add emulated in F64 with Dekker's TwoProduct and Knuth's TwoSum error terms. */
NUMKONG_CONSTEXPR nk_f64_t nk_f64_fma_(nk_f64_t multiplicand, nk_f64_t multiplier,
                                       nk_f64_t addend) NUMKONG_STREAMABLE_ {
    nk_f64_t product = multiplicand * multiplier;
    // Dekker splitting: break each operand into non-overlapping high and low halves
    nk_f64_t const dekker_split = 134217729.0; // 2^27 + 1 for double precision
    nk_f64_t multiplicand_high = dekker_split * multiplicand;
    nk_f64_t multiplicand_low = multiplicand - (multiplicand_high - (multiplicand_high - multiplicand));
    multiplicand_high = multiplicand_high - (multiplicand_high - multiplicand);
    nk_f64_t multiplier_high = dekker_split * multiplier;
    nk_f64_t multiplier_low = multiplier - (multiplier_high - (multiplier_high - multiplier));
    multiplier_high = multiplier_high - (multiplier_high - multiplier);
    // Exact multiplication error from the four cross-products
    nk_f64_t product_error = ((multiplicand_high * multiplier_high - product) + multiplicand_high * multiplier_low +
                              multiplicand_low * multiplier_high) +
                             multiplicand_low * multiplier_low;
    // Knuth TwoSum: add the addend with error tracking
    nk_f64_t result = product + addend;
    nk_f64_t addend_recovered = result - product;
    nk_f64_t product_recovered = result - addend_recovered;
    nk_f64_t addition_error = (product - product_recovered) + (addend - addend_recovered);
    return result + (product_error + addition_error);
}

/** Fused multiply-add emulated in F32 with Dekker's TwoProduct and Knuth's TwoSum error terms. */
NUMKONG_CONSTEXPR nk_f32_t nk_f32_fma_(nk_f32_t multiplicand, nk_f32_t multiplier, nk_f32_t addend) {
    nk_f32_t product = multiplicand * multiplier;
    // Dekker splitting: break each operand into non-overlapping high and low halves
    nk_f32_t const dekker_split = 4097.0f; // 2^12 + 1 for single precision
    nk_f32_t multiplicand_high = dekker_split * multiplicand;
    nk_f32_t multiplicand_low = multiplicand - (multiplicand_high - (multiplicand_high - multiplicand));
    multiplicand_high = multiplicand_high - (multiplicand_high - multiplicand);
    nk_f32_t multiplier_high = dekker_split * multiplier;
    nk_f32_t multiplier_low = multiplier - (multiplier_high - (multiplier_high - multiplier));
    multiplier_high = multiplier_high - (multiplier_high - multiplier);
    // Exact multiplication error from the four cross-products
    nk_f32_t product_error = ((multiplicand_high * multiplier_high - product) + multiplicand_high * multiplier_low +
                              multiplicand_low * multiplier_high) +
                             multiplicand_low * multiplier_low;
    // Knuth TwoSum: add the addend with error tracking
    nk_f32_t result = product + addend;
    nk_f32_t addend_recovered = result - product;
    nk_f32_t product_recovered = result - addend_recovered;
    nk_f32_t addition_error = (product - product_recovered) + (addend - addend_recovered);
    return result + (product_error + addition_error);
}

/*  Keep the serial instantiations below actually scalar, regardless of build type.
 *  See dots/serial.h for rationale. */
#if NUMKONG_TARGET_SERIAL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

NUMKONG_API nk_f32_t nk_f32_rsqrt_serial(nk_f32_t number) { return nk_f32_rsqrt_(number); }
NUMKONG_API nk_f32_t nk_f32_sqrt_serial(nk_f32_t number) { return nk_f32_sqrt_(number); }
NUMKONG_API nk_f64_t nk_f64_rsqrt_serial(nk_f64_t number) { return nk_f64_rsqrt_(number); }
NUMKONG_API nk_f64_t nk_f64_sqrt_serial(nk_f64_t number) { return nk_f64_sqrt_(number); }

NUMKONG_API nk_f16_t nk_f16_sqrt_serial(nk_f16_t x) {
    nk_f32_t x_f32;
    nk_f16_to_f32_(&x, &x_f32);
    x_f32 = nk_f32_sqrt_(x_f32);
    nk_f16_t result;
    nk_f32_to_f16_(&x_f32, &result);
    return result;
}

NUMKONG_API nk_f16_t nk_f16_rsqrt_serial(nk_f16_t x) {
    nk_f32_t x_f32;
    nk_f16_to_f32_(&x, &x_f32);
    x_f32 = nk_f32_rsqrt_(x_f32);
    nk_f16_t result;
    nk_f32_to_f16_(&x_f32, &result);
    return result;
}

NUMKONG_API nk_f64_t nk_f64_fma_serial(nk_f64_t multiplicand, nk_f64_t multiplier, nk_f64_t addend) {
    return nk_f64_fma_(multiplicand, multiplier, addend);
}
NUMKONG_API nk_f32_t nk_f32_fma_serial(nk_f32_t multiplicand, nk_f32_t multiplier, nk_f32_t addend) {
    return nk_f32_fma_(multiplicand, multiplier, addend);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NUMKONG_TARGET_SERIAL

/** Scalar Dot2 accumulator: sum += a × b with error compensation, using the TwoProd, via FMA, and
 *  TwoSum error-free transformations from "Accurate Sum and Dot Product" by T. Ogita, S. M. Rump
 *  and S. Oishi, 2005. */
NUMKONG_INLINE void nk_f64_dot2_(nk_f64_t *sum, nk_f64_t *compensation, nk_f64_t a, nk_f64_t b) NUMKONG_STREAMABLE_ {
    nk_f64_t product = a * b;
    nk_f64_t product_error = nk_f64_fma_(a, b, -product);
    nk_f64_t running_sum = *sum + product;
    nk_f64_t recovered_addend = running_sum - *sum;
    nk_f64_t sum_error = (*sum - (running_sum - recovered_addend)) + (product - recovered_addend);
    *sum = running_sum;
    *compensation += sum_error + product_error;
}

/** Sum of @p a and @p b clamped to the U16 range. */
NUMKONG_CONSTEXPR nk_u16_t nk_u16_saturating_add_(nk_u16_t a, nk_u16_t b) {
    nk_u32_t result = (nk_u32_t)a + (nk_u32_t)b;
    return (result > 65535u) ? (nk_u16_t)65535u : (nk_u16_t)result;
}
/** Sum of @p a and @p b clamped to the U32 range. */
NUMKONG_CONSTEXPR nk_u32_t nk_u32_saturating_add_(nk_u32_t a, nk_u32_t b) {
    nk_u64_t result = (nk_u64_t)a + (nk_u64_t)b;
    return (result > 4294967295u) ? (nk_u32_t)4294967295u : (nk_u32_t)result;
}
/** Sum of @p a and @p b clamped to the U64 range. */
NUMKONG_CONSTEXPR nk_u64_t nk_u64_saturating_add_(nk_u64_t a, nk_u64_t b) {
    return (a + b < a) ? 18446744073709551615ull : (a + b);
}
/** Sum of @p a and @p b clamped to the I16 range. */
NUMKONG_CONSTEXPR nk_i16_t nk_i16_saturating_add_(nk_i16_t a, nk_i16_t b) {
    nk_i32_t result = (nk_i32_t)a + (nk_i32_t)b;
    return (result > 32767) ? 32767 : (result < -32768 ? -32768 : result);
}
/** Sum of @p a and @p b clamped to the I32 range. */
NUMKONG_CONSTEXPR nk_i32_t nk_i32_saturating_add_(nk_i32_t a, nk_i32_t b) {
    nk_i64_t result = (nk_i64_t)a + (nk_i64_t)b;
    return (result > 2147483647ll) ? 2147483647ll : (result < -2147483648ll ? -2147483648ll : (nk_i32_t)result);
}
/** Sum of @p a and @p b clamped to the I64 range, checked before adding. */
NUMKONG_CONSTEXPR nk_i64_t nk_i64_saturating_add_(nk_i64_t a, nk_i64_t b) {
    //? We can't just write `-9223372036854775808ll`, even though it's the smallest signed 64-bit value.
    //? The compiler will complain about the number being too large for the type, as it will process the
    //? constant and the sign separately. So we use the same hint that compilers use to define the `INT64_MIN`.
    if ((b > 0) && (a > (9223372036854775807ll) - b)) return 9223372036854775807ll;
    if ((b < 0) && (a < (-9223372036854775807ll - 1ll) - b)) return -9223372036854775807ll - 1ll;
    return a + b;
}

/** Product of @p a and @p b clamped to the U64 range, from four 32-bit partial products. */
NUMKONG_CONSTEXPR nk_u64_t nk_u64_saturating_mul_(nk_u64_t a, nk_u64_t b) {
    // Split the inputs into high and low 32-bit parts
    nk_u64_t a_high = a >> 32;
    nk_u64_t a_low = a & 0xFFFFFFFF;
    nk_u64_t b_high = b >> 32;
    nk_u64_t b_low = b & 0xFFFFFFFF;

    // Compute partial products
    nk_u64_t upper_product = a_high * b_high;
    nk_u64_t cross_ab = a_high * b_low;
    nk_u64_t cross_ba = a_low * b_high;
    nk_u64_t lower_product = a_low * b_low;

    // Check if the high part of the result overflows
    nk_u64_t cross_sum = cross_ab + cross_ba;
    if (upper_product || (cross_ab >> 32) || (cross_ba >> 32) || (cross_sum < cross_ab) || (cross_sum >> 32))
        return 18446744073709551615ull;
    nk_u64_t result = (cross_sum << 32) + lower_product;
    if (result < lower_product) return 18446744073709551615ull;
    return result;
}

/** Product of @p a and @p b clamped to the I64 range, from the magnitudes' 32-bit partial products. */
NUMKONG_CONSTEXPR nk_i64_t nk_i64_saturating_mul_(nk_i64_t a, nk_i64_t b) {
    int sign = ((a < 0) ^ (b < 0)) ? -1 : 1; // Track the sign of the result

    // Take absolute values for easy multiplication and overflow detection
    nk_u64_t abs_a = (a < 0) ? (0u - (nk_u64_t)a) : (nk_u64_t)a;
    nk_u64_t abs_b = (b < 0) ? (0u - (nk_u64_t)b) : (nk_u64_t)b;

    // Split the absolute values into high and low 32-bit parts
    nk_u64_t a_high = abs_a >> 32;
    nk_u64_t a_low = abs_a & 0xFFFFFFFF;
    nk_u64_t b_high = abs_b >> 32;
    nk_u64_t b_low = abs_b & 0xFFFFFFFF;

    // Compute partial products
    nk_u64_t upper_product = a_high * b_high;
    nk_u64_t cross_ab = a_high * b_low;
    nk_u64_t cross_ba = a_low * b_high;
    nk_u64_t lower_product = a_low * b_low;

    // Check for overflow and saturate based on sign
    nk_u64_t cross_sum = cross_ab + cross_ba;
    if (upper_product || (cross_ab >> 32) || (cross_ba >> 32) || (cross_sum < cross_ab) || (cross_sum >> 32))
        return (sign > 0) ? 9223372036854775807ll : (-9223372036854775807ll - 1ll);
    // Combine parts if no overflow, then apply the sign
    nk_u64_t result = (cross_sum << 32) + lower_product;
    if (result < lower_product || result > 9223372036854775807ull + (sign < 0))
        return (sign > 0) ? 9223372036854775807ll : (-9223372036854775807ll - 1ll);
    return (sign < 0) ? (nk_i64_t)(0u - result) : (nk_i64_t)result;
}

/** Orders two E4M3 values by their sign-magnitude bits: negative, zero or positive, NaNs outermost. */
NUMKONG_CONSTEXPR int nk_e4m3_order_(nk_e4m3_t a, nk_e4m3_t b) {
    int sign_a = a >> 7, sign_b = b >> 7;
    return (a ^ -sign_a) - (b ^ -sign_b);
}
/** Orders two E5M2 values by their sign-magnitude bits: negative, zero or positive, NaNs outermost. */
NUMKONG_CONSTEXPR int nk_e5m2_order_(nk_e5m2_t a, nk_e5m2_t b) {
    int sign_a = a >> 7, sign_b = b >> 7;
    return (a ^ -sign_a) - (b ^ -sign_b);
}
/** Orders two E2M3 values by their 6 sign-magnitude bits: negative, zero or positive. */
NUMKONG_CONSTEXPR int nk_e2m3_order_(nk_e2m3_t a, nk_e2m3_t b) {
    int value_a = a & 0x3F, value_b = b & 0x3F;
    int sign_a = value_a >> 5, sign_b = value_b >> 5;
    return (value_a ^ -sign_a) - (value_b ^ -sign_b);
}
/** Orders two E3M2 values by their 6 sign-magnitude bits: negative, zero or positive. */
NUMKONG_CONSTEXPR int nk_e3m2_order_(nk_e3m2_t a, nk_e3m2_t b) {
    int value_a = a & 0x3F, value_b = b & 0x3F;
    int sign_a = value_a >> 5, sign_b = value_b >> 5;
    return (value_a ^ -sign_a) - (value_b ^ -sign_b);
}
/** Orders two BF16 values by their sign-magnitude bits: negative, zero or positive, NaNs outermost. */
NUMKONG_INLINE int nk_bf16_order_(nk_bf16_t a, nk_bf16_t b) {
    nk_fui16_t a_fui, b_fui;
    a_fui.bf = a, b_fui.bf = b;
    int sign_a = a_fui.u >> 15, sign_b = b_fui.u >> 15;
    return ((int)a_fui.u ^ -sign_a) - ((int)b_fui.u ^ -sign_b);
}
/** Orders two F16 values by their sign-magnitude bits: negative, zero or positive, NaNs outermost. */
NUMKONG_INLINE int nk_f16_order_(nk_f16_t a, nk_f16_t b) {
    nk_fui16_t a_fui, b_fui;
    a_fui.f = a, b_fui.f = b;
    int sign_a = a_fui.u >> 15, sign_b = b_fui.u >> 15;
    return ((int)a_fui.u ^ -sign_a) - ((int)b_fui.u ^ -sign_b);
}

#if NUMKONG_TARGET_SERIAL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

NUMKONG_API nk_f16_t nk_f16_fma_serial(nk_f16_t a, nk_f16_t b, nk_f16_t c) {
    nk_f32_t a_f32, b_f32, c_f32;
    nk_f16_to_f32_(&a, &a_f32);
    nk_f16_to_f32_(&b, &b_f32);
    nk_f16_to_f32_(&c, &c_f32);
    nk_f32_t result_f32 = nk_f32_fma_(a_f32, b_f32, c_f32);
    nk_f16_t result;
    nk_f32_to_f16_(&result_f32, &result);
    return result;
}

NUMKONG_API nk_u8_t nk_u8_saturating_add_serial(nk_u8_t a, nk_u8_t b) {
    nk_u16_t result = (nk_u16_t)a + (nk_u16_t)b;
    return (result > 255u) ? (nk_u8_t)255u : (nk_u8_t)result;
}
NUMKONG_API nk_u16_t nk_u16_saturating_add_serial(nk_u16_t a, nk_u16_t b) { return nk_u16_saturating_add_(a, b); }
NUMKONG_API nk_u32_t nk_u32_saturating_add_serial(nk_u32_t a, nk_u32_t b) { return nk_u32_saturating_add_(a, b); }
NUMKONG_API nk_u64_t nk_u64_saturating_add_serial(nk_u64_t a, nk_u64_t b) { return nk_u64_saturating_add_(a, b); }
NUMKONG_API nk_i8_t nk_i8_saturating_add_serial(nk_i8_t a, nk_i8_t b) {
    nk_i16_t result = (nk_i16_t)a + (nk_i16_t)b;
    return (result > 127) ? 127 : (result < -128 ? -128 : result);
}
NUMKONG_API nk_i16_t nk_i16_saturating_add_serial(nk_i16_t a, nk_i16_t b) { return nk_i16_saturating_add_(a, b); }
NUMKONG_API nk_i32_t nk_i32_saturating_add_serial(nk_i32_t a, nk_i32_t b) { return nk_i32_saturating_add_(a, b); }
NUMKONG_API nk_i64_t nk_i64_saturating_add_serial(nk_i64_t a, nk_i64_t b) { return nk_i64_saturating_add_(a, b); }

NUMKONG_API nk_u8_t nk_u8_saturating_mul_serial(nk_u8_t a, nk_u8_t b) {
    nk_u16_t result = (nk_u16_t)a * (nk_u16_t)b;
    return (result > 255) ? 255 : (nk_u8_t)result;
}

NUMKONG_API nk_u16_t nk_u16_saturating_mul_serial(nk_u16_t a, nk_u16_t b) {
    nk_u32_t result = (nk_u32_t)a * (nk_u32_t)b;
    return (result > 65535) ? 65535 : (nk_u16_t)result;
}

NUMKONG_API nk_u32_t nk_u32_saturating_mul_serial(nk_u32_t a, nk_u32_t b) {
    nk_u64_t result = (nk_u64_t)a * (nk_u64_t)b;
    return (result > 4294967295u) ? 4294967295u : (nk_u32_t)result;
}

NUMKONG_API nk_u64_t nk_u64_saturating_mul_serial(nk_u64_t a, nk_u64_t b) { return nk_u64_saturating_mul_(a, b); }

NUMKONG_API nk_i8_t nk_i8_saturating_mul_serial(nk_i8_t a, nk_i8_t b) {
    nk_i16_t result = (nk_i16_t)a * (nk_i16_t)b;
    return (result > 127) ? 127 : (result < -128 ? -128 : (nk_i8_t)result);
}

NUMKONG_API nk_i16_t nk_i16_saturating_mul_serial(nk_i16_t a, nk_i16_t b) {
    nk_i32_t result = (nk_i32_t)a * (nk_i32_t)b;
    return (result > 32767) ? 32767 : (result < -32768 ? -32768 : (nk_i16_t)result);
}

NUMKONG_API nk_i32_t nk_i32_saturating_mul_serial(nk_i32_t a, nk_i32_t b) {
    nk_i64_t result = (nk_i64_t)a * (nk_i64_t)b;
    return (result > 2147483647ll) ? 2147483647ll : (result < -2147483648ll ? -2147483648ll : (nk_i32_t)result);
}

NUMKONG_API nk_i64_t nk_i64_saturating_mul_serial(nk_i64_t a, nk_i64_t b) { return nk_i64_saturating_mul_(a, b); }

NUMKONG_API nk_i4x2_t nk_i4x2_saturating_add_serial(nk_i4x2_t a, nk_i4x2_t b) {
    nk_i8_t low = nk_i4x2_low_(a) + nk_i4x2_low_(b);
    nk_i8_t high = nk_i4x2_high_(a) + nk_i4x2_high_(b);
    low = (low > 7) ? 7 : (low < -8 ? -8 : low);
    high = (high > 7) ? 7 : (high < -8 ? -8 : high);
    return (nk_i4x2_t)((low & 0x0F) | ((high & 0x0F) << 4));
}
NUMKONG_API nk_u4x2_t nk_u4x2_saturating_add_serial(nk_u4x2_t a, nk_u4x2_t b) {
    nk_u8_t low = nk_u4x2_low_(a) + nk_u4x2_low_(b);
    nk_u8_t high = nk_u4x2_high_(a) + nk_u4x2_high_(b);
    low = (low > 15) ? 15 : low;
    high = (high > 15) ? 15 : high;
    return (nk_u4x2_t)((low & 0x0F) | ((high & 0x0F) << 4));
}
NUMKONG_API nk_i4x2_t nk_i4x2_saturating_mul_serial(nk_i4x2_t a, nk_i4x2_t b) {
    nk_i8_t low = nk_i4x2_low_(a) * nk_i4x2_low_(b);
    nk_i8_t high = nk_i4x2_high_(a) * nk_i4x2_high_(b);
    low = (low > 7) ? 7 : (low < -8 ? -8 : low);
    high = (high > 7) ? 7 : (high < -8 ? -8 : high);
    return (nk_i4x2_t)((low & 0x0F) | ((high & 0x0F) << 4));
}
NUMKONG_API nk_u4x2_t nk_u4x2_saturating_mul_serial(nk_u4x2_t a, nk_u4x2_t b) {
    nk_u8_t low = nk_u4x2_low_(a) * nk_u4x2_low_(b);
    nk_u8_t high = nk_u4x2_high_(a) * nk_u4x2_high_(b);
    low = (low > 15) ? 15 : low;
    high = (high > 15) ? 15 : high;
    return (nk_u4x2_t)((low & 0x0F) | ((high & 0x0F) << 4));
}

NUMKONG_API int nk_e4m3_order_serial(nk_e4m3_t a, nk_e4m3_t b) { return nk_e4m3_order_(a, b); }
NUMKONG_API int nk_e5m2_order_serial(nk_e5m2_t a, nk_e5m2_t b) { return nk_e5m2_order_(a, b); }
NUMKONG_API int nk_e2m3_order_serial(nk_e2m3_t a, nk_e2m3_t b) { return nk_e2m3_order_(a, b); }
NUMKONG_API int nk_e3m2_order_serial(nk_e3m2_t a, nk_e3m2_t b) { return nk_e3m2_order_(a, b); }
NUMKONG_API int nk_bf16_order_serial(nk_bf16_t a, nk_bf16_t b) { return nk_bf16_order_(a, b); }
NUMKONG_API int nk_f16_order_serial(nk_f16_t a, nk_f16_t b) { return nk_f16_order_(a, b); }

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NUMKONG_TARGET_SERIAL

/** Sine of @p angle_radians: Cody-Waite reduction by π and an odd minimax polynomial. */
NUMKONG_INLINE nk_f32_t nk_f32_sin_(nk_f32_t const angle_radians) {

    // Cody-Waite constants for argument reduction, π split into high and low parts
    nk_f32_t const pi_high = 3.1415927f;
    nk_f32_t const pi_low = -8.742278e-8f;
    nk_f32_t const pi_reciprocal = 0.31830988618379067154f; // 1/π

    // Degree-9 minimax coefficients: sin(x) ≈ x + c3*x³ + c5*x⁵ + c7*x⁷ + c9*x⁹
    nk_f32_t const coeff_9 = +2.7557319224e-6f;
    nk_f32_t const coeff_7 = -1.9841269841e-4f;
    nk_f32_t const coeff_5 = +8.3333293855e-3f;
    nk_f32_t const coeff_3 = -1.6666666641e-1f;

    // Compute (multiple_of_pi) = round(angle / π)
    nk_f32_t const quotient = angle_radians * pi_reciprocal;
    int const multiple_of_pi = (int)(quotient < 0 ? quotient - 0.5f : quotient + 0.5f);

    // Cody-Waite range reduction: angle = angle_radians - multiple * (pi_high + pi_low)
    nk_f32_t angle = angle_radians - (nk_f32_t)multiple_of_pi * pi_high;
    angle -= (nk_f32_t)multiple_of_pi * pi_low;
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

/** Cosine of @p angle_radians: Cody-Waite reduction by π around π/2 and an odd polynomial. */
NUMKONG_INLINE nk_f32_t nk_f32_cos_(nk_f32_t const angle_radians) {

    // Cody-Waite constants for argument reduction, π split into high and low parts
    nk_f32_t const pi_high = 3.1415927f;
    nk_f32_t const pi_low = -8.742278e-8f;
    nk_f32_t const pi_half = 1.57079632679489661923f;       // π/2
    nk_f32_t const pi_reciprocal = 0.31830988618379067154f; // 1/π

    // Degree-9 minimax coefficients: sin(x) ≈ x + c3*x³ + c5*x⁵ + c7*x⁷ + c9*x⁹
    nk_f32_t const coeff_9 = +2.7557319224e-6f;
    nk_f32_t const coeff_7 = -1.9841269841e-4f;
    nk_f32_t const coeff_5 = +8.3333293855e-3f;
    nk_f32_t const coeff_3 = -1.6666666641e-1f;

    // Compute (multiple_of_pi) = round(angle / π - 0.5)
    nk_f32_t const quotient = angle_radians * pi_reciprocal - 0.5f;
    int const multiple_of_pi = (int)(quotient < 0 ? quotient - 0.5f : quotient + 0.5f);

    // Cody-Waite range reduction: angle = angle_radians - (multiple * pi + pi/2)
    nk_f32_t const offset = pi_half + (nk_f32_t)multiple_of_pi * pi_high;
    nk_f32_t angle = angle_radians - offset;
    angle -= (nk_f32_t)multiple_of_pi * pi_low;
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

/** Arctangent of @p input: reciprocal folding into [0, 1] and a degree-8 polynomial in x². */
NUMKONG_INLINE nk_f32_t nk_f32_atan_(nk_f32_t const input) {
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
    polynomial = nk_f32_fma_(polynomial, value_squared, coeff_2);
    polynomial = nk_f32_fma_(polynomial, value_squared, coeff_3);
    polynomial = nk_f32_fma_(polynomial, value_squared, coeff_4);
    polynomial = nk_f32_fma_(polynomial, value_squared, coeff_5);
    polynomial = nk_f32_fma_(polynomial, value_squared, coeff_6);
    polynomial = nk_f32_fma_(polynomial, value_squared, coeff_7);
    polynomial = nk_f32_fma_(polynomial, value_squared, coeff_8);

    // Adjust for quadrant
    nk_f32_t result = nk_f32_fma_(polynomial, value_cubed, value);
    nk_f32_t const pi_half = 1.5707963267948966f; // π/2
    if ((quadrant & 1) != 0) result = pi_half - result;
    if ((quadrant & 2) != 0) result = -result;
    return result;
}

/** Four-quadrant arctangent of @p y_input over @p x_input, with IEEE signed zeros and infinities. */
NUMKONG_INLINE nk_f32_t nk_f32_atan2_(nk_f32_t const y_input, nk_f32_t const x_input) {

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
        nk_f32_t const previous_x_abs = x_abs.f;
        x_abs.f = y_abs.f;
        y_abs.f = -previous_x_abs;
        quadrant += 1;
    }

    // Argument reduction
    nk_f32_t const ratio = y_abs.f / x_abs.f;
    nk_f32_t const ratio_squared = ratio * ratio;
    nk_f32_t const ratio_cubed = ratio * ratio_squared;

    // Polynomial evaluation using FMA for improved precision
    nk_f32_t polynomial = coeff_1;
    polynomial = nk_f32_fma_(polynomial, ratio_squared, coeff_2);
    polynomial = nk_f32_fma_(polynomial, ratio_squared, coeff_3);
    polynomial = nk_f32_fma_(polynomial, ratio_squared, coeff_4);
    polynomial = nk_f32_fma_(polynomial, ratio_squared, coeff_5);
    polynomial = nk_f32_fma_(polynomial, ratio_squared, coeff_6);
    polynomial = nk_f32_fma_(polynomial, ratio_squared, coeff_7);
    polynomial = nk_f32_fma_(polynomial, ratio_squared, coeff_8);

    // Compute the result using FMA
    nk_f32_t const pi_half = 1.5707963267948966f; // π/2
    nk_f32_t result = nk_f32_fma_(polynomial, ratio_cubed, ratio);
    result = nk_f32_fma_((nk_f32_t)quadrant, pi_half, result); // quadrant * (π/2)

    // Adjust sign
    nk_u32_t const negative_zero = 0x80000000u;
    nk_fui32_t result_bits;
    result_bits.f = result;
    result_bits.u ^= x_bits.u & negative_zero;
    result_bits.u ^= y_bits.u & negative_zero;
    return result_bits.f;
}

/** Sine of @p angle_radians: Cody-Waite reduction by π and an Estrin-evaluated odd polynomial. */
NUMKONG_INLINE nk_f64_t nk_f64_sin_(nk_f64_t const angle_radians) {

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
    nk_f64_t const poly_67 = nk_f64_fma_(angle_squared, coeff_7, coeff_6);
    nk_f64_t const poly_45 = nk_f64_fma_(angle_squared, coeff_5, coeff_4);
    nk_f64_t const poly_4567 = nk_f64_fma_(angle_quartic, poly_67, poly_45);

    // Compute lower-degree polynomial terms using FMA
    nk_f64_t const poly_23 = nk_f64_fma_(angle_squared, coeff_3, coeff_2);
    nk_f64_t const poly_01 = nk_f64_fma_(angle_squared, coeff_1, coeff_0);
    nk_f64_t const poly_0123 = nk_f64_fma_(angle_quartic, poly_23, poly_01);

    // Combine polynomial terms using FMA
    nk_f64_t result = nk_f64_fma_(angle_octic, poly_4567, poly_0123);
    result = nk_f64_fma_(result, angle_squared, coeff_8);
    result = nk_f64_fma_(result, angle_cubed, angle);

    // Handle the special case of negative zero input
    nk_fui64_t converter;
    converter.f = angle_radians;
    if ((nk_i64_t)converter.u == negative_zero) result = angle;
    return result;
}

/** Cosine of @p angle_radians: Cody-Waite reduction around π/2 and an Estrin-evaluated polynomial. */
NUMKONG_INLINE nk_f64_t nk_f64_cos_(nk_f64_t const angle_radians) {

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
    nk_f64_t const poly_67 = nk_f64_fma_(angle_squared, coeff_7, coeff_6);
    nk_f64_t const poly_45 = nk_f64_fma_(angle_squared, coeff_5, coeff_4);
    nk_f64_t const poly_4567 = nk_f64_fma_(angle_quartic, poly_67, poly_45);

    // Compute lower-degree polynomial terms using FMA
    nk_f64_t const poly_23 = nk_f64_fma_(angle_squared, coeff_3, coeff_2);
    nk_f64_t const poly_01 = nk_f64_fma_(angle_squared, coeff_1, coeff_0);
    nk_f64_t const poly_0123 = nk_f64_fma_(angle_quartic, poly_23, poly_01);

    // Combine polynomial terms using FMA
    nk_f64_t result = nk_f64_fma_(angle_octic, poly_4567, poly_0123);
    result = nk_f64_fma_(result, angle_squared, coeff_8);
    result = nk_f64_fma_(result, angle_cubed, angle);
    return result;
}

/** Arctangent of @p input: reciprocal folding into [0, 1] and a degree-19 polynomial in x². */
NUMKONG_INLINE nk_f64_t nk_f64_atan_(nk_f64_t const input) {
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
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_18);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_17);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_16);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_15);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_14);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_13);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_12);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_11);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_10);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_9);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_8);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_7);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_6);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_5);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_4);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_3);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_2);
    polynomial = nk_f64_fma_(polynomial, value_squared, coeff_1);

    // Adjust for quadrant
    nk_f64_t const pi_half = 1.5707963267948966; // π/2
    nk_f64_t result = nk_f64_fma_(polynomial, value_cubed, value);
    if (quadrant & 1) result = pi_half - result;
    if (quadrant & 2) result = -result;

    return result;
}

/** Four-quadrant arctangent of @p y_input over @p x_input, with IEEE signed zeros and infinities. */
NUMKONG_INLINE nk_f64_t nk_f64_atan2_(nk_f64_t const y_input, nk_f64_t const x_input) {
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
    // Swap the absolute values into a proper fraction, the numerator below the denominator, keeping
    // `x_bits` and `y_bits` as they are for the final quadrant adjustment.
    if (y_abs.f > x_abs.f) {
        nk_f64_t const previous_x_abs = x_abs.f;
        x_abs.f = y_abs.f;
        y_abs.f = -previous_x_abs;
        quadrant += 1;
    }

    // Argument reduction
    nk_f64_t const ratio = y_abs.f / x_abs.f;
    nk_f64_t const ratio_squared = ratio * ratio;
    nk_f64_t const ratio_cubed = ratio * ratio_squared;

    // Polynomial evaluation using FMA for improved precision
    nk_f64_t polynomial = nk_f64_fma_(coeff_19, ratio_squared, coeff_18);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_17);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_16);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_15);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_14);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_13);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_12);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_11);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_10);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_9);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_8);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_7);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_6);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_5);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_4);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_3);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_2);
    polynomial = nk_f64_fma_(polynomial, ratio_squared, coeff_1);

    // Adjust for quadrant
    nk_f64_t const pi = 3.14159265358979323846;     // π
    nk_f64_t const pi_half = 1.5707963267948966;    // π/2
    nk_f64_t const pi_quarter = 0.7853981633974483; // π/4
    nk_u64_t const negative_zero = 0x8000000000000000ull;
    nk_u64_t const positive_infinity = 0x7FF0000000000000ull;
    nk_u64_t const negative_infinity = 0xFFF0000000000000ull;
    nk_f64_t result = nk_f64_fma_(polynomial, ratio_cubed, ratio);
    result = nk_f64_fma_((nk_f64_t)quadrant, pi_half, result);

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

#if NUMKONG_TARGET_SERIAL
NUMKONG_API nk_f32_t nk_f32_sin_serial(nk_f32_t const angle_radians) { return nk_f32_sin_(angle_radians); }
NUMKONG_API nk_f32_t nk_f32_cos_serial(nk_f32_t const angle_radians) { return nk_f32_cos_(angle_radians); }
NUMKONG_API nk_f32_t nk_f32_atan_serial(nk_f32_t const input) { return nk_f32_atan_(input); }
NUMKONG_API nk_f32_t nk_f32_atan2_serial(nk_f32_t const y_input, nk_f32_t const x_input) {
    return nk_f32_atan2_(y_input, x_input);
}
NUMKONG_API nk_f64_t nk_f64_sin_serial(nk_f64_t const angle_radians) { return nk_f64_sin_(angle_radians); }
NUMKONG_API nk_f64_t nk_f64_cos_serial(nk_f64_t const angle_radians) { return nk_f64_cos_(angle_radians); }
NUMKONG_API nk_f64_t nk_f64_atan_serial(nk_f64_t const input) { return nk_f64_atan_(input); }
NUMKONG_API nk_f64_t nk_f64_atan2_serial(nk_f64_t const y_input, nk_f64_t const x_input) {
    return nk_f64_atan2_(y_input, x_input);
}
#endif // NUMKONG_TARGET_SERIAL

/** Scalar `2^x` via a degree-4 minimax polynomial; no libm. The lower clamp is −125 (not the F32
 *  limit) so the smallest result stays a normal float: a denormal operand in a downstream multiply
 *  costs a ~150-cycle microcode assist on most cores. Shared reference for every fused kernel
 *  needing a sigmoid / SiLU / softmax weight; the SIMD backends match this polynomial to keep
 *  serial and vector paths in agreement. */
NUMKONG_INLINE nk_f32_t nk_f32_exp2_serial_(nk_f32_t x) {
    x = x > 127.0f ? 127.0f : x;
    x = x < -125.0f ? -125.0f : x;
    nk_i32_t whole = (nk_i32_t)(x >= 0 ? x + 0.5f : x - 0.5f);
    nk_f32_t reduced = x - (nk_f32_t)whole; // in [-0.5, 0.5]
    nk_f32_t poly = 9.61812910e-3f;
    poly = poly * reduced + 5.55041087e-2f;
    poly = poly * reduced + 2.40226507e-1f;
    poly = poly * reduced + 6.93147181e-1f;
    poly = poly * reduced + 1.0f;
    nk_fui32_t power;
    power.u = (nk_u32_t)(whole + 127) << 23;
    return poly * power.f;
}

/** Scalar logistic sigmoid `1 / (1 + e^-x)`, built on the shared fast exponent. */
NUMKONG_INLINE nk_f32_t nk_f32_sigmoid_serial_(nk_f32_t x) {
    return 1.0f / (1.0f + nk_f32_exp2_serial_(-x * NUMKONG_F32_LOG2E_));
}

/** Scalar SiLU, or swish, x × sigmoid(x), built on the shared fast exponent. */
NUMKONG_INLINE nk_f32_t nk_f32_silu_serial_(nk_f32_t x) { return x * nk_f32_sigmoid_serial_(x); }

#if defined(__GNUC__) && !defined(__clang__) && NUMKONG_ARCH_ARM64_
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_SCALAR_SERIAL_H
