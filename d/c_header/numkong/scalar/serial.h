/**
 *  @brief Software-emulated Scalar Math Helpers for SIMD-free CPUs.
 *  @file include/numkong/scalar/serial.h
 *  @author Ash Vardanian
 *  @date March 1, 2026
 *
 *  @sa include/numkong/scalar.h
 *
 *  Uses the Quake 3 fast inverse square root trick with Newton-Raphson refinement.
 *  Three iterations for f32 (~34.9 correct bits), four for f64 (~69.3 correct bits).
 */
#ifndef NK_SCALAR_SERIAL_H
#define NK_SCALAR_SERIAL_H

#include "numkong/types.h"
#include "numkong/cast/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC nk_f32_t nk_f32_rsqrt_serial(nk_f32_t number) {
    nk_fui32_t conv;
    conv.f = number;
    conv.u = 0x5F375A86 - (conv.u >> 1);
    nk_f32_t y = conv.f;
    y = y * (1.5f - 0.5f * number * y * y);
    y = y * (1.5f - 0.5f * number * y * y);
    y = y * (1.5f - 0.5f * number * y * y);
    return y;
}

NK_PUBLIC nk_f32_t nk_f32_sqrt_serial(nk_f32_t number) { return number > 0 ? number * nk_f32_rsqrt_serial(number) : 0; }

NK_PUBLIC nk_f64_t nk_f64_rsqrt_serial(nk_f64_t number) {
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

NK_PUBLIC nk_f64_t nk_f64_sqrt_serial(nk_f64_t number) { return number > 0 ? number * nk_f64_rsqrt_serial(number) : 0; }

NK_PUBLIC nk_f16_t nk_f16_sqrt_serial(nk_f16_t x) {
    nk_f32_t x_f32;
    nk_f16_to_f32_serial(&x, &x_f32);
    x_f32 = nk_f32_sqrt_serial(x_f32);
    nk_f16_t result;
    nk_f32_to_f16_serial(&x_f32, &result);
    return result;
}

NK_PUBLIC nk_f16_t nk_f16_rsqrt_serial(nk_f16_t x) {
    nk_f32_t x_f32;
    nk_f16_to_f32_serial(&x, &x_f32);
    x_f32 = nk_f32_rsqrt_serial(x_f32);
    nk_f16_t result;
    nk_f32_to_f16_serial(&x_f32, &result);
    return result;
}

/**
 *  @brief Software FMA (Fused Multiply-Add) emulation for f64.
 *  Computes (multiplicand * multiplier + addend) with improved precision
 *  using Dekker's error-free multiplication and Knuth's TwoSum.
 *  @sa std::fma, @sa Rust f64::mul_add
 */
NK_PUBLIC nk_f64_t nk_f64_fma_serial(nk_f64_t multiplicand, nk_f64_t multiplier, nk_f64_t addend) {
    nk_f64_t product = multiplicand * multiplier;
    // Dekker splitting: break each operand into non-overlapping high and low halves
    nk_f64_t const dekker_split = 134217729.0; // 2^27 + 1 for double precision
    nk_f64_t multiplicand_hi = dekker_split * multiplicand;
    nk_f64_t multiplicand_lo = multiplicand - (multiplicand_hi - (multiplicand_hi - multiplicand));
    multiplicand_hi = multiplicand_hi - (multiplicand_hi - multiplicand);
    nk_f64_t multiplier_hi = dekker_split * multiplier;
    nk_f64_t multiplier_lo = multiplier - (multiplier_hi - (multiplier_hi - multiplier));
    multiplier_hi = multiplier_hi - (multiplier_hi - multiplier);
    // Exact multiplication error from the four cross-products
    nk_f64_t product_error = ((multiplicand_hi * multiplier_hi - product) + multiplicand_hi * multiplier_lo +
                              multiplicand_lo * multiplier_hi) +
                             multiplicand_lo * multiplier_lo;
    // Knuth TwoSum: add the addend with error tracking
    nk_f64_t result = product + addend;
    nk_f64_t addend_recovered = result - product;
    nk_f64_t product_recovered = result - addend_recovered;
    nk_f64_t addition_error = (product - product_recovered) + (addend - addend_recovered);
    return result + (product_error + addition_error);
}

/**
 *  @brief Software FMA (Fused Multiply-Add) emulation for f32.
 *  Computes (multiplicand * multiplier + addend) with improved precision
 *  using Dekker's error-free multiplication and Knuth's TwoSum.
 *  @sa std::fma, @sa Rust f32::mul_add
 */
NK_PUBLIC nk_f32_t nk_f32_fma_serial(nk_f32_t multiplicand, nk_f32_t multiplier, nk_f32_t addend) {
    nk_f32_t product = multiplicand * multiplier;
    // Dekker splitting: break each operand into non-overlapping high and low halves
    nk_f32_t const dekker_split = 4097.0f; // 2^12 + 1 for single precision
    nk_f32_t multiplicand_hi = dekker_split * multiplicand;
    nk_f32_t multiplicand_lo = multiplicand - (multiplicand_hi - (multiplicand_hi - multiplicand));
    multiplicand_hi = multiplicand_hi - (multiplicand_hi - multiplicand);
    nk_f32_t multiplier_hi = dekker_split * multiplier;
    nk_f32_t multiplier_lo = multiplier - (multiplier_hi - (multiplier_hi - multiplier));
    multiplier_hi = multiplier_hi - (multiplier_hi - multiplier);
    // Exact multiplication error from the four cross-products
    nk_f32_t product_error = ((multiplicand_hi * multiplier_hi - product) + multiplicand_hi * multiplier_lo +
                              multiplicand_lo * multiplier_hi) +
                             multiplicand_lo * multiplier_lo;
    // Knuth TwoSum: add the addend with error tracking
    nk_f32_t result = product + addend;
    nk_f32_t addend_recovered = result - product;
    nk_f32_t product_recovered = result - addend_recovered;
    nk_f32_t addition_error = (product - product_recovered) + (addend - addend_recovered);
    return result + (product_error + addition_error);
}

/**
 *  @brief Scalar Dot2 accumulator: sum += a * b with error compensation.
 *  Uses TwoProd (via FMA) and TwoSum error-free transformations.
 *  @see Ogita, T., Rump, S.M., Oishi, S. (2005). "Accurate Sum and Dot Product"
 */
NK_INTERNAL void nk_f64_dot2_(nk_f64_t *sum, nk_f64_t *compensation, nk_f64_t a, nk_f64_t b) {
    nk_f64_t product = a * b;
    nk_f64_t product_error = nk_f64_fma_serial(a, b, -product);
    nk_f64_t running_sum = *sum + product;
    nk_f64_t recovered_addend = running_sum - *sum;
    nk_f64_t sum_error = (*sum - (running_sum - recovered_addend)) + (product - recovered_addend);
    *sum = running_sum;
    *compensation += sum_error + product_error;
}

NK_PUBLIC nk_f16_t nk_f16_fma_serial(nk_f16_t a, nk_f16_t b, nk_f16_t c) {
    nk_f32_t a_f32, b_f32, c_f32;
    nk_f16_to_f32_serial(&a, &a_f32);
    nk_f16_to_f32_serial(&b, &b_f32);
    nk_f16_to_f32_serial(&c, &c_f32);
    nk_f32_t result_f32 = nk_f32_fma_serial(a_f32, b_f32, c_f32);
    nk_f16_t result;
    nk_f32_to_f16_serial(&result_f32, &result);
    return result;
}

NK_PUBLIC nk_u8_t nk_u8_saturating_add_serial(nk_u8_t a, nk_u8_t b) {
    nk_u16_t result = (nk_u16_t)a + (nk_u16_t)b;
    return (result > 255u) ? (nk_u8_t)255u : (nk_u8_t)result;
}
NK_PUBLIC nk_u16_t nk_u16_saturating_add_serial(nk_u16_t a, nk_u16_t b) {
    nk_u32_t result = (nk_u32_t)a + (nk_u32_t)b;
    return (result > 65535u) ? (nk_u16_t)65535u : (nk_u16_t)result;
}
NK_PUBLIC nk_u32_t nk_u32_saturating_add_serial(nk_u32_t a, nk_u32_t b) {
    nk_u64_t result = (nk_u64_t)a + (nk_u64_t)b;
    return (result > 4294967295u) ? (nk_u32_t)4294967295u : (nk_u32_t)result;
}
NK_PUBLIC nk_u64_t nk_u64_saturating_add_serial(nk_u64_t a, nk_u64_t b) {
    return (a + b < a) ? 18446744073709551615ull : (a + b);
}
NK_PUBLIC nk_i8_t nk_i8_saturating_add_serial(nk_i8_t a, nk_i8_t b) {
    nk_i16_t result = (nk_i16_t)a + (nk_i16_t)b;
    return (result > 127) ? 127 : (result < -128 ? -128 : result);
}
NK_PUBLIC nk_i16_t nk_i16_saturating_add_serial(nk_i16_t a, nk_i16_t b) {
    nk_i32_t result = (nk_i32_t)a + (nk_i32_t)b;
    return (result > 32767) ? 32767 : (result < -32768 ? -32768 : result);
}
NK_PUBLIC nk_i32_t nk_i32_saturating_add_serial(nk_i32_t a, nk_i32_t b) {
    nk_i64_t result = (nk_i64_t)a + (nk_i64_t)b;
    return (result > 2147483647ll) ? 2147483647ll : (result < -2147483648ll ? -2147483648ll : (nk_i32_t)result);
}
NK_PUBLIC nk_i64_t nk_i64_saturating_add_serial(nk_i64_t a, nk_i64_t b) {
    //? We can't just write `-9223372036854775808ll`, even though it's the smallest signed 64-bit value.
    //? The compiler will complain about the number being too large for the type, as it will process the
    //? constant and the sign separately. So we use the same hint that compilers use to define the `INT64_MIN`.
    if ((b > 0) && (a > (9223372036854775807ll) - b)) return 9223372036854775807ll;
    if ((b < 0) && (a < (-9223372036854775807ll - 1ll) - b)) return -9223372036854775807ll - 1ll;
    return a + b;
}

NK_PUBLIC nk_u8_t nk_u8_saturating_mul_serial(nk_u8_t a, nk_u8_t b) {
    nk_u16_t result = (nk_u16_t)a * (nk_u16_t)b;
    return (result > 255) ? 255 : (nk_u8_t)result;
}

NK_PUBLIC nk_u16_t nk_u16_saturating_mul_serial(nk_u16_t a, nk_u16_t b) {
    nk_u32_t result = (nk_u32_t)a * (nk_u32_t)b;
    return (result > 65535) ? 65535 : (nk_u16_t)result;
}

NK_PUBLIC nk_u32_t nk_u32_saturating_mul_serial(nk_u32_t a, nk_u32_t b) {
    nk_u64_t result = (nk_u64_t)a * (nk_u64_t)b;
    return (result > 4294967295u) ? 4294967295u : (nk_u32_t)result;
}

NK_PUBLIC nk_u64_t nk_u64_saturating_mul_serial(nk_u64_t a, nk_u64_t b) {
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

NK_PUBLIC nk_i8_t nk_i8_saturating_mul_serial(nk_i8_t a, nk_i8_t b) {
    nk_i16_t result = (nk_i16_t)a * (nk_i16_t)b;
    return (result > 127) ? 127 : (result < -128 ? -128 : (nk_i8_t)result);
}

NK_PUBLIC nk_i16_t nk_i16_saturating_mul_serial(nk_i16_t a, nk_i16_t b) {
    nk_i32_t result = (nk_i32_t)a * (nk_i32_t)b;
    return (result > 32767) ? 32767 : (result < -32768 ? -32768 : (nk_i16_t)result);
}

NK_PUBLIC nk_i32_t nk_i32_saturating_mul_serial(nk_i32_t a, nk_i32_t b) {
    nk_i64_t result = (nk_i64_t)a * (nk_i64_t)b;
    return (result > 2147483647ll) ? 2147483647ll : (result < -2147483648ll ? -2147483648ll : (nk_i32_t)result);
}

NK_PUBLIC nk_i64_t nk_i64_saturating_mul_serial(nk_i64_t a, nk_i64_t b) {
    int sign = ((a < 0) ^ (b < 0)) ? -1 : 1; // Track the sign of the result

    // Take absolute values for easy multiplication and overflow detection
    nk_u64_t abs_a = (a < 0) ? -(nk_u64_t)a : (nk_u64_t)a;
    nk_u64_t abs_b = (b < 0) ? -(nk_u64_t)b : (nk_u64_t)b;

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
    return (sign < 0) ? -((nk_i64_t)result) : (nk_i64_t)result;
}

NK_PUBLIC nk_i4x2_t nk_i4x2_saturating_add_serial(nk_i4x2_t a, nk_i4x2_t b) {
    nk_i8_t low = nk_i4x2_low_(a) + nk_i4x2_low_(b);
    nk_i8_t high = nk_i4x2_high_(a) + nk_i4x2_high_(b);
    low = (low > 7) ? 7 : (low < -8 ? -8 : low);
    high = (high > 7) ? 7 : (high < -8 ? -8 : high);
    return (nk_i4x2_t)((low & 0x0F) | ((high & 0x0F) << 4));
}
NK_PUBLIC nk_u4x2_t nk_u4x2_saturating_add_serial(nk_u4x2_t a, nk_u4x2_t b) {
    nk_u8_t low = nk_u4x2_low_(a) + nk_u4x2_low_(b);
    nk_u8_t high = nk_u4x2_high_(a) + nk_u4x2_high_(b);
    low = (low > 15) ? 15 : low;
    high = (high > 15) ? 15 : high;
    return (nk_u4x2_t)((low & 0x0F) | ((high & 0x0F) << 4));
}
NK_PUBLIC nk_i4x2_t nk_i4x2_saturating_mul_serial(nk_i4x2_t a, nk_i4x2_t b) {
    nk_i8_t low = nk_i4x2_low_(a) * nk_i4x2_low_(b);
    nk_i8_t high = nk_i4x2_high_(a) * nk_i4x2_high_(b);
    low = (low > 7) ? 7 : (low < -8 ? -8 : low);
    high = (high > 7) ? 7 : (high < -8 ? -8 : high);
    return (nk_i4x2_t)((low & 0x0F) | ((high & 0x0F) << 4));
}
NK_PUBLIC nk_u4x2_t nk_u4x2_saturating_mul_serial(nk_u4x2_t a, nk_u4x2_t b) {
    nk_u8_t low = nk_u4x2_low_(a) * nk_u4x2_low_(b);
    nk_u8_t high = nk_u4x2_high_(a) * nk_u4x2_high_(b);
    low = (low > 15) ? 15 : low;
    high = (high > 15) ? 15 : high;
    return (nk_u4x2_t)((low & 0x0F) | ((high & 0x0F) << 4));
}

NK_PUBLIC int nk_e4m3_order_serial(nk_e4m3_t a, nk_e4m3_t b) {
    int sign_a = a >> 7, sign_b = b >> 7;
    return (a ^ -sign_a) - (b ^ -sign_b);
}
NK_PUBLIC int nk_e5m2_order_serial(nk_e5m2_t a, nk_e5m2_t b) {
    int sign_a = a >> 7, sign_b = b >> 7;
    return (a ^ -sign_a) - (b ^ -sign_b);
}

NK_PUBLIC int nk_e2m3_order_serial(nk_e2m3_t a, nk_e2m3_t b) {
    int value_a = a & 0x3F, value_b = b & 0x3F;
    int sign_a = value_a >> 5, sign_b = value_b >> 5;
    return (value_a ^ -sign_a) - (value_b ^ -sign_b);
}
NK_PUBLIC int nk_e3m2_order_serial(nk_e3m2_t a, nk_e3m2_t b) {
    int value_a = a & 0x3F, value_b = b & 0x3F;
    int sign_a = value_a >> 5, sign_b = value_b >> 5;
    return (value_a ^ -sign_a) - (value_b ^ -sign_b);
}

NK_PUBLIC int nk_bf16_order_serial(nk_bf16_t a, nk_bf16_t b) {
    nk_fui16_t a_fui, b_fui;
    a_fui.bf = a, b_fui.bf = b;
    int sign_a = a_fui.u >> 15, sign_b = b_fui.u >> 15;
    return ((int)a_fui.u ^ -sign_a) - ((int)b_fui.u ^ -sign_b);
}

NK_PUBLIC int nk_f16_order_serial(nk_f16_t a, nk_f16_t b) {
    nk_fui16_t a_fui, b_fui;
    a_fui.f = a, b_fui.f = b;
    int sign_a = a_fui.u >> 15, sign_b = b_fui.u >> 15;
    return ((int)a_fui.u ^ -sign_a) - ((int)b_fui.u ^ -sign_b);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_SCALAR_SERIAL_H
