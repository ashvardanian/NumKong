/**
 *  @file include/numkong/scalar.h
 *  @author Ash Vardanian
 *  @date November 20, 2024
 *  @brief SIMD-accelerated scalar math helpers.
 *
 *  Provides dispatchable scalar helpers: sqrt, rsqrt, fma, saturating arithmetic, and ordering,
 *  each behind a dispatch point like @c nk_f32_sqrt_best, which runs the best CPU capability of its
 *  capability mask and falls back to serial. Trigonometry of one value, like
 *  @c nk_f64_sin_serial, has only the serial kernel.
 *
 *  For hardware architectures:
 *
 *  - Serial: software-emulated (Quake 3 rsqrt, bit-manipulation casts)
 *  - Arm: NEON (sqrt, fma, saturating_add)
 *  - x86: Haswell (sqrt, rsqrt, fma)
 *  - RISC-V: RVV (sqrt, rsqrt, fma, saturating_add via vfrsqrt7 + Newton-Raphson)
 *  - WASM: V128 (sqrt, rsqrt), V128Relaxed (fma)
 *  - Power: VSX (sqrt, rsqrt, fma)
 *  - LoongArch: LASX (sqrt, rsqrt)
 */
#ifndef NUMKONG_SCALAR_H
#define NUMKONG_SCALAR_H

#include "numkong/capabilities.h" // `nk_capability_kernels_t`, `nk_kernel_pick_`

#if defined(__cplusplus)
extern "C" {
#endif

/** Scalar kernels of one shape, cast back from @c nk_kernel_punned_t by their dispatch points. */
typedef nk_f32_t (*nk_f32_unary_punned_t)(nk_f32_t);
typedef nk_f64_t (*nk_f64_unary_punned_t)(nk_f64_t);
typedef nk_f16_t (*nk_f16_unary_punned_t)(nk_f16_t);
typedef nk_f32_t (*nk_f32_ternary_punned_t)(nk_f32_t, nk_f32_t, nk_f32_t);
typedef nk_f64_t (*nk_f64_ternary_punned_t)(nk_f64_t, nk_f64_t, nk_f64_t);
typedef nk_f16_t (*nk_f16_ternary_punned_t)(nk_f16_t, nk_f16_t, nk_f16_t);
typedef nk_u8_t (*nk_u8_binary_punned_t)(nk_u8_t, nk_u8_t);
typedef nk_i8_t (*nk_i8_binary_punned_t)(nk_i8_t, nk_i8_t);
typedef nk_u16_t (*nk_u16_binary_punned_t)(nk_u16_t, nk_u16_t);
typedef nk_i16_t (*nk_i16_binary_punned_t)(nk_i16_t, nk_i16_t);
typedef nk_u32_t (*nk_u32_binary_punned_t)(nk_u32_t, nk_u32_t);
typedef nk_i32_t (*nk_i32_binary_punned_t)(nk_i32_t, nk_i32_t);
typedef nk_u64_t (*nk_u64_binary_punned_t)(nk_u64_t, nk_u64_t);
typedef nk_i64_t (*nk_i64_binary_punned_t)(nk_i64_t, nk_i64_t);
typedef int (*nk_f16_compare_punned_t)(nk_f16_t, nk_f16_t);
typedef int (*nk_bf16_compare_punned_t)(nk_bf16_t, nk_bf16_t);
typedef int (*nk_u8_compare_punned_t)(nk_u8_t, nk_u8_t);

/**
 *  @brief Scalar square root, √x.
 *
 *  @param[in] x The input value.
 *  @param[in] capabilities The CPU's capabilities, like @c nk_cpu_capabilities_enabled reports;
 *      serial always stays.
 *  @return The square root of @p x.
 */
NUMKONG_API_RUNTIME nk_f32_t nk_f32_sqrt_best(nk_f32_t x, nk_capability_t capabilities);
/** @copydoc nk_f32_sqrt_best */
NUMKONG_API_RUNTIME nk_f64_t nk_f64_sqrt_best(nk_f64_t x, nk_capability_t capabilities);

/**
 *  @brief Scalar reciprocal square root, 1 / √x.
 *  @sa C++ @c std::rsqrt
 *  @sa Rust @c f32::rsqrt
 *
 *  @param[in] x The input value.
 *  @param[in] capabilities The CPU's capabilities, like @c nk_cpu_capabilities_enabled reports;
 *      serial always stays.
 *  @return The reciprocal square root of @p x.
 */
NUMKONG_API_RUNTIME nk_f32_t nk_f32_rsqrt_best(nk_f32_t x, nk_capability_t capabilities);
/** @copydoc nk_f32_rsqrt_best */
NUMKONG_API_RUNTIME nk_f64_t nk_f64_rsqrt_best(nk_f64_t x, nk_capability_t capabilities);

/**
 *  @brief Scalar fused multiply-add, a × b + c.
 *  @sa C++ @c std::fma
 *  @sa Rust @c f32::mul_add
 *
 *  @param[in] a Multiplicand.
 *  @param[in] b Multiplier.
 *  @param[in] c Addend.
 *  @param[in] capabilities The CPU's capabilities, like @c nk_cpu_capabilities_enabled reports;
 *      serial always stays.
 *  @return a × b + c computed without intermediate rounding.
 */
NUMKONG_API_RUNTIME nk_f32_t nk_f32_fma_best(nk_f32_t a, nk_f32_t b, nk_f32_t c, nk_capability_t capabilities);
/** @copydoc nk_f32_fma_best */
NUMKONG_API_RUNTIME nk_f64_t nk_f64_fma_best(nk_f64_t a, nk_f64_t b, nk_f64_t c, nk_capability_t capabilities);

/** @copydoc nk_f32_sqrt_best */
NUMKONG_API_RUNTIME nk_f16_t nk_f16_sqrt_best(nk_f16_t x, nk_capability_t capabilities);
/** @copydoc nk_f32_rsqrt_best */
NUMKONG_API_RUNTIME nk_f16_t nk_f16_rsqrt_best(nk_f16_t x, nk_capability_t capabilities);
/** @copydoc nk_f32_fma_best */
NUMKONG_API_RUNTIME nk_f16_t nk_f16_fma_best(nk_f16_t a, nk_f16_t b, nk_f16_t c, nk_capability_t capabilities);

/**
 *  @brief Saturating addition clamped to the representable range of the type.
 *
 *  @param[in] a First operand.
 *  @param[in] b Second operand.
 *  @param[in] capabilities The CPU's capabilities, like @c nk_cpu_capabilities_enabled reports;
 *      serial always stays.
 *  @return `clamp(a + b, MIN, MAX)`.
 */
NUMKONG_API_RUNTIME nk_u8_t nk_u8_saturating_add_best(nk_u8_t a, nk_u8_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_RUNTIME nk_i8_t nk_i8_saturating_add_best(nk_i8_t a, nk_i8_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_RUNTIME nk_u16_t nk_u16_saturating_add_best(nk_u16_t a, nk_u16_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_RUNTIME nk_i16_t nk_i16_saturating_add_best(nk_i16_t a, nk_i16_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_RUNTIME nk_u32_t nk_u32_saturating_add_best(nk_u32_t a, nk_u32_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_RUNTIME nk_i32_t nk_i32_saturating_add_best(nk_i32_t a, nk_i32_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_RUNTIME nk_u64_t nk_u64_saturating_add_best(nk_u64_t a, nk_u64_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_RUNTIME nk_i64_t nk_i64_saturating_add_best(nk_i64_t a, nk_i64_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_RUNTIME nk_i4x2_t nk_i4x2_saturating_add_best(nk_i4x2_t a, nk_i4x2_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_RUNTIME nk_u4x2_t nk_u4x2_saturating_add_best(nk_u4x2_t a, nk_u4x2_t b, nk_capability_t capabilities);

/**
 *  @brief Saturating multiplication clamped to the representable range of the type.
 *
 *  @param[in] a First operand.
 *  @param[in] b Second operand.
 *  @param[in] capabilities The CPU's capabilities, like @c nk_cpu_capabilities_enabled reports;
 *      serial always stays.
 *  @return `clamp(a * b, MIN, MAX)`.
 */
NUMKONG_API_RUNTIME nk_u8_t nk_u8_saturating_mul_best(nk_u8_t a, nk_u8_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_RUNTIME nk_i8_t nk_i8_saturating_mul_best(nk_i8_t a, nk_i8_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_RUNTIME nk_u16_t nk_u16_saturating_mul_best(nk_u16_t a, nk_u16_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_RUNTIME nk_i16_t nk_i16_saturating_mul_best(nk_i16_t a, nk_i16_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_RUNTIME nk_u32_t nk_u32_saturating_mul_best(nk_u32_t a, nk_u32_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_RUNTIME nk_i32_t nk_i32_saturating_mul_best(nk_i32_t a, nk_i32_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_RUNTIME nk_u64_t nk_u64_saturating_mul_best(nk_u64_t a, nk_u64_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_RUNTIME nk_i64_t nk_i64_saturating_mul_best(nk_i64_t a, nk_i64_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_RUNTIME nk_i4x2_t nk_i4x2_saturating_mul_best(nk_i4x2_t a, nk_i4x2_t b, nk_capability_t capabilities);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_RUNTIME nk_u4x2_t nk_u4x2_saturating_mul_best(nk_u4x2_t a, nk_u4x2_t b, nk_capability_t capabilities);

/**
 *  @brief Branchless sign-magnitude ordering for non-native floating-point scalars.
 *  @sa std::strong_order, Rust total_cmp
 *
 *  Uses `mask = -sign; ordered = value ^ mask` — the constant offset cancels in subtraction.
 *  Returns negative if a < b, 0 if equal, positive if a > b.
 *
 *  @param[in] a First operand.
 *  @param[in] b Second operand.
 *  @param[in] capabilities The CPU's capabilities, like @c nk_cpu_capabilities_enabled reports;
 *      serial always stays.
 *  @return Negative if `a < b`, zero if `a == b`, positive if `a > b`.
 *
 *  @note NaN values are ordered at the extremes per IEEE 754 totalOrder (negative NaN < all finite
 *      < positive NaN). Callers requiring NaN-exclusion semantics must filter NaN before calling.
 */
NUMKONG_API_RUNTIME int nk_f16_order_best(nk_f16_t a, nk_f16_t b, nk_capability_t capabilities);
/** @copydoc nk_f16_order_best */
NUMKONG_API_RUNTIME int nk_bf16_order_best(nk_bf16_t a, nk_bf16_t b, nk_capability_t capabilities);
/** @copydoc nk_f16_order_best */
NUMKONG_API_RUNTIME int nk_e4m3_order_best(nk_e4m3_t a, nk_e4m3_t b, nk_capability_t capabilities);
/** @copydoc nk_f16_order_best */
NUMKONG_API_RUNTIME int nk_e5m2_order_best(nk_e5m2_t a, nk_e5m2_t b, nk_capability_t capabilities);
/** @copydoc nk_f16_order_best */
NUMKONG_API_RUNTIME int nk_e2m3_order_best(nk_e2m3_t a, nk_e2m3_t b, nk_capability_t capabilities);
/** @copydoc nk_f16_order_best */
NUMKONG_API_RUNTIME int nk_e3m2_order_best(nk_e3m2_t a, nk_e3m2_t b, nk_capability_t capabilities);

/*  Sine, cosine and arc-tangents of one value in radians, the serial twins of the batched
 *  @c nk_trig_sin_f32_best and its siblings. */

/**
 *  @brief Approximates the sine of an angle in radians within @b 3-ULP error on [-2π, 2π].
 *
 *  Based on @c xfastsinf_u3500 in the SLEEF library.
 *
 *  @param[in] angle_radians The input angle in radians.
 *  @return The approximate sine of the input angle in [-1, 1] range.
 */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_sin_serial(nk_f32_t angle_radians);

/**
 *  @brief Approximates the cosine of an angle in radians within @b 3-ULP error on [-2π, 2π].
 *
 *  Based on @c xfastcosf_u3500 in the SLEEF library.
 *
 *  @param[in] angle_radians The input angle in radians.
 *  @return The approximate cosine of the input angle in [-1, 1] range.
 */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_cos_serial(nk_f32_t angle_radians);

/**
 *  @brief Computes the arc-tangent of a value with @b 0-ULP error bound.
 *
 *  Based on @c xatanf in the SLEEF library.
 *
 *  @param[in] input The input value.
 *  @return The arc-tangent of the input value in [-π/2, π/2] radians range.
 */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_atan_serial(nk_f32_t input);

/**
 *  @brief Computes the arc-tangent of (y/x) with @b 0-ULP error bound.
 *
 *  Based on @c xatan2f in the SLEEF library.
 *
 *  @param[in] y_input The input sine value.
 *  @param[in] x_input The input cosine value.
 *  @return The arc-tangent of @p y_input / @p x_input in [-π, π] radians range.
 */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_atan2_serial(nk_f32_t y_input, nk_f32_t x_input);

/**
 *  @brief Computes the sine of the given angle in radians with @b 0-ULP error bound in [-2π, 2π].
 *
 *  Based on @c xsin in the SLEEF library.
 *
 *  @param[in] angle_radians The input angle in radians.
 *  @return The approximate sine of the input angle.
 */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_sin_serial(nk_f64_t angle_radians);

/**
 *  @brief Computes the cosine of the given angle in radians with @b 0-ULP error bound in [-2π, 2π].
 *
 *  Based on @c xcos in the SLEEF library.
 *
 *  @param[in] angle_radians The input angle in radians.
 *  @return The approximate cosine of the input angle in [-1, 1] range.
 */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_cos_serial(nk_f64_t angle_radians);

/**
 *  @brief Computes the arc-tangent of a value with @b 0-ULP error bound.
 *
 *  Based on @c xatan in the SLEEF library.
 *
 *  @param[in] input The input value.
 *  @return The arc-tangent of the input value in [-π/2, π/2] radians range.
 */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_atan_serial(nk_f64_t input);

/**
 *  @brief Computes the arc-tangent of (y/x) with @b 0-ULP error bound.
 *
 *  Based on @c xatan2 in the SLEEF library.
 *
 *  @param[in] y_input The input sine value.
 *  @param[in] x_input The input cosine value.
 *  @return The arc-tangent of @p y_input / @p x_input in [-π, π] radians range.
 */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_atan2_serial(nk_f64_t y_input, nk_f64_t x_input);

/** @copydoc nk_f32_sqrt_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_sqrt_serial(nk_f32_t x);
/** @copydoc nk_f64_sqrt_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_sqrt_serial(nk_f64_t x);
/** @copydoc nk_f32_rsqrt_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_rsqrt_serial(nk_f32_t x);
/** @copydoc nk_f64_rsqrt_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_rsqrt_serial(nk_f64_t x);

/**
 *  @copydoc nk_f32_fma_best
 *  @note Emulates the fused rounding with Dekker's error-free product and Knuth's TwoSum.
 */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_fma_serial(nk_f32_t a, nk_f32_t b, nk_f32_t c);

/**
 *  @copydoc nk_f64_fma_best
 *  @note Emulates the fused rounding with Dekker's error-free product and Knuth's TwoSum.
 */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_fma_serial(nk_f64_t a, nk_f64_t b, nk_f64_t c);

/** @copydoc nk_f16_sqrt_best */
NUMKONG_API_COMPTIME nk_f16_t nk_f16_sqrt_serial(nk_f16_t x);
/** @copydoc nk_f16_rsqrt_best */
NUMKONG_API_COMPTIME nk_f16_t nk_f16_rsqrt_serial(nk_f16_t x);
/** @copydoc nk_f16_fma_best */
NUMKONG_API_COMPTIME nk_f16_t nk_f16_fma_serial(nk_f16_t a, nk_f16_t b, nk_f16_t c);

/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_u8_t nk_u8_saturating_add_serial(nk_u8_t a, nk_u8_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_i8_t nk_i8_saturating_add_serial(nk_i8_t a, nk_i8_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_u16_t nk_u16_saturating_add_serial(nk_u16_t a, nk_u16_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_i16_t nk_i16_saturating_add_serial(nk_i16_t a, nk_i16_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_u32_t nk_u32_saturating_add_serial(nk_u32_t a, nk_u32_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_i32_t nk_i32_saturating_add_serial(nk_i32_t a, nk_i32_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_u64_t nk_u64_saturating_add_serial(nk_u64_t a, nk_u64_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_i64_t nk_i64_saturating_add_serial(nk_i64_t a, nk_i64_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_i4x2_t nk_i4x2_saturating_add_serial(nk_i4x2_t a, nk_i4x2_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_u4x2_t nk_u4x2_saturating_add_serial(nk_u4x2_t a, nk_u4x2_t b);

/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_u8_t nk_u8_saturating_mul_serial(nk_u8_t a, nk_u8_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_i8_t nk_i8_saturating_mul_serial(nk_i8_t a, nk_i8_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_u16_t nk_u16_saturating_mul_serial(nk_u16_t a, nk_u16_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_i16_t nk_i16_saturating_mul_serial(nk_i16_t a, nk_i16_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_u32_t nk_u32_saturating_mul_serial(nk_u32_t a, nk_u32_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_i32_t nk_i32_saturating_mul_serial(nk_i32_t a, nk_i32_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_u64_t nk_u64_saturating_mul_serial(nk_u64_t a, nk_u64_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_i64_t nk_i64_saturating_mul_serial(nk_i64_t a, nk_i64_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_i4x2_t nk_i4x2_saturating_mul_serial(nk_i4x2_t a, nk_i4x2_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_u4x2_t nk_u4x2_saturating_mul_serial(nk_u4x2_t a, nk_u4x2_t b);

/** @copydoc nk_f16_order_best */
NUMKONG_API_COMPTIME int nk_f16_order_serial(nk_f16_t a, nk_f16_t b);
/** @copydoc nk_f16_order_best */
NUMKONG_API_COMPTIME int nk_bf16_order_serial(nk_bf16_t a, nk_bf16_t b);
/** @copydoc nk_f16_order_best */
NUMKONG_API_COMPTIME int nk_e4m3_order_serial(nk_e4m3_t a, nk_e4m3_t b);
/** @copydoc nk_f16_order_best */
NUMKONG_API_COMPTIME int nk_e5m2_order_serial(nk_e5m2_t a, nk_e5m2_t b);
/** @copydoc nk_f16_order_best */
NUMKONG_API_COMPTIME int nk_e2m3_order_serial(nk_e2m3_t a, nk_e2m3_t b);
/** @copydoc nk_f16_order_best */
NUMKONG_API_COMPTIME int nk_e3m2_order_serial(nk_e3m2_t a, nk_e3m2_t b);

#if NUMKONG_TARGET_NEON
/** @copydoc nk_f32_sqrt_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_sqrt_neon(nk_f32_t x);
/** @copydoc nk_f64_sqrt_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_sqrt_neon(nk_f64_t x);
/** @copydoc nk_f32_rsqrt_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_rsqrt_neon(nk_f32_t x);
/** @copydoc nk_f64_rsqrt_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_rsqrt_neon(nk_f64_t x);
/** @copydoc nk_f32_fma_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_fma_neon(nk_f32_t a, nk_f32_t b, nk_f32_t c);
/** @copydoc nk_f64_fma_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_fma_neon(nk_f64_t a, nk_f64_t b, nk_f64_t c);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_u8_t nk_u8_saturating_add_neon(nk_u8_t a, nk_u8_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_i8_t nk_i8_saturating_add_neon(nk_i8_t a, nk_i8_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_u16_t nk_u16_saturating_add_neon(nk_u16_t a, nk_u16_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_i16_t nk_i16_saturating_add_neon(nk_i16_t a, nk_i16_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_u32_t nk_u32_saturating_add_neon(nk_u32_t a, nk_u32_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_i32_t nk_i32_saturating_add_neon(nk_i32_t a, nk_i32_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_u64_t nk_u64_saturating_add_neon(nk_u64_t a, nk_u64_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_i64_t nk_i64_saturating_add_neon(nk_i64_t a, nk_i64_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_u64_t nk_u64_saturating_mul_neon(nk_u64_t a, nk_u64_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_i64_t nk_i64_saturating_mul_neon(nk_i64_t a, nk_i64_t b);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONHALF
/** @copydoc nk_f16_sqrt_best */
NUMKONG_API_COMPTIME nk_f16_t nk_f16_sqrt_neonhalf(nk_f16_t x);
/** @copydoc nk_f16_rsqrt_best */
NUMKONG_API_COMPTIME nk_f16_t nk_f16_rsqrt_neonhalf(nk_f16_t x);
/** @copydoc nk_f16_fma_best */
NUMKONG_API_COMPTIME nk_f16_t nk_f16_fma_neonhalf(nk_f16_t a, nk_f16_t b, nk_f16_t c);
#endif // NUMKONG_TARGET_NEONHALF

#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_f32_sqrt_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_sqrt_haswell(nk_f32_t x);
/** @copydoc nk_f64_sqrt_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_sqrt_haswell(nk_f64_t x);
/** @copydoc nk_f32_rsqrt_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_rsqrt_haswell(nk_f32_t x);
/** @copydoc nk_f64_rsqrt_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_rsqrt_haswell(nk_f64_t x);
/** @copydoc nk_f32_fma_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_fma_haswell(nk_f32_t a, nk_f32_t b, nk_f32_t c);
/** @copydoc nk_f64_fma_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_fma_haswell(nk_f64_t a, nk_f64_t b, nk_f64_t c);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_u8_t nk_u8_saturating_add_haswell(nk_u8_t a, nk_u8_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_i8_t nk_i8_saturating_add_haswell(nk_i8_t a, nk_i8_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_u16_t nk_u16_saturating_add_haswell(nk_u16_t a, nk_u16_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_i16_t nk_i16_saturating_add_haswell(nk_i16_t a, nk_i16_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_u64_t nk_u64_saturating_mul_haswell(nk_u64_t a, nk_u64_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_i64_t nk_i64_saturating_mul_haswell(nk_i64_t a, nk_i64_t b);
/** @copydoc nk_f16_sqrt_best */
NUMKONG_API_COMPTIME nk_f16_t nk_f16_sqrt_haswell(nk_f16_t x);
/** @copydoc nk_f16_rsqrt_best */
NUMKONG_API_COMPTIME nk_f16_t nk_f16_rsqrt_haswell(nk_f16_t x);
/** @copydoc nk_f16_fma_best */
NUMKONG_API_COMPTIME nk_f16_t nk_f16_fma_haswell(nk_f16_t a, nk_f16_t b, nk_f16_t c);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SAPPHIRE
/** @copydoc nk_f16_order_best */
NUMKONG_API_COMPTIME int nk_f16_order_sapphire(nk_f16_t a, nk_f16_t b);
/** @copydoc nk_f16_sqrt_best */
NUMKONG_API_COMPTIME nk_f16_t nk_f16_sqrt_sapphire(nk_f16_t x);
/** @copydoc nk_f16_rsqrt_best */
NUMKONG_API_COMPTIME nk_f16_t nk_f16_rsqrt_sapphire(nk_f16_t x);
/** @copydoc nk_f16_fma_best */
NUMKONG_API_COMPTIME nk_f16_t nk_f16_fma_sapphire(nk_f16_t a, nk_f16_t b, nk_f16_t c);
#endif // NUMKONG_TARGET_SAPPHIRE

#if NUMKONG_TARGET_RVV
/** @copydoc nk_f32_sqrt_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_sqrt_rvv(nk_f32_t x);
/** @copydoc nk_f64_sqrt_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_sqrt_rvv(nk_f64_t x);
/** @copydoc nk_f32_rsqrt_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_rsqrt_rvv(nk_f32_t x);
/** @copydoc nk_f64_rsqrt_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_rsqrt_rvv(nk_f64_t x);
/** @copydoc nk_f32_fma_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_fma_rvv(nk_f32_t a, nk_f32_t b, nk_f32_t c);
/** @copydoc nk_f64_fma_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_fma_rvv(nk_f64_t a, nk_f64_t b, nk_f64_t c);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_u8_t nk_u8_saturating_add_rvv(nk_u8_t a, nk_u8_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_i8_t nk_i8_saturating_add_rvv(nk_i8_t a, nk_i8_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_u16_t nk_u16_saturating_add_rvv(nk_u16_t a, nk_u16_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_i16_t nk_i16_saturating_add_rvv(nk_i16_t a, nk_i16_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_u32_t nk_u32_saturating_add_rvv(nk_u32_t a, nk_u32_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_i32_t nk_i32_saturating_add_rvv(nk_i32_t a, nk_i32_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_u64_t nk_u64_saturating_add_rvv(nk_u64_t a, nk_u64_t b);
/** @copydoc nk_u8_saturating_add_best */
NUMKONG_API_COMPTIME nk_i64_t nk_i64_saturating_add_rvv(nk_i64_t a, nk_i64_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_u8_t nk_u8_saturating_mul_rvv(nk_u8_t a, nk_u8_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_i8_t nk_i8_saturating_mul_rvv(nk_i8_t a, nk_i8_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_u16_t nk_u16_saturating_mul_rvv(nk_u16_t a, nk_u16_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_i16_t nk_i16_saturating_mul_rvv(nk_i16_t a, nk_i16_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_u32_t nk_u32_saturating_mul_rvv(nk_u32_t a, nk_u32_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_i32_t nk_i32_saturating_mul_rvv(nk_i32_t a, nk_i32_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_u64_t nk_u64_saturating_mul_rvv(nk_u64_t a, nk_u64_t b);
/** @copydoc nk_u8_saturating_mul_best */
NUMKONG_API_COMPTIME nk_i64_t nk_i64_saturating_mul_rvv(nk_i64_t a, nk_i64_t b);
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128
/** @copydoc nk_f32_sqrt_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_sqrt_v128(nk_f32_t x);
/** @copydoc nk_f64_sqrt_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_sqrt_v128(nk_f64_t x);
/** @copydoc nk_f32_rsqrt_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_rsqrt_v128(nk_f32_t x);
/** @copydoc nk_f64_rsqrt_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_rsqrt_v128(nk_f64_t x);
#endif // NUMKONG_TARGET_V128

#if NUMKONG_TARGET_V128RELAXED
/** @copydoc nk_f32_fma_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_fma_v128relaxed(nk_f32_t a, nk_f32_t b, nk_f32_t c);
/** @copydoc nk_f64_fma_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_fma_v128relaxed(nk_f64_t a, nk_f64_t b, nk_f64_t c);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_POWERVSX
/** @copydoc nk_f32_sqrt_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_sqrt_powervsx(nk_f32_t x);
/** @copydoc nk_f64_sqrt_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_sqrt_powervsx(nk_f64_t x);
/** @copydoc nk_f32_rsqrt_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_rsqrt_powervsx(nk_f32_t x);
/** @copydoc nk_f64_rsqrt_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_rsqrt_powervsx(nk_f64_t x);
/** @copydoc nk_f32_fma_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_fma_powervsx(nk_f32_t a, nk_f32_t b, nk_f32_t c);
/** @copydoc nk_f64_fma_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_fma_powervsx(nk_f64_t a, nk_f64_t b, nk_f64_t c);
#endif // NUMKONG_TARGET_POWERVSX

#if NUMKONG_TARGET_LOONGSONASX
/** @copydoc nk_f32_sqrt_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_sqrt_loongsonasx(nk_f32_t x);
/** @copydoc nk_f64_sqrt_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_sqrt_loongsonasx(nk_f64_t x);
/** @copydoc nk_f32_rsqrt_best */
NUMKONG_API_COMPTIME nk_f32_t nk_f32_rsqrt_loongsonasx(nk_f32_t x);
/** @copydoc nk_f64_rsqrt_best */
NUMKONG_API_COMPTIME nk_f64_t nk_f64_rsqrt_loongsonasx(nk_f64_t x);
#endif // NUMKONG_TARGET_LOONGSONASX

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/scalar/serial.h"      // `nk_f32_rsqrt_serial`
#include "numkong/scalar/neon.h"        // `nk_f32_sqrt_neon`
#include "numkong/scalar/neonhalf.h"    // `nk_f16_sqrt_neonhalf`
#include "numkong/scalar/haswell.h"     // `nk_f32_sqrt_haswell`
#include "numkong/scalar/sapphire.h"    // `nk_f16_order_sapphire`
#include "numkong/scalar/rvv.h"         // `nk_f32_rsqrt_rvv`
#include "numkong/scalar/powervsx.h"    // `nk_f32_sqrt_powervsx`
#include "numkong/scalar/loongsonasx.h" // `nk_f32_sqrt_loongsonasx`
#include "numkong/scalar/v128.h"        // `nk_f32_sqrt_v128`
#include "numkong/scalar/v128relaxed.h" // `nk_f32_fma_v128relaxed`

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_f32_sqrt_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_f32_sqrt_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_f32_sqrt_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f32_sqrt_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_f32_sqrt_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_f32_sqrt_v128,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_f32_sqrt_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_f32_sqrt_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_f64_sqrt_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_f64_sqrt_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_f64_sqrt_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f64_sqrt_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_f64_sqrt_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_f64_sqrt_v128,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_f64_sqrt_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_f64_sqrt_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_f32_rsqrt_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_f32_rsqrt_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_f32_rsqrt_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f32_rsqrt_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_f32_rsqrt_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_f32_rsqrt_v128,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_f32_rsqrt_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_f32_rsqrt_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_f64_rsqrt_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_f64_rsqrt_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_f64_rsqrt_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f64_rsqrt_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_f64_rsqrt_rvv,
#endif
#if NUMKONG_TARGET_V128
        (nk_kernel_punned_t)&nk_f64_rsqrt_v128,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_f64_rsqrt_powervsx,
#endif
#if NUMKONG_TARGET_LOONGSONASX
        (nk_kernel_punned_t)&nk_f64_rsqrt_loongsonasx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128_k * NUMKONG_TARGET_V128 |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX | nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_f32_fma_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_f32_fma_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_f32_fma_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f32_fma_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_f32_fma_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_f32_fma_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_f32_fma_powervsx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_f64_fma_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_f64_fma_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_f64_fma_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f64_fma_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_f64_fma_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_f64_fma_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_f64_fma_powervsx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED |
             nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_f16_sqrt_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_f16_sqrt_serial,
#if NUMKONG_TARGET_NEONHALF
        (nk_kernel_punned_t)&nk_f16_sqrt_neonhalf,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f16_sqrt_haswell,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_f16_sqrt_sapphire,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonhalf_k * NUMKONG_TARGET_NEONHALF | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_f16_rsqrt_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_f16_rsqrt_serial,
#if NUMKONG_TARGET_NEONHALF
        (nk_kernel_punned_t)&nk_f16_rsqrt_neonhalf,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f16_rsqrt_haswell,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_f16_rsqrt_sapphire,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonhalf_k * NUMKONG_TARGET_NEONHALF | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_f16_fma_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_f16_fma_serial,
#if NUMKONG_TARGET_NEONHALF
        (nk_kernel_punned_t)&nk_f16_fma_neonhalf,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f16_fma_haswell,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_f16_fma_sapphire,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonhalf_k * NUMKONG_TARGET_NEONHALF | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_u8_saturating_add_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_u8_saturating_add_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_u8_saturating_add_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_u8_saturating_add_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_u8_saturating_add_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_i8_saturating_add_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_i8_saturating_add_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_i8_saturating_add_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_i8_saturating_add_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_i8_saturating_add_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_u16_saturating_add_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_u16_saturating_add_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_u16_saturating_add_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_u16_saturating_add_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_u16_saturating_add_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_i16_saturating_add_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_i16_saturating_add_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_i16_saturating_add_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_i16_saturating_add_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_i16_saturating_add_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_u32_saturating_add_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_u32_saturating_add_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_u32_saturating_add_neon,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_u32_saturating_add_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_i32_saturating_add_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_i32_saturating_add_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_i32_saturating_add_neon,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_i32_saturating_add_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_u64_saturating_add_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_u64_saturating_add_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_u64_saturating_add_neon,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_u64_saturating_add_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_i64_saturating_add_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_i64_saturating_add_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_i64_saturating_add_neon,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_i64_saturating_add_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_i4x2_saturating_add_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_i4x2_saturating_add_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_u4x2_saturating_add_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_u4x2_saturating_add_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_u8_saturating_mul_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_u8_saturating_mul_serial,
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_u8_saturating_mul_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_i8_saturating_mul_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_i8_saturating_mul_serial,
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_i8_saturating_mul_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_u16_saturating_mul_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_u16_saturating_mul_serial,
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_u16_saturating_mul_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_i16_saturating_mul_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_i16_saturating_mul_serial,
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_i16_saturating_mul_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_u32_saturating_mul_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_u32_saturating_mul_serial,
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_u32_saturating_mul_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_i32_saturating_mul_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_i32_saturating_mul_serial,
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_i32_saturating_mul_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_rvv_k * NUMKONG_TARGET_RVV, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_u64_saturating_mul_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_u64_saturating_mul_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_u64_saturating_mul_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_u64_saturating_mul_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_u64_saturating_mul_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_i64_saturating_mul_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_i64_saturating_mul_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_i64_saturating_mul_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_i64_saturating_mul_haswell,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_i64_saturating_mul_rvv,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV,
         cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_i4x2_saturating_mul_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_i4x2_saturating_mul_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_u4x2_saturating_mul_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_u4x2_saturating_mul_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_f16_order_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_f16_order_serial,
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_f16_order_sapphire,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_bf16_order_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_bf16_order_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_e4m3_order_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_e4m3_order_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_e5m2_order_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_e5m2_order_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_e2m3_order_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_e2m3_order_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

NUMKONG_HELPER_INLINE nk_capability_kernels_t const *nk_e3m2_order_capabilities_(void) {
    static nk_kernel_punned_t const cpu[] = {
        (nk_kernel_punned_t)&nk_e3m2_order_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
        {0, NUMKONG_NULL},
    };
    return lists;
}

#if !NUMKONG_RUNTIME_DISPATCH

NUMKONG_API_COMPTIME nk_f32_t nk_f32_sqrt_best(nk_f32_t x, nk_capability_t capabilities) {
    return ((nk_f32_unary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_f32_sqrt_capabilities_()))(x);
}

NUMKONG_API_COMPTIME nk_f64_t nk_f64_sqrt_best(nk_f64_t x, nk_capability_t capabilities) {
    return ((nk_f64_unary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_f64_sqrt_capabilities_()))(x);
}

NUMKONG_API_COMPTIME nk_f32_t nk_f32_rsqrt_best(nk_f32_t x, nk_capability_t capabilities) {
    return ((nk_f32_unary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_f32_rsqrt_capabilities_()))(x);
}

NUMKONG_API_COMPTIME nk_f64_t nk_f64_rsqrt_best(nk_f64_t x, nk_capability_t capabilities) {
    return ((nk_f64_unary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_f64_rsqrt_capabilities_()))(x);
}

NUMKONG_API_COMPTIME nk_f32_t nk_f32_fma_best(nk_f32_t a, nk_f32_t b, nk_f32_t c, nk_capability_t capabilities) {
    return ((nk_f32_ternary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                     nk_f32_fma_capabilities_()))(a, b, c);
}

NUMKONG_API_COMPTIME nk_f64_t nk_f64_fma_best(nk_f64_t a, nk_f64_t b, nk_f64_t c, nk_capability_t capabilities) {
    return ((nk_f64_ternary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                     nk_f64_fma_capabilities_()))(a, b, c);
}

NUMKONG_API_COMPTIME nk_f16_t nk_f16_sqrt_best(nk_f16_t x, nk_capability_t capabilities) {
    return ((nk_f16_unary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_f16_sqrt_capabilities_()))(x);
}

NUMKONG_API_COMPTIME nk_f16_t nk_f16_rsqrt_best(nk_f16_t x, nk_capability_t capabilities) {
    return ((nk_f16_unary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_f16_rsqrt_capabilities_()))(x);
}

NUMKONG_API_COMPTIME nk_f16_t nk_f16_fma_best(nk_f16_t a, nk_f16_t b, nk_f16_t c, nk_capability_t capabilities) {
    return ((nk_f16_ternary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                     nk_f16_fma_capabilities_()))(a, b, c);
}

NUMKONG_API_COMPTIME nk_u8_t nk_u8_saturating_add_best(nk_u8_t a, nk_u8_t b, nk_capability_t capabilities) {
    return ((nk_u8_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_u8_saturating_add_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_i8_t nk_i8_saturating_add_best(nk_i8_t a, nk_i8_t b, nk_capability_t capabilities) {
    return ((nk_i8_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_i8_saturating_add_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_u16_t nk_u16_saturating_add_best(nk_u16_t a, nk_u16_t b, nk_capability_t capabilities) {
    return ((nk_u16_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_u16_saturating_add_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_i16_t nk_i16_saturating_add_best(nk_i16_t a, nk_i16_t b, nk_capability_t capabilities) {
    return ((nk_i16_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_i16_saturating_add_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_u32_t nk_u32_saturating_add_best(nk_u32_t a, nk_u32_t b, nk_capability_t capabilities) {
    return ((nk_u32_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_u32_saturating_add_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_i32_t nk_i32_saturating_add_best(nk_i32_t a, nk_i32_t b, nk_capability_t capabilities) {
    return ((nk_i32_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_i32_saturating_add_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_u64_t nk_u64_saturating_add_best(nk_u64_t a, nk_u64_t b, nk_capability_t capabilities) {
    return ((nk_u64_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_u64_saturating_add_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_i64_t nk_i64_saturating_add_best(nk_i64_t a, nk_i64_t b, nk_capability_t capabilities) {
    return ((nk_i64_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_i64_saturating_add_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_i4x2_t nk_i4x2_saturating_add_best(nk_i4x2_t a, nk_i4x2_t b, nk_capability_t capabilities) {
    return ((nk_u8_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_i4x2_saturating_add_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_u4x2_t nk_u4x2_saturating_add_best(nk_u4x2_t a, nk_u4x2_t b, nk_capability_t capabilities) {
    return ((nk_u8_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_u4x2_saturating_add_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_u8_t nk_u8_saturating_mul_best(nk_u8_t a, nk_u8_t b, nk_capability_t capabilities) {
    return ((nk_u8_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_u8_saturating_mul_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_i8_t nk_i8_saturating_mul_best(nk_i8_t a, nk_i8_t b, nk_capability_t capabilities) {
    return ((nk_i8_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_i8_saturating_mul_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_u16_t nk_u16_saturating_mul_best(nk_u16_t a, nk_u16_t b, nk_capability_t capabilities) {
    return ((nk_u16_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_u16_saturating_mul_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_i16_t nk_i16_saturating_mul_best(nk_i16_t a, nk_i16_t b, nk_capability_t capabilities) {
    return ((nk_i16_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_i16_saturating_mul_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_u32_t nk_u32_saturating_mul_best(nk_u32_t a, nk_u32_t b, nk_capability_t capabilities) {
    return ((nk_u32_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_u32_saturating_mul_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_i32_t nk_i32_saturating_mul_best(nk_i32_t a, nk_i32_t b, nk_capability_t capabilities) {
    return ((nk_i32_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_i32_saturating_mul_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_u64_t nk_u64_saturating_mul_best(nk_u64_t a, nk_u64_t b, nk_capability_t capabilities) {
    return ((nk_u64_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_u64_saturating_mul_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_i64_t nk_i64_saturating_mul_best(nk_i64_t a, nk_i64_t b, nk_capability_t capabilities) {
    return ((nk_i64_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_i64_saturating_mul_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_i4x2_t nk_i4x2_saturating_mul_best(nk_i4x2_t a, nk_i4x2_t b, nk_capability_t capabilities) {
    return ((nk_u8_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_i4x2_saturating_mul_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME nk_u4x2_t nk_u4x2_saturating_mul_best(nk_u4x2_t a, nk_u4x2_t b, nk_capability_t capabilities) {
    return ((nk_u8_binary_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_u4x2_saturating_mul_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME int nk_f16_order_best(nk_f16_t a, nk_f16_t b, nk_capability_t capabilities) {
    return ((nk_f16_compare_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                     nk_f16_order_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME int nk_bf16_order_best(nk_bf16_t a, nk_bf16_t b, nk_capability_t capabilities) {
    return ((nk_bf16_compare_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                      nk_bf16_order_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME int nk_e4m3_order_best(nk_e4m3_t a, nk_e4m3_t b, nk_capability_t capabilities) {
    return ((nk_u8_compare_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_e4m3_order_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME int nk_e5m2_order_best(nk_e5m2_t a, nk_e5m2_t b, nk_capability_t capabilities) {
    return ((nk_u8_compare_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_e5m2_order_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME int nk_e2m3_order_best(nk_e2m3_t a, nk_e2m3_t b, nk_capability_t capabilities) {
    return ((nk_u8_compare_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_e2m3_order_capabilities_()))(a, b);
}

NUMKONG_API_COMPTIME int nk_e3m2_order_best(nk_e3m2_t a, nk_e3m2_t b, nk_capability_t capabilities) {
    return ((nk_u8_compare_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                    nk_e3m2_order_capabilities_()))(a, b);
}

#endif // !NUMKONG_RUNTIME_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_SCALAR_H
