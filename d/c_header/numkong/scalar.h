/**
 *  @brief SIMD-accelerated Scalar Math Helpers.
 *  @file include/numkong/scalar.h
 *  @author Ash Vardanian
 *  @date March 1, 2026
 *
 *  Provides dispatchable scalar helpers: sqrt, rsqrt, fma, saturating arithmetic,
 *  and ordering. Each ISA file is header-only with
 *  `NK_PUBLIC static inline` implementations; compile-time dispatch selects the
 *  best available backend when `NK_DYNAMIC_DISPATCH` is off.
 *
 *  For hardware architectures:
 *
 *  - Serial: software-emulated (Quake 3 rsqrt, bit-manipulation casts)
 *  - Arm: NEON (sqrt, fma, saturating_add)
 *  - x86: Haswell (sqrt, rsqrt, fma)
 *  - RISC-V: RVV (sqrt, rsqrt, fma, saturating_add via vfrsqrt7 + Newton-Raphson)
 *  - WASM: V128Relaxed (sqrt)
 */
#ifndef NK_SCALAR_H
#define NK_SCALAR_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Scalar square root: `√x`.
 *
 *  @param[in] x The input value.
 *  @return The square root of @p x.
 */
NK_DYNAMIC nk_f32_t nk_f32_sqrt(nk_f32_t x);
/** @copydoc nk_f32_sqrt */
NK_DYNAMIC nk_f64_t nk_f64_sqrt(nk_f64_t x);

/**
 *  @brief Scalar reciprocal square root: `1/√x`.
 *  @sa std::rsqrt, @sa Rust f32::rsqrt
 *
 *  @param[in] x The input value.
 *  @return The reciprocal square root of @p x.
 */
NK_DYNAMIC nk_f32_t nk_f32_rsqrt(nk_f32_t x);
/** @copydoc nk_f32_rsqrt */
NK_DYNAMIC nk_f64_t nk_f64_rsqrt(nk_f64_t x);

/**
 *  @brief Scalar fused multiply-add: `a × b + c`.
 *  @sa std::fma, @sa Rust f32::mul_add
 *
 *  @param[in] a Multiplicand.
 *  @param[in] b Multiplier.
 *  @param[in] c Addend.
 *  @return `a * b + c` computed without intermediate rounding.
 */
NK_DYNAMIC nk_f32_t nk_f32_fma(nk_f32_t a, nk_f32_t b, nk_f32_t c);
/** @copydoc nk_f32_fma */
NK_DYNAMIC nk_f64_t nk_f64_fma(nk_f64_t a, nk_f64_t b, nk_f64_t c);

/** @copydoc nk_f32_sqrt */
NK_DYNAMIC nk_f16_t nk_f16_sqrt(nk_f16_t x);
/** @copydoc nk_f32_rsqrt */
NK_DYNAMIC nk_f16_t nk_f16_rsqrt(nk_f16_t x);
/** @copydoc nk_f32_fma */
NK_DYNAMIC nk_f16_t nk_f16_fma(nk_f16_t a, nk_f16_t b, nk_f16_t c);

/**
 *  @brief Saturating addition clamped to the representable range of the type.
 *
 *  @param[in] a First operand.
 *  @param[in] b Second operand.
 *  @return `clamp(a + b, MIN, MAX)`.
 */
NK_DYNAMIC nk_u8_t nk_u8_saturating_add(nk_u8_t a, nk_u8_t b);
/** @copydoc nk_u8_saturating_add */
NK_DYNAMIC nk_i8_t nk_i8_saturating_add(nk_i8_t a, nk_i8_t b);
/** @copydoc nk_u8_saturating_add */
NK_DYNAMIC nk_u16_t nk_u16_saturating_add(nk_u16_t a, nk_u16_t b);
/** @copydoc nk_u8_saturating_add */
NK_DYNAMIC nk_i16_t nk_i16_saturating_add(nk_i16_t a, nk_i16_t b);
/** @copydoc nk_u8_saturating_add */
NK_DYNAMIC nk_u32_t nk_u32_saturating_add(nk_u32_t a, nk_u32_t b);
/** @copydoc nk_u8_saturating_add */
NK_DYNAMIC nk_i32_t nk_i32_saturating_add(nk_i32_t a, nk_i32_t b);
/** @copydoc nk_u8_saturating_add */
NK_DYNAMIC nk_u64_t nk_u64_saturating_add(nk_u64_t a, nk_u64_t b);
/** @copydoc nk_u8_saturating_add */
NK_DYNAMIC nk_i64_t nk_i64_saturating_add(nk_i64_t a, nk_i64_t b);
/** @copydoc nk_u8_saturating_add */
NK_DYNAMIC nk_i4x2_t nk_i4x2_saturating_add(nk_i4x2_t a, nk_i4x2_t b);
/** @copydoc nk_u8_saturating_add */
NK_DYNAMIC nk_u4x2_t nk_u4x2_saturating_add(nk_u4x2_t a, nk_u4x2_t b);

/**
 *  @brief Saturating multiplication clamped to the representable range of the type.
 *
 *  @param[in] a First operand.
 *  @param[in] b Second operand.
 *  @return `clamp(a * b, MIN, MAX)`.
 */
NK_DYNAMIC nk_u8_t nk_u8_saturating_mul(nk_u8_t a, nk_u8_t b);
/** @copydoc nk_u8_saturating_mul */
NK_DYNAMIC nk_i8_t nk_i8_saturating_mul(nk_i8_t a, nk_i8_t b);
/** @copydoc nk_u8_saturating_mul */
NK_DYNAMIC nk_u16_t nk_u16_saturating_mul(nk_u16_t a, nk_u16_t b);
/** @copydoc nk_u8_saturating_mul */
NK_DYNAMIC nk_i16_t nk_i16_saturating_mul(nk_i16_t a, nk_i16_t b);
/** @copydoc nk_u8_saturating_mul */
NK_DYNAMIC nk_u32_t nk_u32_saturating_mul(nk_u32_t a, nk_u32_t b);
/** @copydoc nk_u8_saturating_mul */
NK_DYNAMIC nk_i32_t nk_i32_saturating_mul(nk_i32_t a, nk_i32_t b);
/** @copydoc nk_u8_saturating_mul */
NK_DYNAMIC nk_u64_t nk_u64_saturating_mul(nk_u64_t a, nk_u64_t b);
/** @copydoc nk_u8_saturating_mul */
NK_DYNAMIC nk_i64_t nk_i64_saturating_mul(nk_i64_t a, nk_i64_t b);
/** @copydoc nk_u8_saturating_mul */
NK_DYNAMIC nk_i4x2_t nk_i4x2_saturating_mul(nk_i4x2_t a, nk_i4x2_t b);
/** @copydoc nk_u8_saturating_mul */
NK_DYNAMIC nk_u4x2_t nk_u4x2_saturating_mul(nk_u4x2_t a, nk_u4x2_t b);

/**
 *  @brief Branchless sign-magnitude ordering for non-native floating-point scalars.
 *  @sa std::strong_order, Rust total_cmp
 *
 *  Uses `mask = -sign; ordered = value ^ mask` — the constant offset cancels in subtraction.
 *  Returns negative if a < b, 0 if equal, positive if a > b.
 *
 *  @param[in] a First operand.
 *  @param[in] b Second operand.
 *  @return Negative if `a < b`, zero if `a == b`, positive if `a > b`.
 *
 *  @note NaN values are ordered at the extremes per IEEE 754 totalOrder
 *  (negative NaN < all finite < positive NaN). Callers requiring NaN-exclusion
 *  semantics must filter NaN before calling.
 */
NK_DYNAMIC int nk_f16_order(nk_f16_t a, nk_f16_t b);
/** @copydoc nk_f16_order */
NK_DYNAMIC int nk_bf16_order(nk_bf16_t a, nk_bf16_t b);
/** @copydoc nk_f16_order */
NK_DYNAMIC int nk_e4m3_order(nk_e4m3_t a, nk_e4m3_t b);
/** @copydoc nk_f16_order */
NK_DYNAMIC int nk_e5m2_order(nk_e5m2_t a, nk_e5m2_t b);
/** @copydoc nk_f16_order */
NK_DYNAMIC int nk_e2m3_order(nk_e2m3_t a, nk_e2m3_t b);
/** @copydoc nk_f16_order */
NK_DYNAMIC int nk_e3m2_order(nk_e3m2_t a, nk_e3m2_t b);

/** @copydoc nk_f32_sqrt */
NK_PUBLIC nk_f32_t nk_f32_sqrt_serial(nk_f32_t x);
/** @copydoc nk_f64_sqrt */
NK_PUBLIC nk_f64_t nk_f64_sqrt_serial(nk_f64_t x);
/** @copydoc nk_f32_rsqrt */
NK_PUBLIC nk_f32_t nk_f32_rsqrt_serial(nk_f32_t x);
/** @copydoc nk_f64_rsqrt */
NK_PUBLIC nk_f64_t nk_f64_rsqrt_serial(nk_f64_t x);
/** @copydoc nk_f32_fma */
NK_PUBLIC nk_f32_t nk_f32_fma_serial(nk_f32_t a, nk_f32_t b, nk_f32_t c);
/** @copydoc nk_f64_fma */
NK_PUBLIC nk_f64_t nk_f64_fma_serial(nk_f64_t a, nk_f64_t b, nk_f64_t c);

/** @copydoc nk_f16_sqrt */
NK_PUBLIC nk_f16_t nk_f16_sqrt_serial(nk_f16_t x);
/** @copydoc nk_f16_rsqrt */
NK_PUBLIC nk_f16_t nk_f16_rsqrt_serial(nk_f16_t x);
/** @copydoc nk_f16_fma */
NK_PUBLIC nk_f16_t nk_f16_fma_serial(nk_f16_t a, nk_f16_t b, nk_f16_t c);

/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_u8_t nk_u8_saturating_add_serial(nk_u8_t a, nk_u8_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_i8_t nk_i8_saturating_add_serial(nk_i8_t a, nk_i8_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_u16_t nk_u16_saturating_add_serial(nk_u16_t a, nk_u16_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_i16_t nk_i16_saturating_add_serial(nk_i16_t a, nk_i16_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_u32_t nk_u32_saturating_add_serial(nk_u32_t a, nk_u32_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_i32_t nk_i32_saturating_add_serial(nk_i32_t a, nk_i32_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_u64_t nk_u64_saturating_add_serial(nk_u64_t a, nk_u64_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_i64_t nk_i64_saturating_add_serial(nk_i64_t a, nk_i64_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_i4x2_t nk_i4x2_saturating_add_serial(nk_i4x2_t a, nk_i4x2_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_u4x2_t nk_u4x2_saturating_add_serial(nk_u4x2_t a, nk_u4x2_t b);

/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_u8_t nk_u8_saturating_mul_serial(nk_u8_t a, nk_u8_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_i8_t nk_i8_saturating_mul_serial(nk_i8_t a, nk_i8_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_u16_t nk_u16_saturating_mul_serial(nk_u16_t a, nk_u16_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_i16_t nk_i16_saturating_mul_serial(nk_i16_t a, nk_i16_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_u32_t nk_u32_saturating_mul_serial(nk_u32_t a, nk_u32_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_i32_t nk_i32_saturating_mul_serial(nk_i32_t a, nk_i32_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_u64_t nk_u64_saturating_mul_serial(nk_u64_t a, nk_u64_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_i64_t nk_i64_saturating_mul_serial(nk_i64_t a, nk_i64_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_i4x2_t nk_i4x2_saturating_mul_serial(nk_i4x2_t a, nk_i4x2_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_u4x2_t nk_u4x2_saturating_mul_serial(nk_u4x2_t a, nk_u4x2_t b);

/** @copydoc nk_f16_order */
NK_PUBLIC int nk_f16_order_serial(nk_f16_t a, nk_f16_t b);
/** @copydoc nk_f16_order */
NK_PUBLIC int nk_bf16_order_serial(nk_bf16_t a, nk_bf16_t b);
/** @copydoc nk_f16_order */
NK_PUBLIC int nk_e4m3_order_serial(nk_e4m3_t a, nk_e4m3_t b);
/** @copydoc nk_f16_order */
NK_PUBLIC int nk_e5m2_order_serial(nk_e5m2_t a, nk_e5m2_t b);
/** @copydoc nk_f16_order */
NK_PUBLIC int nk_e2m3_order_serial(nk_e2m3_t a, nk_e2m3_t b);
/** @copydoc nk_f16_order */
NK_PUBLIC int nk_e3m2_order_serial(nk_e3m2_t a, nk_e3m2_t b);

#if NK_TARGET_NEON
/** @copydoc nk_f32_sqrt */
NK_PUBLIC nk_f32_t nk_f32_sqrt_neon(nk_f32_t x);
/** @copydoc nk_f64_sqrt */
NK_PUBLIC nk_f64_t nk_f64_sqrt_neon(nk_f64_t x);
/** @copydoc nk_f32_rsqrt */
NK_PUBLIC nk_f32_t nk_f32_rsqrt_neon(nk_f32_t x);
/** @copydoc nk_f64_rsqrt */
NK_PUBLIC nk_f64_t nk_f64_rsqrt_neon(nk_f64_t x);
/** @copydoc nk_f32_fma */
NK_PUBLIC nk_f32_t nk_f32_fma_neon(nk_f32_t a, nk_f32_t b, nk_f32_t c);
/** @copydoc nk_f64_fma */
NK_PUBLIC nk_f64_t nk_f64_fma_neon(nk_f64_t a, nk_f64_t b, nk_f64_t c);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_u8_t nk_u8_saturating_add_neon(nk_u8_t a, nk_u8_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_i8_t nk_i8_saturating_add_neon(nk_i8_t a, nk_i8_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_u16_t nk_u16_saturating_add_neon(nk_u16_t a, nk_u16_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_i16_t nk_i16_saturating_add_neon(nk_i16_t a, nk_i16_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_u32_t nk_u32_saturating_add_neon(nk_u32_t a, nk_u32_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_i32_t nk_i32_saturating_add_neon(nk_i32_t a, nk_i32_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_u64_t nk_u64_saturating_add_neon(nk_u64_t a, nk_u64_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_i64_t nk_i64_saturating_add_neon(nk_i64_t a, nk_i64_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_u64_t nk_u64_saturating_mul_neon(nk_u64_t a, nk_u64_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_i64_t nk_i64_saturating_mul_neon(nk_i64_t a, nk_i64_t b);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
/** @copydoc nk_f16_sqrt */
NK_PUBLIC nk_f16_t nk_f16_sqrt_neonhalf(nk_f16_t x);
/** @copydoc nk_f16_rsqrt */
NK_PUBLIC nk_f16_t nk_f16_rsqrt_neonhalf(nk_f16_t x);
/** @copydoc nk_f16_fma */
NK_PUBLIC nk_f16_t nk_f16_fma_neonhalf(nk_f16_t a, nk_f16_t b, nk_f16_t c);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_HASWELL
/** @copydoc nk_f32_sqrt */
NK_PUBLIC nk_f32_t nk_f32_sqrt_haswell(nk_f32_t x);
/** @copydoc nk_f64_sqrt */
NK_PUBLIC nk_f64_t nk_f64_sqrt_haswell(nk_f64_t x);
/** @copydoc nk_f32_rsqrt */
NK_PUBLIC nk_f32_t nk_f32_rsqrt_haswell(nk_f32_t x);
/** @copydoc nk_f64_rsqrt */
NK_PUBLIC nk_f64_t nk_f64_rsqrt_haswell(nk_f64_t x);
/** @copydoc nk_f32_fma */
NK_PUBLIC nk_f32_t nk_f32_fma_haswell(nk_f32_t a, nk_f32_t b, nk_f32_t c);
/** @copydoc nk_f64_fma */
NK_PUBLIC nk_f64_t nk_f64_fma_haswell(nk_f64_t a, nk_f64_t b, nk_f64_t c);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_u8_t nk_u8_saturating_add_haswell(nk_u8_t a, nk_u8_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_i8_t nk_i8_saturating_add_haswell(nk_i8_t a, nk_i8_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_u16_t nk_u16_saturating_add_haswell(nk_u16_t a, nk_u16_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_i16_t nk_i16_saturating_add_haswell(nk_i16_t a, nk_i16_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_u64_t nk_u64_saturating_mul_haswell(nk_u64_t a, nk_u64_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_i64_t nk_i64_saturating_mul_haswell(nk_i64_t a, nk_i64_t b);
/** @copydoc nk_f16_sqrt */
NK_PUBLIC nk_f16_t nk_f16_sqrt_haswell(nk_f16_t x);
/** @copydoc nk_f16_rsqrt */
NK_PUBLIC nk_f16_t nk_f16_rsqrt_haswell(nk_f16_t x);
/** @copydoc nk_f16_fma */
NK_PUBLIC nk_f16_t nk_f16_fma_haswell(nk_f16_t a, nk_f16_t b, nk_f16_t c);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SAPPHIRE
/** @copydoc nk_f16_order */
NK_PUBLIC int nk_f16_order_sapphire(nk_f16_t a, nk_f16_t b);
/** @copydoc nk_f16_sqrt */
NK_PUBLIC nk_f16_t nk_f16_sqrt_sapphire(nk_f16_t x);
/** @copydoc nk_f16_rsqrt */
NK_PUBLIC nk_f16_t nk_f16_rsqrt_sapphire(nk_f16_t x);
/** @copydoc nk_f16_fma */
NK_PUBLIC nk_f16_t nk_f16_fma_sapphire(nk_f16_t a, nk_f16_t b, nk_f16_t c);
#endif // NK_TARGET_SAPPHIRE

#if NK_TARGET_RVV
/** @copydoc nk_f32_sqrt */
NK_PUBLIC nk_f32_t nk_f32_sqrt_rvv(nk_f32_t x);
/** @copydoc nk_f64_sqrt */
NK_PUBLIC nk_f64_t nk_f64_sqrt_rvv(nk_f64_t x);
/** @copydoc nk_f32_rsqrt */
NK_PUBLIC nk_f32_t nk_f32_rsqrt_rvv(nk_f32_t x);
/** @copydoc nk_f64_rsqrt */
NK_PUBLIC nk_f64_t nk_f64_rsqrt_rvv(nk_f64_t x);
/** @copydoc nk_f32_fma */
NK_PUBLIC nk_f32_t nk_f32_fma_rvv(nk_f32_t a, nk_f32_t b, nk_f32_t c);
/** @copydoc nk_f64_fma */
NK_PUBLIC nk_f64_t nk_f64_fma_rvv(nk_f64_t a, nk_f64_t b, nk_f64_t c);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_u8_t nk_u8_saturating_add_rvv(nk_u8_t a, nk_u8_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_i8_t nk_i8_saturating_add_rvv(nk_i8_t a, nk_i8_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_u16_t nk_u16_saturating_add_rvv(nk_u16_t a, nk_u16_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_i16_t nk_i16_saturating_add_rvv(nk_i16_t a, nk_i16_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_u32_t nk_u32_saturating_add_rvv(nk_u32_t a, nk_u32_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_i32_t nk_i32_saturating_add_rvv(nk_i32_t a, nk_i32_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_u64_t nk_u64_saturating_add_rvv(nk_u64_t a, nk_u64_t b);
/** @copydoc nk_u8_saturating_add */
NK_PUBLIC nk_i64_t nk_i64_saturating_add_rvv(nk_i64_t a, nk_i64_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_u8_t nk_u8_saturating_mul_rvv(nk_u8_t a, nk_u8_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_i8_t nk_i8_saturating_mul_rvv(nk_i8_t a, nk_i8_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_u16_t nk_u16_saturating_mul_rvv(nk_u16_t a, nk_u16_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_i16_t nk_i16_saturating_mul_rvv(nk_i16_t a, nk_i16_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_u32_t nk_u32_saturating_mul_rvv(nk_u32_t a, nk_u32_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_i32_t nk_i32_saturating_mul_rvv(nk_i32_t a, nk_i32_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_u64_t nk_u64_saturating_mul_rvv(nk_u64_t a, nk_u64_t b);
/** @copydoc nk_u8_saturating_mul */
NK_PUBLIC nk_i64_t nk_i64_saturating_mul_rvv(nk_i64_t a, nk_i64_t b);
#endif // NK_TARGET_RVV

#if NK_TARGET_V128RELAXED
/** @copydoc nk_f32_sqrt */
NK_PUBLIC nk_f32_t nk_f32_sqrt_v128relaxed(nk_f32_t x);
/** @copydoc nk_f64_sqrt */
NK_PUBLIC nk_f64_t nk_f64_sqrt_v128relaxed(nk_f64_t x);
/** @copydoc nk_f32_rsqrt */
NK_PUBLIC nk_f32_t nk_f32_rsqrt_v128relaxed(nk_f32_t x);
/** @copydoc nk_f64_rsqrt */
NK_PUBLIC nk_f64_t nk_f64_rsqrt_v128relaxed(nk_f64_t x);
/** @copydoc nk_f32_fma */
NK_PUBLIC nk_f32_t nk_f32_fma_v128relaxed(nk_f32_t a, nk_f32_t b, nk_f32_t c);
/** @copydoc nk_f64_fma */
NK_PUBLIC nk_f64_t nk_f64_fma_v128relaxed(nk_f64_t a, nk_f64_t b, nk_f64_t c);
#endif // NK_TARGET_V128RELAXED

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/scalar/serial.h"      // `nk_f32_rsqrt_serial`
#include "numkong/scalar/neon.h"        // `nk_f32_sqrt_neon`
#include "numkong/scalar/neonhalf.h"    // `nk_f16_sqrt_neonhalf`
#include "numkong/scalar/haswell.h"     // `nk_f32_sqrt_haswell`
#include "numkong/scalar/sapphire.h"    // `nk_f16_order_sapphire`
#include "numkong/scalar/rvv.h"         // `nk_f32_rsqrt_rvv`
#include "numkong/scalar/v128relaxed.h" // `nk_f32_sqrt_v128relaxed`

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC nk_f32_t nk_f32_sqrt(nk_f32_t x) {
#if NK_TARGET_HASWELL
    return nk_f32_sqrt_haswell(x);
#elif NK_TARGET_NEON
    return nk_f32_sqrt_neon(x);
#elif NK_TARGET_RVV
    return nk_f32_sqrt_rvv(x);
#elif NK_TARGET_V128RELAXED
    return nk_f32_sqrt_v128relaxed(x);
#else
    return nk_f32_sqrt_serial(x);
#endif
}

NK_PUBLIC nk_f64_t nk_f64_sqrt(nk_f64_t x) {
#if NK_TARGET_HASWELL
    return nk_f64_sqrt_haswell(x);
#elif NK_TARGET_NEON
    return nk_f64_sqrt_neon(x);
#elif NK_TARGET_RVV
    return nk_f64_sqrt_rvv(x);
#elif NK_TARGET_V128RELAXED
    return nk_f64_sqrt_v128relaxed(x);
#else
    return nk_f64_sqrt_serial(x);
#endif
}

NK_PUBLIC nk_f32_t nk_f32_rsqrt(nk_f32_t x) {
#if NK_TARGET_HASWELL
    return nk_f32_rsqrt_haswell(x);
#elif NK_TARGET_NEON
    return nk_f32_rsqrt_neon(x);
#elif NK_TARGET_RVV
    return nk_f32_rsqrt_rvv(x);
#elif NK_TARGET_V128RELAXED
    return nk_f32_rsqrt_v128relaxed(x);
#else
    return nk_f32_rsqrt_serial(x);
#endif
}

NK_PUBLIC nk_f64_t nk_f64_rsqrt(nk_f64_t x) {
#if NK_TARGET_HASWELL
    return nk_f64_rsqrt_haswell(x);
#elif NK_TARGET_NEON
    return nk_f64_rsqrt_neon(x);
#elif NK_TARGET_RVV
    return nk_f64_rsqrt_rvv(x);
#elif NK_TARGET_V128RELAXED
    return nk_f64_rsqrt_v128relaxed(x);
#else
    return nk_f64_rsqrt_serial(x);
#endif
}

NK_PUBLIC nk_f32_t nk_f32_fma(nk_f32_t a, nk_f32_t b, nk_f32_t c) {
#if NK_TARGET_HASWELL
    return nk_f32_fma_haswell(a, b, c);
#elif NK_TARGET_NEON
    return nk_f32_fma_neon(a, b, c);
#elif NK_TARGET_RVV
    return nk_f32_fma_rvv(a, b, c);
#elif NK_TARGET_V128RELAXED
    return nk_f32_fma_v128relaxed(a, b, c);
#else
    return nk_f32_fma_serial(a, b, c);
#endif
}

NK_PUBLIC nk_f64_t nk_f64_fma(nk_f64_t a, nk_f64_t b, nk_f64_t c) {
#if NK_TARGET_HASWELL
    return nk_f64_fma_haswell(a, b, c);
#elif NK_TARGET_NEON
    return nk_f64_fma_neon(a, b, c);
#elif NK_TARGET_RVV
    return nk_f64_fma_rvv(a, b, c);
#elif NK_TARGET_V128RELAXED
    return nk_f64_fma_v128relaxed(a, b, c);
#else
    return nk_f64_fma_serial(a, b, c);
#endif
}

NK_PUBLIC nk_f16_t nk_f16_sqrt(nk_f16_t x) {
#if NK_TARGET_SAPPHIRE
    return nk_f16_sqrt_sapphire(x);
#elif NK_TARGET_NEONHALF
    return nk_f16_sqrt_neonhalf(x);
#elif NK_TARGET_HASWELL
    return nk_f16_sqrt_haswell(x);
#else
    return nk_f16_sqrt_serial(x);
#endif
}

NK_PUBLIC nk_f16_t nk_f16_rsqrt(nk_f16_t x) {
#if NK_TARGET_SAPPHIRE
    return nk_f16_rsqrt_sapphire(x);
#elif NK_TARGET_NEONHALF
    return nk_f16_rsqrt_neonhalf(x);
#elif NK_TARGET_HASWELL
    return nk_f16_rsqrt_haswell(x);
#else
    return nk_f16_rsqrt_serial(x);
#endif
}

NK_PUBLIC nk_f16_t nk_f16_fma(nk_f16_t a, nk_f16_t b, nk_f16_t c) {
#if NK_TARGET_SAPPHIRE
    return nk_f16_fma_sapphire(a, b, c);
#elif NK_TARGET_NEONHALF
    return nk_f16_fma_neonhalf(a, b, c);
#elif NK_TARGET_HASWELL
    return nk_f16_fma_haswell(a, b, c);
#else
    return nk_f16_fma_serial(a, b, c);
#endif
}

NK_PUBLIC nk_u8_t nk_u8_saturating_add(nk_u8_t a, nk_u8_t b) {
#if NK_TARGET_HASWELL
    return nk_u8_saturating_add_haswell(a, b);
#elif NK_TARGET_NEON
    return nk_u8_saturating_add_neon(a, b);
#elif NK_TARGET_RVV
    return nk_u8_saturating_add_rvv(a, b);
#else
    return nk_u8_saturating_add_serial(a, b);
#endif
}
NK_PUBLIC nk_i8_t nk_i8_saturating_add(nk_i8_t a, nk_i8_t b) {
#if NK_TARGET_HASWELL
    return nk_i8_saturating_add_haswell(a, b);
#elif NK_TARGET_NEON
    return nk_i8_saturating_add_neon(a, b);
#elif NK_TARGET_RVV
    return nk_i8_saturating_add_rvv(a, b);
#else
    return nk_i8_saturating_add_serial(a, b);
#endif
}
NK_PUBLIC nk_u16_t nk_u16_saturating_add(nk_u16_t a, nk_u16_t b) {
#if NK_TARGET_HASWELL
    return nk_u16_saturating_add_haswell(a, b);
#elif NK_TARGET_NEON
    return nk_u16_saturating_add_neon(a, b);
#elif NK_TARGET_RVV
    return nk_u16_saturating_add_rvv(a, b);
#else
    return nk_u16_saturating_add_serial(a, b);
#endif
}
NK_PUBLIC nk_i16_t nk_i16_saturating_add(nk_i16_t a, nk_i16_t b) {
#if NK_TARGET_HASWELL
    return nk_i16_saturating_add_haswell(a, b);
#elif NK_TARGET_NEON
    return nk_i16_saturating_add_neon(a, b);
#elif NK_TARGET_RVV
    return nk_i16_saturating_add_rvv(a, b);
#else
    return nk_i16_saturating_add_serial(a, b);
#endif
}
NK_PUBLIC nk_u32_t nk_u32_saturating_add(nk_u32_t a, nk_u32_t b) {
#if NK_TARGET_NEON
    return nk_u32_saturating_add_neon(a, b);
#elif NK_TARGET_RVV
    return nk_u32_saturating_add_rvv(a, b);
#else
    return nk_u32_saturating_add_serial(a, b);
#endif
}
NK_PUBLIC nk_i32_t nk_i32_saturating_add(nk_i32_t a, nk_i32_t b) {
#if NK_TARGET_NEON
    return nk_i32_saturating_add_neon(a, b);
#elif NK_TARGET_RVV
    return nk_i32_saturating_add_rvv(a, b);
#else
    return nk_i32_saturating_add_serial(a, b);
#endif
}
NK_PUBLIC nk_u64_t nk_u64_saturating_add(nk_u64_t a, nk_u64_t b) {
#if NK_TARGET_NEON
    return nk_u64_saturating_add_neon(a, b);
#elif NK_TARGET_RVV
    return nk_u64_saturating_add_rvv(a, b);
#else
    return nk_u64_saturating_add_serial(a, b);
#endif
}
NK_PUBLIC nk_i64_t nk_i64_saturating_add(nk_i64_t a, nk_i64_t b) {
#if NK_TARGET_NEON
    return nk_i64_saturating_add_neon(a, b);
#elif NK_TARGET_RVV
    return nk_i64_saturating_add_rvv(a, b);
#else
    return nk_i64_saturating_add_serial(a, b);
#endif
}
NK_PUBLIC nk_i4x2_t nk_i4x2_saturating_add(nk_i4x2_t a, nk_i4x2_t b) { return nk_i4x2_saturating_add_serial(a, b); }
NK_PUBLIC nk_u4x2_t nk_u4x2_saturating_add(nk_u4x2_t a, nk_u4x2_t b) { return nk_u4x2_saturating_add_serial(a, b); }

NK_PUBLIC nk_u8_t nk_u8_saturating_mul(nk_u8_t a, nk_u8_t b) {
#if NK_TARGET_RVV
    return nk_u8_saturating_mul_rvv(a, b);
#else
    return nk_u8_saturating_mul_serial(a, b);
#endif
}
NK_PUBLIC nk_i8_t nk_i8_saturating_mul(nk_i8_t a, nk_i8_t b) {
#if NK_TARGET_RVV
    return nk_i8_saturating_mul_rvv(a, b);
#else
    return nk_i8_saturating_mul_serial(a, b);
#endif
}
NK_PUBLIC nk_u16_t nk_u16_saturating_mul(nk_u16_t a, nk_u16_t b) {
#if NK_TARGET_RVV
    return nk_u16_saturating_mul_rvv(a, b);
#else
    return nk_u16_saturating_mul_serial(a, b);
#endif
}
NK_PUBLIC nk_i16_t nk_i16_saturating_mul(nk_i16_t a, nk_i16_t b) {
#if NK_TARGET_RVV
    return nk_i16_saturating_mul_rvv(a, b);
#else
    return nk_i16_saturating_mul_serial(a, b);
#endif
}
NK_PUBLIC nk_u32_t nk_u32_saturating_mul(nk_u32_t a, nk_u32_t b) {
#if NK_TARGET_RVV
    return nk_u32_saturating_mul_rvv(a, b);
#else
    return nk_u32_saturating_mul_serial(a, b);
#endif
}
NK_PUBLIC nk_i32_t nk_i32_saturating_mul(nk_i32_t a, nk_i32_t b) {
#if NK_TARGET_RVV
    return nk_i32_saturating_mul_rvv(a, b);
#else
    return nk_i32_saturating_mul_serial(a, b);
#endif
}
NK_PUBLIC nk_u64_t nk_u64_saturating_mul(nk_u64_t a, nk_u64_t b) {
#if NK_TARGET_HASWELL
    return nk_u64_saturating_mul_haswell(a, b);
#elif NK_TARGET_NEON
    return nk_u64_saturating_mul_neon(a, b);
#elif NK_TARGET_RVV
    return nk_u64_saturating_mul_rvv(a, b);
#else
    return nk_u64_saturating_mul_serial(a, b);
#endif
}
NK_PUBLIC nk_i64_t nk_i64_saturating_mul(nk_i64_t a, nk_i64_t b) {
#if NK_TARGET_HASWELL
    return nk_i64_saturating_mul_haswell(a, b);
#elif NK_TARGET_NEON
    return nk_i64_saturating_mul_neon(a, b);
#elif NK_TARGET_RVV
    return nk_i64_saturating_mul_rvv(a, b);
#else
    return nk_i64_saturating_mul_serial(a, b);
#endif
}
NK_PUBLIC nk_i4x2_t nk_i4x2_saturating_mul(nk_i4x2_t a, nk_i4x2_t b) { return nk_i4x2_saturating_mul_serial(a, b); }
NK_PUBLIC nk_u4x2_t nk_u4x2_saturating_mul(nk_u4x2_t a, nk_u4x2_t b) { return nk_u4x2_saturating_mul_serial(a, b); }

NK_PUBLIC int nk_f16_order(nk_f16_t a, nk_f16_t b) {
#if NK_TARGET_SAPPHIRE
    return nk_f16_order_sapphire(a, b);
#else
    return nk_f16_order_serial(a, b);
#endif
}
NK_PUBLIC int nk_bf16_order(nk_bf16_t a, nk_bf16_t b) { return nk_bf16_order_serial(a, b); }
NK_PUBLIC int nk_e4m3_order(nk_e4m3_t a, nk_e4m3_t b) { return nk_e4m3_order_serial(a, b); }
NK_PUBLIC int nk_e5m2_order(nk_e5m2_t a, nk_e5m2_t b) { return nk_e5m2_order_serial(a, b); }
NK_PUBLIC int nk_e2m3_order(nk_e2m3_t a, nk_e2m3_t b) { return nk_e2m3_order_serial(a, b); }
NK_PUBLIC int nk_e3m2_order(nk_e3m2_t a, nk_e3m2_t b) { return nk_e3m2_order_serial(a, b); }

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_SCALAR_H
