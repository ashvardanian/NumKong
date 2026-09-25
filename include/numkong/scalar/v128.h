/**
 *  @file include/numkong/scalar/v128.h
 *  @author Ash Vardanian
 *  @date September 12, 2026
 *  @brief SIMD-accelerated square roots and reciprocal square roots for WASM.
 *
 *  @sa include/numkong/scalar.h
 */
#ifndef NUMKONG_SCALAR_V128_H
#define NUMKONG_SCALAR_V128_H

#if NUMKONG_TARGET_V128

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

NUMKONG_API_COMPTIME nk_f32_t nk_f32_sqrt_v128(nk_f32_t x) {
    return wasm_f32x4_extract_lane(wasm_f32x4_sqrt(wasm_f32x4_splat(x)), 0);
}
NUMKONG_API_COMPTIME nk_f64_t nk_f64_sqrt_v128(nk_f64_t x) {
    return wasm_f64x2_extract_lane(wasm_f64x2_sqrt(wasm_f64x2_splat(x)), 0);
}
NUMKONG_API_COMPTIME nk_f32_t nk_f32_rsqrt_v128(nk_f32_t x) {
    v128_t sqrt_f32x4 = wasm_f32x4_sqrt(wasm_f32x4_splat(x));
    return wasm_f32x4_extract_lane(wasm_f32x4_div(wasm_f32x4_splat(1.0f), sqrt_f32x4), 0);
}
NUMKONG_API_COMPTIME nk_f64_t nk_f64_rsqrt_v128(nk_f64_t x) {
    v128_t sqrt_f64x2 = wasm_f64x2_sqrt(wasm_f64x2_splat(x));
    return wasm_f64x2_extract_lane(wasm_f64x2_div(wasm_f64x2_splat(1.0), sqrt_f64x2), 0);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_V128
#endif // NUMKONG_SCALAR_V128_H
