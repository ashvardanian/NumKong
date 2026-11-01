/**
 *  @brief SIMD-accelerated Scalar Math Helpers for WASM.
 *  @file include/numkong/scalar/v128relaxed.h
 *  @author Ash Vardanian
 *  @date March 1, 2026
 *
 *  @sa include/numkong/scalar.h
 */
#ifndef NK_SCALAR_V128RELAXED_H
#define NK_SCALAR_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

NK_PUBLIC nk_f32_t nk_f32_sqrt_v128relaxed(nk_f32_t x) {
    return wasm_f32x4_extract_lane(wasm_f32x4_sqrt(wasm_f32x4_splat(x)), 0);
}
NK_PUBLIC nk_f64_t nk_f64_sqrt_v128relaxed(nk_f64_t x) {
    return wasm_f64x2_extract_lane(wasm_f64x2_sqrt(wasm_f64x2_splat(x)), 0);
}
NK_PUBLIC nk_f32_t nk_f32_rsqrt_v128relaxed(nk_f32_t x) {
    v128_t sqrt_f32x4 = wasm_f32x4_sqrt(wasm_f32x4_splat(x));
    return wasm_f32x4_extract_lane(wasm_f32x4_div(wasm_f32x4_splat(1.0f), sqrt_f32x4), 0);
}
NK_PUBLIC nk_f64_t nk_f64_rsqrt_v128relaxed(nk_f64_t x) {
    v128_t sqrt_f64x2 = wasm_f64x2_sqrt(wasm_f64x2_splat(x));
    return wasm_f64x2_extract_lane(wasm_f64x2_div(wasm_f64x2_splat(1.0), sqrt_f64x2), 0);
}
NK_PUBLIC nk_f32_t nk_f32_fma_v128relaxed(nk_f32_t a, nk_f32_t b, nk_f32_t c) {
    v128_t result_f32x4 = wasm_f32x4_relaxed_madd(wasm_f32x4_splat(a), wasm_f32x4_splat(b), wasm_f32x4_splat(c));
    return wasm_f32x4_extract_lane(result_f32x4, 0);
}
NK_PUBLIC nk_f64_t nk_f64_fma_v128relaxed(nk_f64_t a, nk_f64_t b, nk_f64_t c) {
    v128_t result_f64x2 = wasm_f64x2_relaxed_madd(wasm_f64x2_splat(a), wasm_f64x2_splat(b), wasm_f64x2_splat(c));
    return wasm_f64x2_extract_lane(result_f64x2, 0);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_SCALAR_V128RELAXED_H
