/**
 *  @brief Relaxed-SIMD Scalar Math Helpers for WASM: fused multiply-add.
 *  @file include/numkong/scalar/v128relaxed.h
 *  @author Ash Vardanian
 *  @date March 1, 2026
 *
 *  @sa include/numkong/scalar.h
 *  @sa include/numkong/scalar/v128.h for the square roots every SIMD128 engine runs.
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

NK_API_COMPTIME nk_f32_t nk_f32_fma_v128relaxed(nk_f32_t a, nk_f32_t b, nk_f32_t c) {
    v128_t result_f32x4 = wasm_f32x4_relaxed_madd(wasm_f32x4_splat(a), wasm_f32x4_splat(b), wasm_f32x4_splat(c));
    return wasm_f32x4_extract_lane(result_f32x4, 0);
}
NK_API_COMPTIME nk_f64_t nk_f64_fma_v128relaxed(nk_f64_t a, nk_f64_t b, nk_f64_t c) {
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
