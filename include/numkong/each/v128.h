/**
 *  @brief SIMD-accelerated Elementwise Arithmetic for WASM.
 *  @file include/numkong/each/v128.h
 *  @author Ash Vardanian
 *  @date September 12, 2026
 *
 *  @sa include/numkong/each.h
 *
 *  Provides WASM SIMD128 implementations of elementwise operations:
 *  - Sum: result[i] = a[i] + b[i]
 *
 *  For dtypes: f32, bf16, i8, u8
 */
#ifndef NK_EACH_V128_H
#define NK_EACH_V128_H

#if NK_TARGET_V128

#include "numkong/types.h"
#include "numkong/cast/serial.h"
#include "numkong/cast/v128.h" // `nk_bf16x4_to_f32x4_v128_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

#pragma region F32 Floats

NK_API_COMPTIME void nk_each_sum_f32_v128(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        v128_t a_f32x4 = wasm_v128_load(a + i);
        v128_t b_f32x4 = wasm_v128_load(b + i);
        wasm_v128_store(result + i, wasm_f32x4_add(a_f32x4, b_f32x4));
    }
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

#pragma endregion F32 Floats
#pragma region BF16 Floats

NK_API_COMPTIME void nk_each_sum_bf16_v128(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        nk_b64_vec_t a_bf16_vec, b_bf16_vec;
        nk_load_b64_serial_(a + i, &a_bf16_vec);
        nk_load_b64_serial_(b + i, &b_bf16_vec);
        nk_b128_vec_t a_f32_vec = nk_bf16x4_to_f32x4_v128_(a_bf16_vec);
        nk_b128_vec_t b_f32_vec = nk_bf16x4_to_f32x4_v128_(b_bf16_vec);
        nk_b128_vec_t result_f32_vec;
        result_f32_vec.v128 = wasm_f32x4_add(a_f32_vec.v128, b_f32_vec.v128);
        nk_b64_vec_t result_bf16_vec = nk_f32x4_to_bf16x4_v128_(result_f32_vec);
        nk_store_b64_serial_(&result_bf16_vec, result + i);
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_bf16_to_f32_serial(a + i, &ai);
        nk_bf16_to_f32_serial(b + i, &bi);
        nk_f32_t sum = ai + bi;
        nk_f32_to_bf16_serial(&sum, result + i);
    }
}

#pragma endregion BF16 Floats
#pragma region I8 Integers

NK_API_COMPTIME void nk_each_sum_i8_v128(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        v128_t a_i8x16 = wasm_v128_load(a + i);
        v128_t b_i8x16 = wasm_v128_load(b + i);
        wasm_v128_store(result + i, wasm_i8x16_add_sat(a_i8x16, b_i8x16));
    }
    for (; i < n; ++i) {
        nk_f32_t sum = (nk_f32_t)a[i] + b[i];
        nk_f32_to_i8_serial(&sum, result + i);
    }
}

#pragma endregion I8 Integers
#pragma region U8 Integers

NK_API_COMPTIME void nk_each_sum_u8_v128(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result) {
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        v128_t a_u8x16 = wasm_v128_load(a + i);
        v128_t b_u8x16 = wasm_v128_load(b + i);
        wasm_v128_store(result + i, wasm_u8x16_add_sat(a_u8x16, b_u8x16));
    }
    for (; i < n; ++i) {
        nk_f32_t sum = (nk_f32_t)a[i] + b[i];
        nk_f32_to_u8_serial(&sum, result + i);
    }
}

#pragma endregion U8 Integers

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128
#endif // NK_EACH_V128_H
