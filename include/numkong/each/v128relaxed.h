/**
 *  @brief SIMD-accelerated Elementwise Arithmetic for WASM.
 *  @file include/numkong/each/v128relaxed.h
 *  @author Ash Vardanian
 *  @date April 3, 2026
 *
 *  @sa include/numkong/each.h
 *
 *  Provides WASM relaxed-SIMD implementations of elementwise operations:
 *  - Sum: result[i] = a[i] + b[i]
 *  - Scale: result[i] = α·a[i] + β
 *  - Blend: result[i] = α·a[i] + β·b[i]
 *  - FMA: result[i] = α·a[i]·b[i] + β·c[i]
 *
 *  For dtypes: f32, f16, bf16, i8, u8
 */
#ifndef NK_EACH_V128RELAXED_H
#define NK_EACH_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/cast/serial.h"
#include "numkong/cast/v128relaxed.h"
#include "numkong/reduce/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

#pragma region F32 Floats

NK_API_COMPTIME void nk_each_sum_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        v128_t a_f32x4 = wasm_v128_load(a + i);
        v128_t b_f32x4 = wasm_v128_load(b + i);
        wasm_v128_store(result + i, wasm_f32x4_add(a_f32x4, b_f32x4));
    }
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

NK_API_COMPTIME void nk_each_scale_f32_v128relaxed(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                   nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    v128_t alpha_f32x4 = wasm_f32x4_splat(alpha_val);
    v128_t beta_f32x4 = wasm_f32x4_splat(beta_val);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        v128_t a_f32x4 = wasm_v128_load(a + i);
        wasm_v128_store(result + i, wasm_f32x4_relaxed_madd(a_f32x4, alpha_f32x4, beta_f32x4));
    }
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val;
}

NK_API_COMPTIME void nk_each_blend_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                   nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    if (alpha_val == 1 && beta_val == 1) {
        nk_each_sum_f32_v128relaxed(a, b, n, result);
        return;
    }
    else if (alpha_val == 0 || beta_val == 0) {
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_f32_v128relaxed(a, n, alpha, &zero, result); }
        else { nk_each_scale_f32_v128relaxed(b, n, beta, &zero, result); }
        return;
    }
    v128_t alpha_f32x4 = wasm_f32x4_splat(alpha_val);
    v128_t beta_f32x4 = wasm_f32x4_splat(beta_val);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        v128_t a_f32x4 = wasm_v128_load(a + i);
        v128_t b_f32x4 = wasm_v128_load(b + i);
        v128_t a_scaled_f32x4 = wasm_f32x4_mul(a_f32x4, alpha_f32x4);
        wasm_v128_store(result + i, wasm_f32x4_relaxed_madd(b_f32x4, beta_f32x4, a_scaled_f32x4));
    }
    for (; i < n; ++i) result[i] = alpha_val * a[i] + beta_val * b[i];
}

NK_API_COMPTIME void nk_each_fma_f32_v128relaxed(            //
    nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    v128_t alpha_f32x4 = wasm_f32x4_splat(alpha_val);
    v128_t beta_f32x4 = wasm_f32x4_splat(beta_val);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        v128_t a_f32x4 = wasm_v128_load(a + i);
        v128_t b_f32x4 = wasm_v128_load(b + i);
        v128_t c_f32x4 = wasm_v128_load(c + i);
        v128_t ab_f32x4 = wasm_f32x4_mul(a_f32x4, b_f32x4);
        v128_t ab_scaled_f32x4 = wasm_f32x4_mul(ab_f32x4, alpha_f32x4);
        wasm_v128_store(result + i, wasm_f32x4_relaxed_madd(c_f32x4, beta_f32x4, ab_scaled_f32x4));
    }
    for (; i < n; ++i) result[i] = alpha_val * a[i] * b[i] + beta_val * c[i];
}

#pragma endregion F32 Floats
#pragma region F16 Floats

NK_API_COMPTIME void nk_each_sum_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        nk_b64_vec_t a_f16_vec, b_f16_vec;
        nk_load_b64_serial_(a + i, &a_f16_vec);
        nk_load_b64_serial_(b + i, &b_f16_vec);
        nk_b128_vec_t a_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(a_f16_vec);
        nk_b128_vec_t b_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(b_f16_vec);
        nk_b128_vec_t result_f32_vec;
        result_f32_vec.v128 = wasm_f32x4_add(a_f32_vec.v128, b_f32_vec.v128);
        nk_b64_vec_t result_f16_vec = nk_f32x4_to_f16x4_v128relaxed_(result_f32_vec);
        nk_store_b64_serial_(&result_f16_vec, result + i);
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_f16_to_f32_serial(a + i, &ai);
        nk_f16_to_f32_serial(b + i, &bi);
        nk_f32_t sum = ai + bi;
        nk_f32_to_f16_serial(&sum, result + i);
    }
}

NK_API_COMPTIME void nk_each_scale_f16_v128relaxed(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                   nk_f32_t const *beta, nk_f16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    v128_t alpha_f32x4 = wasm_f32x4_splat(alpha_val);
    v128_t beta_f32x4 = wasm_f32x4_splat(beta_val);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        nk_b64_vec_t a_f16_vec;
        nk_load_b64_serial_(a + i, &a_f16_vec);
        nk_b128_vec_t a_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(a_f16_vec);
        nk_b128_vec_t result_f32_vec;
        result_f32_vec.v128 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, alpha_f32x4, beta_f32x4);
        nk_b64_vec_t result_f16_vec = nk_f32x4_to_f16x4_v128relaxed_(result_f32_vec);
        nk_store_b64_serial_(&result_f16_vec, result + i);
    }
    for (; i < n; ++i) {
        nk_f32_t ai;
        nk_f16_to_f32_serial(a + i, &ai);
        nk_f32_t sum = alpha_val * ai + beta_val;
        nk_f32_to_f16_serial(&sum, result + i);
    }
}

NK_API_COMPTIME void nk_each_blend_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                   nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    if (alpha_val == 1 && beta_val == 1) {
        nk_each_sum_f16_v128relaxed(a, b, n, result);
        return;
    }
    else if (alpha_val == 0 || beta_val == 0) {
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_f16_v128relaxed(a, n, alpha, &zero, result); }
        else { nk_each_scale_f16_v128relaxed(b, n, beta, &zero, result); }
        return;
    }
    v128_t alpha_f32x4 = wasm_f32x4_splat(alpha_val);
    v128_t beta_f32x4 = wasm_f32x4_splat(beta_val);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        nk_b64_vec_t a_f16_vec, b_f16_vec;
        nk_load_b64_serial_(a + i, &a_f16_vec);
        nk_load_b64_serial_(b + i, &b_f16_vec);
        nk_b128_vec_t a_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(a_f16_vec);
        nk_b128_vec_t b_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(b_f16_vec);
        v128_t a_scaled_f32x4 = wasm_f32x4_mul(a_f32_vec.v128, alpha_f32x4);
        nk_b128_vec_t result_f32_vec;
        result_f32_vec.v128 = wasm_f32x4_relaxed_madd(b_f32_vec.v128, beta_f32x4, a_scaled_f32x4);
        nk_b64_vec_t result_f16_vec = nk_f32x4_to_f16x4_v128relaxed_(result_f32_vec);
        nk_store_b64_serial_(&result_f16_vec, result + i);
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_f16_to_f32_serial(a + i, &ai);
        nk_f16_to_f32_serial(b + i, &bi);
        nk_f32_t sum = alpha_val * ai + beta_val * bi;
        nk_f32_to_f16_serial(&sum, result + i);
    }
}

NK_API_COMPTIME void nk_each_fma_f16_v128relaxed(            //
    nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    v128_t alpha_f32x4 = wasm_f32x4_splat(alpha_val);
    v128_t beta_f32x4 = wasm_f32x4_splat(beta_val);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        nk_b64_vec_t a_f16_vec, b_f16_vec, c_f16_vec;
        nk_load_b64_serial_(a + i, &a_f16_vec);
        nk_load_b64_serial_(b + i, &b_f16_vec);
        nk_load_b64_serial_(c + i, &c_f16_vec);
        nk_b128_vec_t a_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(a_f16_vec);
        nk_b128_vec_t b_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(b_f16_vec);
        nk_b128_vec_t c_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(c_f16_vec);
        v128_t ab_f32x4 = wasm_f32x4_mul(a_f32_vec.v128, b_f32_vec.v128);
        v128_t ab_scaled_f32x4 = wasm_f32x4_mul(ab_f32x4, alpha_f32x4);
        nk_b128_vec_t result_f32_vec;
        result_f32_vec.v128 = wasm_f32x4_relaxed_madd(c_f32_vec.v128, beta_f32x4, ab_scaled_f32x4);
        nk_b64_vec_t result_f16_vec = nk_f32x4_to_f16x4_v128relaxed_(result_f32_vec);
        nk_store_b64_serial_(&result_f16_vec, result + i);
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi, ci;
        nk_f16_to_f32_serial(a + i, &ai);
        nk_f16_to_f32_serial(b + i, &bi);
        nk_f16_to_f32_serial(c + i, &ci);
        nk_f32_t sum = alpha_val * ai * bi + beta_val * ci;
        nk_f32_to_f16_serial(&sum, result + i);
    }
}

#pragma endregion F16 Floats
#pragma region BF16 Floats

NK_API_COMPTIME void nk_each_sum_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                  nk_bf16_t *result) {
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        nk_b64_vec_t a_bf16_vec, b_bf16_vec;
        nk_load_b64_serial_(a + i, &a_bf16_vec);
        nk_load_b64_serial_(b + i, &b_bf16_vec);
        nk_b128_vec_t a_f32_vec = nk_bf16x4_to_f32x4_v128relaxed_(a_bf16_vec);
        nk_b128_vec_t b_f32_vec = nk_bf16x4_to_f32x4_v128relaxed_(b_bf16_vec);
        nk_b128_vec_t result_f32_vec;
        result_f32_vec.v128 = wasm_f32x4_add(a_f32_vec.v128, b_f32_vec.v128);
        nk_b64_vec_t result_bf16_vec = nk_f32x4_to_bf16x4_v128relaxed_(result_f32_vec);
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

NK_API_COMPTIME void nk_each_scale_bf16_v128relaxed(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                    nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    v128_t alpha_f32x4 = wasm_f32x4_splat(alpha_val);
    v128_t beta_f32x4 = wasm_f32x4_splat(beta_val);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        nk_b64_vec_t a_bf16_vec;
        nk_load_b64_serial_(a + i, &a_bf16_vec);
        nk_b128_vec_t a_f32_vec = nk_bf16x4_to_f32x4_v128relaxed_(a_bf16_vec);
        nk_b128_vec_t result_f32_vec;
        result_f32_vec.v128 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, alpha_f32x4, beta_f32x4);
        nk_b64_vec_t result_bf16_vec = nk_f32x4_to_bf16x4_v128relaxed_(result_f32_vec);
        nk_store_b64_serial_(&result_bf16_vec, result + i);
    }
    for (; i < n; ++i) {
        nk_f32_t ai;
        nk_bf16_to_f32_serial(a + i, &ai);
        nk_f32_t sum = alpha_val * ai + beta_val;
        nk_f32_to_bf16_serial(&sum, result + i);
    }
}

NK_API_COMPTIME void nk_each_blend_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                    nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    if (alpha_val == 1 && beta_val == 1) {
        nk_each_sum_bf16_v128relaxed(a, b, n, result);
        return;
    }
    else if (alpha_val == 0 || beta_val == 0) {
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_bf16_v128relaxed(a, n, alpha, &zero, result); }
        else { nk_each_scale_bf16_v128relaxed(b, n, beta, &zero, result); }
        return;
    }
    v128_t alpha_f32x4 = wasm_f32x4_splat(alpha_val);
    v128_t beta_f32x4 = wasm_f32x4_splat(beta_val);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        nk_b64_vec_t a_bf16_vec, b_bf16_vec;
        nk_load_b64_serial_(a + i, &a_bf16_vec);
        nk_load_b64_serial_(b + i, &b_bf16_vec);
        nk_b128_vec_t a_f32_vec = nk_bf16x4_to_f32x4_v128relaxed_(a_bf16_vec);
        nk_b128_vec_t b_f32_vec = nk_bf16x4_to_f32x4_v128relaxed_(b_bf16_vec);
        v128_t a_scaled_f32x4 = wasm_f32x4_mul(a_f32_vec.v128, alpha_f32x4);
        nk_b128_vec_t result_f32_vec;
        result_f32_vec.v128 = wasm_f32x4_relaxed_madd(b_f32_vec.v128, beta_f32x4, a_scaled_f32x4);
        nk_b64_vec_t result_bf16_vec = nk_f32x4_to_bf16x4_v128relaxed_(result_f32_vec);
        nk_store_b64_serial_(&result_bf16_vec, result + i);
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_bf16_to_f32_serial(a + i, &ai);
        nk_bf16_to_f32_serial(b + i, &bi);
        nk_f32_t sum = alpha_val * ai + beta_val * bi;
        nk_f32_to_bf16_serial(&sum, result + i);
    }
}

NK_API_COMPTIME void nk_each_fma_bf16_v128relaxed(              //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    v128_t alpha_f32x4 = wasm_f32x4_splat(alpha_val);
    v128_t beta_f32x4 = wasm_f32x4_splat(beta_val);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        nk_b64_vec_t a_bf16_vec, b_bf16_vec, c_bf16_vec;
        nk_load_b64_serial_(a + i, &a_bf16_vec);
        nk_load_b64_serial_(b + i, &b_bf16_vec);
        nk_load_b64_serial_(c + i, &c_bf16_vec);
        nk_b128_vec_t a_f32_vec = nk_bf16x4_to_f32x4_v128relaxed_(a_bf16_vec);
        nk_b128_vec_t b_f32_vec = nk_bf16x4_to_f32x4_v128relaxed_(b_bf16_vec);
        nk_b128_vec_t c_f32_vec = nk_bf16x4_to_f32x4_v128relaxed_(c_bf16_vec);
        v128_t ab_f32x4 = wasm_f32x4_mul(a_f32_vec.v128, b_f32_vec.v128);
        v128_t ab_scaled_f32x4 = wasm_f32x4_mul(ab_f32x4, alpha_f32x4);
        nk_b128_vec_t result_f32_vec;
        result_f32_vec.v128 = wasm_f32x4_relaxed_madd(c_f32_vec.v128, beta_f32x4, ab_scaled_f32x4);
        nk_b64_vec_t result_bf16_vec = nk_f32x4_to_bf16x4_v128relaxed_(result_f32_vec);
        nk_store_b64_serial_(&result_bf16_vec, result + i);
    }
    for (; i < n; ++i) {
        nk_f32_t ai, bi, ci;
        nk_bf16_to_f32_serial(a + i, &ai);
        nk_bf16_to_f32_serial(b + i, &bi);
        nk_bf16_to_f32_serial(c + i, &ci);
        nk_f32_t sum = alpha_val * ai * bi + beta_val * ci;
        nk_f32_to_bf16_serial(&sum, result + i);
    }
}

#pragma endregion BF16 Floats
#pragma region I8 Integers

NK_API_COMPTIME void nk_each_sum_i8_v128relaxed(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result) {
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

NK_API_COMPTIME void nk_each_scale_i8_v128relaxed(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                  nk_f32_t const *beta, nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    v128_t alpha_f32x4 = wasm_f32x4_splat(alpha_val);
    v128_t beta_f32x4 = wasm_f32x4_splat(beta_val);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        nk_b32_vec_t raw;
        nk_load_b32_serial_(a + i, &raw);
        nk_b128_vec_t a_f32_vec = nk_i8x4_to_f32x4_v128relaxed_(raw);
        nk_b128_vec_t result_f32_vec;
        result_f32_vec.v128 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, alpha_f32x4, beta_f32x4);
        nk_b32_vec_t result_i8_vec = nk_f32x4_to_i8x4_v128relaxed_(result_f32_vec);
        nk_store_b32_serial_(&result_i8_vec, result + i);
    }
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_val * a[i] + beta_val;
        nk_f32_to_i8_serial(&sum, result + i);
    }
}

NK_API_COMPTIME void nk_each_blend_i8_v128relaxed(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n,
                                                  nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    if (alpha_val == 1 && beta_val == 1) {
        nk_each_sum_i8_v128relaxed(a, b, n, result);
        return;
    }
    else if (alpha_val == 0 || beta_val == 0) {
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_i8_v128relaxed(a, n, alpha, &zero, result); }
        else { nk_each_scale_i8_v128relaxed(b, n, beta, &zero, result); }
        return;
    }
    v128_t alpha_f32x4 = wasm_f32x4_splat(alpha_val);
    v128_t beta_f32x4 = wasm_f32x4_splat(beta_val);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        nk_b32_vec_t a_raw, b_raw;
        nk_load_b32_serial_(a + i, &a_raw);
        nk_load_b32_serial_(b + i, &b_raw);
        nk_b128_vec_t a_f32_vec = nk_i8x4_to_f32x4_v128relaxed_(a_raw);
        nk_b128_vec_t b_f32_vec = nk_i8x4_to_f32x4_v128relaxed_(b_raw);
        v128_t a_scaled_f32x4 = wasm_f32x4_mul(a_f32_vec.v128, alpha_f32x4);
        nk_b128_vec_t result_f32_vec;
        result_f32_vec.v128 = wasm_f32x4_relaxed_madd(b_f32_vec.v128, beta_f32x4, a_scaled_f32x4);
        nk_b32_vec_t result_i8_vec = nk_f32x4_to_i8x4_v128relaxed_(result_f32_vec);
        nk_store_b32_serial_(&result_i8_vec, result + i);
    }
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_val * a[i] + beta_val * b[i];
        nk_f32_to_i8_serial(&sum, result + i);
    }
}

NK_API_COMPTIME void nk_each_fma_i8_v128relaxed(          //
    nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    v128_t alpha_f32x4 = wasm_f32x4_splat(alpha_val);
    v128_t beta_f32x4 = wasm_f32x4_splat(beta_val);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        nk_b32_vec_t a_raw, b_raw, c_raw;
        nk_load_b32_serial_(a + i, &a_raw);
        nk_load_b32_serial_(b + i, &b_raw);
        nk_load_b32_serial_(c + i, &c_raw);
        nk_b128_vec_t a_f32_vec = nk_i8x4_to_f32x4_v128relaxed_(a_raw);
        nk_b128_vec_t b_f32_vec = nk_i8x4_to_f32x4_v128relaxed_(b_raw);
        nk_b128_vec_t c_f32_vec = nk_i8x4_to_f32x4_v128relaxed_(c_raw);
        v128_t ab_f32x4 = wasm_f32x4_mul(a_f32_vec.v128, b_f32_vec.v128);
        v128_t ab_scaled_f32x4 = wasm_f32x4_mul(ab_f32x4, alpha_f32x4);
        nk_b128_vec_t result_f32_vec;
        result_f32_vec.v128 = wasm_f32x4_relaxed_madd(c_f32_vec.v128, beta_f32x4, ab_scaled_f32x4);
        nk_b32_vec_t result_i8_vec = nk_f32x4_to_i8x4_v128relaxed_(result_f32_vec);
        nk_store_b32_serial_(&result_i8_vec, result + i);
    }
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_val * a[i] * b[i] + beta_val * c[i];
        nk_f32_to_i8_serial(&sum, result + i);
    }
}

#pragma endregion I8 Integers
#pragma region U8 Integers

NK_API_COMPTIME void nk_each_sum_u8_v128relaxed(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result) {
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

NK_API_COMPTIME void nk_each_scale_u8_v128relaxed(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha,
                                                  nk_f32_t const *beta, nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    v128_t alpha_f32x4 = wasm_f32x4_splat(alpha_val);
    v128_t beta_f32x4 = wasm_f32x4_splat(beta_val);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        nk_b32_vec_t raw;
        nk_load_b32_serial_(a + i, &raw);
        nk_b128_vec_t a_f32_vec = nk_u8x4_to_f32x4_v128relaxed_(raw);
        nk_b128_vec_t result_f32_vec;
        result_f32_vec.v128 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, alpha_f32x4, beta_f32x4);
        nk_b32_vec_t result_u8_vec = nk_f32x4_to_u8x4_v128relaxed_(result_f32_vec);
        nk_store_b32_serial_(&result_u8_vec, result + i);
    }
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_val * a[i] + beta_val;
        nk_f32_to_u8_serial(&sum, result + i);
    }
}

NK_API_COMPTIME void nk_each_blend_u8_v128relaxed(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n,
                                                  nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    if (alpha_val == 1 && beta_val == 1) {
        nk_each_sum_u8_v128relaxed(a, b, n, result);
        return;
    }
    else if (alpha_val == 0 || beta_val == 0) {
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_u8_v128relaxed(a, n, alpha, &zero, result); }
        else { nk_each_scale_u8_v128relaxed(b, n, beta, &zero, result); }
        return;
    }
    v128_t alpha_f32x4 = wasm_f32x4_splat(alpha_val);
    v128_t beta_f32x4 = wasm_f32x4_splat(beta_val);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        nk_b32_vec_t a_raw, b_raw;
        nk_load_b32_serial_(a + i, &a_raw);
        nk_load_b32_serial_(b + i, &b_raw);
        nk_b128_vec_t a_f32_vec = nk_u8x4_to_f32x4_v128relaxed_(a_raw);
        nk_b128_vec_t b_f32_vec = nk_u8x4_to_f32x4_v128relaxed_(b_raw);
        v128_t a_scaled_f32x4 = wasm_f32x4_mul(a_f32_vec.v128, alpha_f32x4);
        nk_b128_vec_t result_f32_vec;
        result_f32_vec.v128 = wasm_f32x4_relaxed_madd(b_f32_vec.v128, beta_f32x4, a_scaled_f32x4);
        nk_b32_vec_t result_u8_vec = nk_f32x4_to_u8x4_v128relaxed_(result_f32_vec);
        nk_store_b32_serial_(&result_u8_vec, result + i);
    }
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_val * a[i] + beta_val * b[i];
        nk_f32_to_u8_serial(&sum, result + i);
    }
}

NK_API_COMPTIME void nk_each_fma_u8_v128relaxed(          //
    nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, //
    nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    v128_t alpha_f32x4 = wasm_f32x4_splat(alpha_val);
    v128_t beta_f32x4 = wasm_f32x4_splat(beta_val);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        nk_b32_vec_t a_raw, b_raw, c_raw;
        nk_load_b32_serial_(a + i, &a_raw);
        nk_load_b32_serial_(b + i, &b_raw);
        nk_load_b32_serial_(c + i, &c_raw);
        nk_b128_vec_t a_f32_vec = nk_u8x4_to_f32x4_v128relaxed_(a_raw);
        nk_b128_vec_t b_f32_vec = nk_u8x4_to_f32x4_v128relaxed_(b_raw);
        nk_b128_vec_t c_f32_vec = nk_u8x4_to_f32x4_v128relaxed_(c_raw);
        v128_t ab_f32x4 = wasm_f32x4_mul(a_f32_vec.v128, b_f32_vec.v128);
        v128_t ab_scaled_f32x4 = wasm_f32x4_mul(ab_f32x4, alpha_f32x4);
        nk_b128_vec_t result_f32_vec;
        result_f32_vec.v128 = wasm_f32x4_relaxed_madd(c_f32_vec.v128, beta_f32x4, ab_scaled_f32x4);
        nk_b32_vec_t result_u8_vec = nk_f32x4_to_u8x4_v128relaxed_(result_f32_vec);
        nk_store_b32_serial_(&result_u8_vec, result + i);
    }
    for (; i < n; ++i) {
        nk_f32_t sum = alpha_val * a[i] * b[i] + beta_val * c[i];
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

#endif // NK_TARGET_V128RELAXED

#endif // NK_EACH_V128RELAXED_H
