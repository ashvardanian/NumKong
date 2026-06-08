/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for WASM.
 *  @file include/numkong/spatial/v128relaxed.h
 *  @author Ash Vardanian
 *  @date February 2, 2026
 *
 *  Contains:
 *  - Euclidean (L2) distance
 *  - Squared Euclidean (L2SQ) distance
 *  - Angular distance (1 - cosine similarity)
 *
 *  For dtypes:
 *  - 64-bit IEEE floating point (f64)
 *  - 32-bit IEEE floating point (f32)
 *  - 16-bit IEEE floating point (f16)
 *  - 16-bit brain floating point (bf16)
 *
 *  Key improvements:
 *  - F32→F64 upcast for angular_f32 (matches Haswell/NEON precision strategy)
 *  - Parallel SIMD sqrt for normalization (computes both sqrts simultaneously)
 *  - Edge case handling (zero vectors, numerical stability)
 *  - Uses relaxed FMA for optimal throughput
 *
 *  @see For pattern references:
 *  - Haswell: include/numkong/spatial/haswell.h
 *  - NEON: include/numkong/spatial/neon.h
 */

#ifndef NK_SPATIAL_V128RELAXED_H
#define NK_SPATIAL_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/scalar/v128relaxed.h" // `nk_f32_sqrt_v128relaxed`
#include "numkong/reduce/v128relaxed.h"
#include "numkong/cast/serial.h"
#include "numkong/cast/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

NK_INTERNAL nk_f64_t nk_angular_normalize_f64_v128relaxed_(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {
    // Edge case: both vectors have zero magnitude
    if (a2 == 0.0 && b2 == 0.0) return 0.0;
    // Edge case: dot product is zero (perpendicular or one vector is zero)
    if (ab == 0.0) return 1.0;

    // Compute both square roots in parallel using SIMD (more efficient than 2 scalar sqrts)
    v128_t squares_f64x2 = wasm_f64x2_make(a2, b2);
    v128_t sqrts_f64x2 = wasm_f64x2_sqrt(squares_f64x2);
    nk_f64_t a_sqrt = wasm_f64x2_extract_lane(sqrts_f64x2, 0);
    nk_f64_t b_sqrt = wasm_f64x2_extract_lane(sqrts_f64x2, 1);

    // Compute angular distance: 1 - cosine_similarity
    nk_f64_t result = 1.0 - ab / (a_sqrt * b_sqrt);

    // Clamp negative results to 0 (can occur due to floating-point rounding)
    return result > 0.0 ? result : 0.0;
}

#pragma region F32 and F64 Floats

NK_PUBLIC void nk_sqeuclidean_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    v128_t sum_f64x2 = wasm_f64x2_splat(0.0);
    nk_f32_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b64_vec_t a_f32_vec, b_f32_vec;

nk_sqeuclidean_f32_v128relaxed_cycle:
    if (count_scalars < 2) {
        nk_partial_load_b32x2_serial_(a_scalars, &a_f32_vec, count_scalars);
        nk_partial_load_b32x2_serial_(b_scalars, &b_f32_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b64_serial_(a_scalars, &a_f32_vec);
        nk_load_b64_serial_(b_scalars, &b_f32_vec);
        a_scalars += 2, b_scalars += 2, count_scalars -= 2;
    }
    v128_t a_f32x2 = wasm_i64x2_splat(a_f32_vec.u64);
    v128_t b_f32x2 = wasm_i64x2_splat(b_f32_vec.u64);
    v128_t a_f64x2 = wasm_f64x2_promote_low_f32x4(a_f32x2);
    v128_t b_f64x2 = wasm_f64x2_promote_low_f32x4(b_f32x2);
    v128_t diff_f64x2 = wasm_f64x2_sub(a_f64x2, b_f64x2);
    sum_f64x2 = wasm_f64x2_relaxed_madd(diff_f64x2, diff_f64x2, sum_f64x2);
    if (count_scalars) goto nk_sqeuclidean_f32_v128relaxed_cycle;

    *result = nk_reduce_add_f64x2_v128relaxed_(sum_f64x2);
}

NK_PUBLIC void nk_sqeuclidean_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    v128_t sum_f64x2 = wasm_f64x2_splat(0.0);
    nk_f64_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b128_vec_t a_vec, b_vec;

nk_sqeuclidean_f64_v128relaxed_cycle:
    if (count_scalars < 2) {
        nk_partial_load_b64x2_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b64x2_serial_(b_scalars, &b_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b128_v128relaxed_(a_scalars, &a_vec);
        nk_load_b128_v128relaxed_(b_scalars, &b_vec);
        a_scalars += 2, b_scalars += 2, count_scalars -= 2;
    }
    v128_t diff_f64x2 = wasm_f64x2_sub(a_vec.v128, b_vec.v128);
    sum_f64x2 = wasm_f64x2_relaxed_madd(diff_f64x2, diff_f64x2, sum_f64x2);
    if (count_scalars) goto nk_sqeuclidean_f64_v128relaxed_cycle;

    *result = nk_reduce_add_f64x2_v128relaxed_(sum_f64x2);
}

NK_PUBLIC void nk_euclidean_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_f64_t l2sq;
    nk_sqeuclidean_f32_v128relaxed(a, b, n, &l2sq);
    *result = nk_f64_sqrt_v128relaxed(l2sq);
}

NK_PUBLIC void nk_euclidean_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_f64_t l2sq;
    nk_sqeuclidean_f64_v128relaxed(a, b, n, &l2sq);
    *result = nk_f64_sqrt_v128relaxed(l2sq);
}

NK_PUBLIC void nk_angular_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    // F32 → F64 upcast for numerical stability
    v128_t ab_f64x2 = wasm_f64x2_splat(0.0);
    v128_t a2_f64x2 = wasm_f64x2_splat(0.0);
    v128_t b2_f64x2 = wasm_f64x2_splat(0.0);
    nk_f32_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b64_vec_t a_f32_vec, b_f32_vec;

nk_angular_f32_v128relaxed_cycle:
    if (count_scalars < 2) {
        nk_partial_load_b32x2_serial_(a_scalars, &a_f32_vec, count_scalars);
        nk_partial_load_b32x2_serial_(b_scalars, &b_f32_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b64_serial_(a_scalars, &a_f32_vec);
        nk_load_b64_serial_(b_scalars, &b_f32_vec);
        a_scalars += 2, b_scalars += 2, count_scalars -= 2;
    }

    // Upcast F32x2 → F64x2 for high-precision accumulation
    v128_t a_f32x2 = wasm_i64x2_splat(a_f32_vec.u64);
    v128_t b_f32x2 = wasm_i64x2_splat(b_f32_vec.u64);
    v128_t a_f64x2 = wasm_f64x2_promote_low_f32x4(a_f32x2);
    v128_t b_f64x2 = wasm_f64x2_promote_low_f32x4(b_f32x2);

    // Accumulate: ab += a·b, a2 += a·a, b2 += b·b
    ab_f64x2 = wasm_f64x2_relaxed_madd(a_f64x2, b_f64x2, ab_f64x2);
    a2_f64x2 = wasm_f64x2_relaxed_madd(a_f64x2, a_f64x2, a2_f64x2);
    b2_f64x2 = wasm_f64x2_relaxed_madd(b_f64x2, b_f64x2, b2_f64x2);
    if (count_scalars) goto nk_angular_f32_v128relaxed_cycle;

    // Reduce and normalize using F64 arithmetic
    nk_f64_t ab_f64 = nk_reduce_add_f64x2_v128relaxed_(ab_f64x2);
    nk_f64_t a2_f64 = nk_reduce_add_f64x2_v128relaxed_(a2_f64x2);
    nk_f64_t b2_f64 = nk_reduce_add_f64x2_v128relaxed_(b2_f64x2);
    *result = nk_angular_normalize_f64_v128relaxed_(ab_f64, a2_f64, b2_f64);
}

NK_PUBLIC void nk_angular_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    v128_t ab_f64x2 = wasm_f64x2_splat(0.0);
    v128_t a2_f64x2 = wasm_f64x2_splat(0.0);
    v128_t b2_f64x2 = wasm_f64x2_splat(0.0);
    nk_f64_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b128_vec_t a_vec, b_vec;

nk_angular_f64_v128relaxed_cycle:
    if (count_scalars < 2) {
        nk_partial_load_b64x2_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b64x2_serial_(b_scalars, &b_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b128_v128relaxed_(a_scalars, &a_vec);
        nk_load_b128_v128relaxed_(b_scalars, &b_vec);
        a_scalars += 2, b_scalars += 2, count_scalars -= 2;
    }

    // Accumulate: ab += a·b, a2 += a·a, b2 += b·b
    ab_f64x2 = wasm_f64x2_relaxed_madd(a_vec.v128, b_vec.v128, ab_f64x2);
    a2_f64x2 = wasm_f64x2_relaxed_madd(a_vec.v128, a_vec.v128, a2_f64x2);
    b2_f64x2 = wasm_f64x2_relaxed_madd(b_vec.v128, b_vec.v128, b2_f64x2);
    if (count_scalars) goto nk_angular_f64_v128relaxed_cycle;

    // Reduce and normalize
    nk_f64_t ab = nk_reduce_add_f64x2_v128relaxed_(ab_f64x2);
    nk_f64_t a2 = nk_reduce_add_f64x2_v128relaxed_(a2_f64x2);
    nk_f64_t b2 = nk_reduce_add_f64x2_v128relaxed_(b2_f64x2);
    *result = nk_angular_normalize_f64_v128relaxed_(ab, a2, b2);
}

#pragma endregion F32 and F64 Floats
#pragma region F16 and BF16 Floats

NK_PUBLIC void nk_sqeuclidean_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);
    nk_f16_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b64_vec_t a_f16_vec, b_f16_vec;

nk_sqeuclidean_f16_v128relaxed_cycle:
    // Tail or full load
    if (count_scalars < 4) {
        nk_partial_load_b16x4_serial_(a_scalars, &a_f16_vec, count_scalars);
        nk_partial_load_b16x4_serial_(b_scalars, &b_f16_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b64_serial_(a_scalars, &a_f16_vec);
        nk_load_b64_serial_(b_scalars, &b_f16_vec);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }

    // Convert f16 → f32 (4 elements)
    nk_b128_vec_t a_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(a_f16_vec);
    nk_b128_vec_t b_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(b_f16_vec);

    // Accumulate (a - b)²
    v128_t diff_f32x4 = wasm_f32x4_sub(a_f32_vec.v128, b_f32_vec.v128);
    sum_f32x4 = wasm_f32x4_relaxed_madd(diff_f32x4, diff_f32x4, sum_f32x4);

    if (count_scalars) goto nk_sqeuclidean_f16_v128relaxed_cycle;

    *result = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_f32_t l2sq;
    nk_sqeuclidean_f16_v128relaxed(a, b, n, &l2sq);
    *result = nk_f32_sqrt_v128relaxed(l2sq);
}

NK_PUBLIC void nk_angular_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t ab_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t a2_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t b2_f32x4 = wasm_f32x4_splat(0.0f);
    nk_f16_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b64_vec_t a_f16_vec, b_f16_vec;

nk_angular_f16_v128relaxed_cycle:
    if (count_scalars < 4) {
        nk_partial_load_b16x4_serial_(a_scalars, &a_f16_vec, count_scalars);
        nk_partial_load_b16x4_serial_(b_scalars, &b_f16_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b64_serial_(a_scalars, &a_f16_vec);
        nk_load_b64_serial_(b_scalars, &b_f16_vec);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }

    // Convert f16 → f32
    nk_b128_vec_t a_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(a_f16_vec);
    nk_b128_vec_t b_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(b_f16_vec);

    // Triple accumulation: ab, a², b²
    ab_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, b_f32_vec.v128, ab_f32x4);
    a2_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, a_f32_vec.v128, a2_f32x4);
    b2_f32x4 = wasm_f32x4_relaxed_madd(b_f32_vec.v128, b_f32_vec.v128, b2_f32x4);

    if (count_scalars) goto nk_angular_f16_v128relaxed_cycle;

    // Reduce accumulators
    nk_f32_t ab = nk_reduce_add_f32x4_v128relaxed_(ab_f32x4);
    nk_f32_t a2 = nk_reduce_add_f32x4_v128relaxed_(a2_f32x4);
    nk_f32_t b2 = nk_reduce_add_f32x4_v128relaxed_(b2_f32x4);

    // Normalize using f64 helper (handles edge cases: zero vectors, perpendicular, clamping)
    *result = (nk_f32_t)nk_angular_normalize_f64_v128relaxed_((nk_f64_t)ab, (nk_f64_t)a2, (nk_f64_t)b2);
}

NK_PUBLIC void nk_sqeuclidean_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t mask_high_u32x4 = wasm_i32x4_splat((int)0xFFFF0000);
    nk_bf16_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b128_vec_t a_bf16_vec, b_bf16_vec;

nk_sqeuclidean_bf16_v128relaxed_cycle:
    if (count_scalars < 8) {
        nk_partial_load_b16x8_serial_(a_scalars, &a_bf16_vec, count_scalars);
        nk_partial_load_b16x8_serial_(b_scalars, &b_bf16_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b128_v128relaxed_(a_scalars, &a_bf16_vec);
        nk_load_b128_v128relaxed_(b_scalars, &b_bf16_vec);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    v128_t a_even_f32x4 = wasm_i32x4_shl(a_bf16_vec.v128, 16);
    v128_t b_even_f32x4 = wasm_i32x4_shl(b_bf16_vec.v128, 16);
    v128_t diff_even_f32x4 = wasm_f32x4_sub(a_even_f32x4, b_even_f32x4);
    sum_f32x4 = wasm_f32x4_relaxed_madd(diff_even_f32x4, diff_even_f32x4, sum_f32x4);
    v128_t a_odd_f32x4 = wasm_v128_and(a_bf16_vec.v128, mask_high_u32x4);
    v128_t b_odd_f32x4 = wasm_v128_and(b_bf16_vec.v128, mask_high_u32x4);
    v128_t diff_odd_f32x4 = wasm_f32x4_sub(a_odd_f32x4, b_odd_f32x4);
    sum_f32x4 = wasm_f32x4_relaxed_madd(diff_odd_f32x4, diff_odd_f32x4, sum_f32x4);
    if (count_scalars) goto nk_sqeuclidean_bf16_v128relaxed_cycle;

    *result = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_f32_t l2sq;
    nk_sqeuclidean_bf16_v128relaxed(a, b, n, &l2sq);
    *result = nk_f32_sqrt_v128relaxed(l2sq);
}

NK_PUBLIC void nk_angular_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t ab_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t a2_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t b2_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t mask_high_u32x4 = wasm_i32x4_splat((int)0xFFFF0000);
    nk_bf16_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b128_vec_t a_bf16_vec, b_bf16_vec;

nk_angular_bf16_v128relaxed_cycle:
    if (count_scalars < 8) {
        nk_partial_load_b16x8_serial_(a_scalars, &a_bf16_vec, count_scalars);
        nk_partial_load_b16x8_serial_(b_scalars, &b_bf16_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b128_v128relaxed_(a_scalars, &a_bf16_vec);
        nk_load_b128_v128relaxed_(b_scalars, &b_bf16_vec);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    v128_t a_even_f32x4 = wasm_i32x4_shl(a_bf16_vec.v128, 16);
    v128_t b_even_f32x4 = wasm_i32x4_shl(b_bf16_vec.v128, 16);
    ab_f32x4 = wasm_f32x4_relaxed_madd(a_even_f32x4, b_even_f32x4, ab_f32x4);
    a2_f32x4 = wasm_f32x4_relaxed_madd(a_even_f32x4, a_even_f32x4, a2_f32x4);
    b2_f32x4 = wasm_f32x4_relaxed_madd(b_even_f32x4, b_even_f32x4, b2_f32x4);
    v128_t a_odd_f32x4 = wasm_v128_and(a_bf16_vec.v128, mask_high_u32x4);
    v128_t b_odd_f32x4 = wasm_v128_and(b_bf16_vec.v128, mask_high_u32x4);
    ab_f32x4 = wasm_f32x4_relaxed_madd(a_odd_f32x4, b_odd_f32x4, ab_f32x4);
    a2_f32x4 = wasm_f32x4_relaxed_madd(a_odd_f32x4, a_odd_f32x4, a2_f32x4);
    b2_f32x4 = wasm_f32x4_relaxed_madd(b_odd_f32x4, b_odd_f32x4, b2_f32x4);
    if (count_scalars) goto nk_angular_bf16_v128relaxed_cycle;

    nk_f32_t ab = nk_reduce_add_f32x4_v128relaxed_(ab_f32x4);
    nk_f32_t a2 = nk_reduce_add_f32x4_v128relaxed_(a2_f32x4);
    nk_f32_t b2 = nk_reduce_add_f32x4_v128relaxed_(b2_f32x4);
    *result = (nk_f32_t)nk_angular_normalize_f64_v128relaxed_((nk_f64_t)ab, (nk_f64_t)a2, (nk_f64_t)b2);
}

#pragma endregion F16 and BF16 Floats
#pragma region FP8 Floats

NK_PUBLIC void nk_sqeuclidean_e4m3_v128relaxed(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);
    nk_e4m3_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b32_vec_t a_raw, b_raw;

nk_sqeuclidean_e4m3_v128relaxed_cycle:
    if (count_scalars < 4) {
        a_raw = nk_partial_load_b8x4_serial_(a_scalars, count_scalars);
        b_raw = nk_partial_load_b8x4_serial_(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b32_serial_(a_scalars, &a_raw);
        nk_load_b32_serial_(b_scalars, &b_raw);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    nk_b128_vec_t a_f32_vec = nk_e4m3x4_to_f32x4_v128relaxed_(a_raw);
    nk_b128_vec_t b_f32_vec = nk_e4m3x4_to_f32x4_v128relaxed_(b_raw);
    v128_t diff_f32x4 = wasm_f32x4_sub(a_f32_vec.v128, b_f32_vec.v128);
    sum_f32x4 = wasm_f32x4_relaxed_madd(diff_f32x4, diff_f32x4, sum_f32x4);
    if (count_scalars) goto nk_sqeuclidean_e4m3_v128relaxed_cycle;

    *result = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_e4m3_v128relaxed(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e4m3_v128relaxed(a, b, n, result);
    *result = nk_f32_sqrt_v128relaxed(*result);
}

NK_PUBLIC void nk_angular_e4m3_v128relaxed(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t ab_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t a2_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t b2_f32x4 = wasm_f32x4_splat(0.0f);
    nk_e4m3_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b32_vec_t a_raw, b_raw;

nk_angular_e4m3_v128relaxed_cycle:
    if (count_scalars < 4) {
        a_raw = nk_partial_load_b8x4_serial_(a_scalars, count_scalars);
        b_raw = nk_partial_load_b8x4_serial_(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b32_serial_(a_scalars, &a_raw);
        nk_load_b32_serial_(b_scalars, &b_raw);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    nk_b128_vec_t a_f32_vec = nk_e4m3x4_to_f32x4_v128relaxed_(a_raw);
    nk_b128_vec_t b_f32_vec = nk_e4m3x4_to_f32x4_v128relaxed_(b_raw);
    ab_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, b_f32_vec.v128, ab_f32x4);
    a2_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, a_f32_vec.v128, a2_f32x4);
    b2_f32x4 = wasm_f32x4_relaxed_madd(b_f32_vec.v128, b_f32_vec.v128, b2_f32x4);
    if (count_scalars) goto nk_angular_e4m3_v128relaxed_cycle;

    nk_f32_t ab = nk_reduce_add_f32x4_v128relaxed_(ab_f32x4);
    nk_f32_t a2 = nk_reduce_add_f32x4_v128relaxed_(a2_f32x4);
    nk_f32_t b2 = nk_reduce_add_f32x4_v128relaxed_(b2_f32x4);
    *result = (nk_f32_t)nk_angular_normalize_f64_v128relaxed_((nk_f64_t)ab, (nk_f64_t)a2, (nk_f64_t)b2);
}

NK_PUBLIC void nk_sqeuclidean_e5m2_v128relaxed(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);
    nk_e5m2_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b32_vec_t a_raw, b_raw;

nk_sqeuclidean_e5m2_v128relaxed_cycle:
    if (count_scalars < 4) {
        a_raw = nk_partial_load_b8x4_serial_(a_scalars, count_scalars);
        b_raw = nk_partial_load_b8x4_serial_(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b32_serial_(a_scalars, &a_raw);
        nk_load_b32_serial_(b_scalars, &b_raw);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    nk_b128_vec_t a_f32_vec = nk_e5m2x4_to_f32x4_v128relaxed_(a_raw);
    nk_b128_vec_t b_f32_vec = nk_e5m2x4_to_f32x4_v128relaxed_(b_raw);
    v128_t diff_f32x4 = wasm_f32x4_sub(a_f32_vec.v128, b_f32_vec.v128);
    sum_f32x4 = wasm_f32x4_relaxed_madd(diff_f32x4, diff_f32x4, sum_f32x4);
    if (count_scalars) goto nk_sqeuclidean_e5m2_v128relaxed_cycle;

    *result = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_e5m2_v128relaxed(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e5m2_v128relaxed(a, b, n, result);
    *result = nk_f32_sqrt_v128relaxed(*result);
}

NK_PUBLIC void nk_angular_e5m2_v128relaxed(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t ab_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t a2_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t b2_f32x4 = wasm_f32x4_splat(0.0f);
    nk_e5m2_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b32_vec_t a_raw, b_raw;

nk_angular_e5m2_v128relaxed_cycle:
    if (count_scalars < 4) {
        a_raw = nk_partial_load_b8x4_serial_(a_scalars, count_scalars);
        b_raw = nk_partial_load_b8x4_serial_(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b32_serial_(a_scalars, &a_raw);
        nk_load_b32_serial_(b_scalars, &b_raw);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    nk_b128_vec_t a_f32_vec = nk_e5m2x4_to_f32x4_v128relaxed_(a_raw);
    nk_b128_vec_t b_f32_vec = nk_e5m2x4_to_f32x4_v128relaxed_(b_raw);
    ab_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, b_f32_vec.v128, ab_f32x4);
    a2_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, a_f32_vec.v128, a2_f32x4);
    b2_f32x4 = wasm_f32x4_relaxed_madd(b_f32_vec.v128, b_f32_vec.v128, b2_f32x4);
    if (count_scalars) goto nk_angular_e5m2_v128relaxed_cycle;

    nk_f32_t ab = nk_reduce_add_f32x4_v128relaxed_(ab_f32x4);
    nk_f32_t a2 = nk_reduce_add_f32x4_v128relaxed_(a2_f32x4);
    nk_f32_t b2 = nk_reduce_add_f32x4_v128relaxed_(b2_f32x4);
    *result = (nk_f32_t)nk_angular_normalize_f64_v128relaxed_((nk_f64_t)ab, (nk_f64_t)a2, (nk_f64_t)b2);
}

NK_PUBLIC void nk_sqeuclidean_e2m3_v128relaxed(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);
    nk_e2m3_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b32_vec_t a_raw, b_raw;

nk_sqeuclidean_e2m3_v128relaxed_cycle:
    if (count_scalars < 4) {
        a_raw = nk_partial_load_b8x4_serial_(a_scalars, count_scalars);
        b_raw = nk_partial_load_b8x4_serial_(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b32_serial_(a_scalars, &a_raw);
        nk_load_b32_serial_(b_scalars, &b_raw);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    nk_b128_vec_t a_f32_vec = nk_e2m3x4_to_f32x4_v128relaxed_(a_raw);
    nk_b128_vec_t b_f32_vec = nk_e2m3x4_to_f32x4_v128relaxed_(b_raw);
    v128_t diff_f32x4 = wasm_f32x4_sub(a_f32_vec.v128, b_f32_vec.v128);
    sum_f32x4 = wasm_f32x4_relaxed_madd(diff_f32x4, diff_f32x4, sum_f32x4);
    if (count_scalars) goto nk_sqeuclidean_e2m3_v128relaxed_cycle;

    *result = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_e2m3_v128relaxed(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e2m3_v128relaxed(a, b, n, result);
    *result = nk_f32_sqrt_v128relaxed(*result);
}

NK_PUBLIC void nk_angular_e2m3_v128relaxed(nk_e2m3_t const *a, nk_e2m3_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t ab_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t a2_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t b2_f32x4 = wasm_f32x4_splat(0.0f);
    nk_e2m3_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b32_vec_t a_raw, b_raw;

nk_angular_e2m3_v128relaxed_cycle:
    if (count_scalars < 4) {
        a_raw = nk_partial_load_b8x4_serial_(a_scalars, count_scalars);
        b_raw = nk_partial_load_b8x4_serial_(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b32_serial_(a_scalars, &a_raw);
        nk_load_b32_serial_(b_scalars, &b_raw);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    nk_b128_vec_t a_f32_vec = nk_e2m3x4_to_f32x4_v128relaxed_(a_raw);
    nk_b128_vec_t b_f32_vec = nk_e2m3x4_to_f32x4_v128relaxed_(b_raw);
    ab_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, b_f32_vec.v128, ab_f32x4);
    a2_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, a_f32_vec.v128, a2_f32x4);
    b2_f32x4 = wasm_f32x4_relaxed_madd(b_f32_vec.v128, b_f32_vec.v128, b2_f32x4);
    if (count_scalars) goto nk_angular_e2m3_v128relaxed_cycle;

    nk_f32_t ab = nk_reduce_add_f32x4_v128relaxed_(ab_f32x4);
    nk_f32_t a2 = nk_reduce_add_f32x4_v128relaxed_(a2_f32x4);
    nk_f32_t b2 = nk_reduce_add_f32x4_v128relaxed_(b2_f32x4);
    *result = (nk_f32_t)nk_angular_normalize_f64_v128relaxed_((nk_f64_t)ab, (nk_f64_t)a2, (nk_f64_t)b2);
}

NK_PUBLIC void nk_sqeuclidean_e3m2_v128relaxed(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);
    nk_e3m2_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b32_vec_t a_raw, b_raw;

nk_sqeuclidean_e3m2_v128relaxed_cycle:
    if (count_scalars < 4) {
        a_raw = nk_partial_load_b8x4_serial_(a_scalars, count_scalars);
        b_raw = nk_partial_load_b8x4_serial_(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b32_serial_(a_scalars, &a_raw);
        nk_load_b32_serial_(b_scalars, &b_raw);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    nk_b128_vec_t a_f32_vec = nk_e3m2x4_to_f32x4_v128relaxed_(a_raw);
    nk_b128_vec_t b_f32_vec = nk_e3m2x4_to_f32x4_v128relaxed_(b_raw);
    v128_t diff_f32x4 = wasm_f32x4_sub(a_f32_vec.v128, b_f32_vec.v128);
    sum_f32x4 = wasm_f32x4_relaxed_madd(diff_f32x4, diff_f32x4, sum_f32x4);
    if (count_scalars) goto nk_sqeuclidean_e3m2_v128relaxed_cycle;

    *result = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_e3m2_v128relaxed(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_e3m2_v128relaxed(a, b, n, result);
    *result = nk_f32_sqrt_v128relaxed(*result);
}

NK_PUBLIC void nk_angular_e3m2_v128relaxed(nk_e3m2_t const *a, nk_e3m2_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t ab_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t a2_f32x4 = wasm_f32x4_splat(0.0f);
    v128_t b2_f32x4 = wasm_f32x4_splat(0.0f);
    nk_e3m2_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b32_vec_t a_raw, b_raw;

nk_angular_e3m2_v128relaxed_cycle:
    if (count_scalars < 4) {
        a_raw = nk_partial_load_b8x4_serial_(a_scalars, count_scalars);
        b_raw = nk_partial_load_b8x4_serial_(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b32_serial_(a_scalars, &a_raw);
        nk_load_b32_serial_(b_scalars, &b_raw);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    nk_b128_vec_t a_f32_vec = nk_e3m2x4_to_f32x4_v128relaxed_(a_raw);
    nk_b128_vec_t b_f32_vec = nk_e3m2x4_to_f32x4_v128relaxed_(b_raw);
    ab_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, b_f32_vec.v128, ab_f32x4);
    a2_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, a_f32_vec.v128, a2_f32x4);
    b2_f32x4 = wasm_f32x4_relaxed_madd(b_f32_vec.v128, b_f32_vec.v128, b2_f32x4);
    if (count_scalars) goto nk_angular_e3m2_v128relaxed_cycle;

    nk_f32_t ab = nk_reduce_add_f32x4_v128relaxed_(ab_f32x4);
    nk_f32_t a2 = nk_reduce_add_f32x4_v128relaxed_(a2_f32x4);
    nk_f32_t b2 = nk_reduce_add_f32x4_v128relaxed_(b2_f32x4);
    *result = (nk_f32_t)nk_angular_normalize_f64_v128relaxed_((nk_f64_t)ab, (nk_f64_t)a2, (nk_f64_t)b2);
}

#pragma endregion FP8 Floats
#pragma region Spatial From Dot Helpers

/** @brief Angular from_dot: computes 1 − dot / √(query_sumsq × target_sumsq) for 4 pairs in f32. */
NK_INTERNAL void nk_angular_through_f32_from_dot_v128relaxed_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
                                                              nk_b128_vec_t const *target_sumsqs_vec,
                                                              nk_b128_vec_t *result_vec) {
    v128_t dots_f32x4 = dots_vec->v128;
    v128_t query_sumsq_f32x4 = wasm_f32x4_splat(query_sumsq);
    v128_t products_f32x4 = wasm_f32x4_mul(query_sumsq_f32x4, target_sumsqs_vec->v128);
    v128_t sqrt_products_f32x4 = wasm_f32x4_sqrt(products_f32x4);
    v128_t normalized_f32x4 = wasm_f32x4_div(dots_f32x4, sqrt_products_f32x4);
    v128_t angular_f32x4 = wasm_f32x4_sub(wasm_f32x4_splat(1.0f), normalized_f32x4);
    result_vec->v128 = wasm_f32x4_max(angular_f32x4, wasm_f32x4_splat(0.0f));
}

/** @brief Euclidean from_dot: computes √(query_sumsq + target_sumsq − 2 × dot) for 4 pairs in f32. */
NK_INTERNAL void nk_euclidean_through_f32_from_dot_v128relaxed_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
                                                                nk_b128_vec_t const *target_sumsqs_vec,
                                                                nk_b128_vec_t *result_vec) {
    v128_t dots_f32x4 = dots_vec->v128;
    v128_t query_sumsq_f32x4 = wasm_f32x4_splat(query_sumsq);
    v128_t two_f32x4 = wasm_f32x4_splat(2.0f);
    v128_t sum_sq_f32x4 = wasm_f32x4_add(query_sumsq_f32x4, target_sumsqs_vec->v128);
    v128_t dist_sq_f32x4 = wasm_f32x4_relaxed_nmadd(two_f32x4, dots_f32x4, sum_sq_f32x4);
    dist_sq_f32x4 = wasm_f32x4_max(dist_sq_f32x4, wasm_f32x4_splat(0.0f));
    result_vec->v128 = wasm_f32x4_sqrt(dist_sq_f32x4);
}

/** @brief Angular from_dot for i32 accumulators: cast to f32, then angular normalization. 4 pairs. */
NK_INTERNAL void nk_angular_through_i32_from_dot_v128relaxed_(nk_b128_vec_t const *dots_vec, nk_i32_t query_sumsq,
                                                              nk_b128_vec_t const *target_sumsqs_vec,
                                                              nk_b128_vec_t *result_vec) {
    v128_t dots_f32x4 = wasm_f32x4_convert_i32x4(dots_vec->v128);
    v128_t query_sumsq_f32x4 = wasm_f32x4_splat((nk_f32_t)query_sumsq);
    v128_t products_f32x4 = wasm_f32x4_mul(query_sumsq_f32x4, wasm_f32x4_convert_i32x4(target_sumsqs_vec->v128));
    v128_t sqrt_products_f32x4 = wasm_f32x4_sqrt(products_f32x4);
    v128_t normalized_f32x4 = wasm_f32x4_div(dots_f32x4, sqrt_products_f32x4);
    v128_t angular_f32x4 = wasm_f32x4_sub(wasm_f32x4_splat(1.0f), normalized_f32x4);
    result_vec->v128 = wasm_f32x4_max(angular_f32x4, wasm_f32x4_splat(0.0f));
}

/** @brief Euclidean from_dot for i32 accumulators: cast to f32, then √(a² + b² − 2ab). 4 pairs. */
NK_INTERNAL void nk_euclidean_through_i32_from_dot_v128relaxed_(nk_b128_vec_t const *dots_vec, nk_i32_t query_sumsq,
                                                                nk_b128_vec_t const *target_sumsqs_vec,
                                                                nk_b128_vec_t *result_vec) {
    v128_t dots_f32x4 = wasm_f32x4_convert_i32x4(dots_vec->v128);
    v128_t query_sumsq_f32x4 = wasm_f32x4_splat((nk_f32_t)query_sumsq);
    v128_t two_f32x4 = wasm_f32x4_splat(2.0f);
    v128_t sum_sq_f32x4 = wasm_f32x4_add(query_sumsq_f32x4, wasm_f32x4_convert_i32x4(target_sumsqs_vec->v128));
    v128_t dist_sq_f32x4 = wasm_f32x4_relaxed_nmadd(two_f32x4, dots_f32x4, sum_sq_f32x4);
    dist_sq_f32x4 = wasm_f32x4_max(dist_sq_f32x4, wasm_f32x4_splat(0.0f));
    result_vec->v128 = wasm_f32x4_sqrt(dist_sq_f32x4);
}

/** @brief Angular from_dot for u32 accumulators: cast to f32, then angular normalization. 4 pairs. */
NK_INTERNAL void nk_angular_through_u32_from_dot_v128relaxed_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                              nk_b128_vec_t const *target_sumsqs_vec,
                                                              nk_b128_vec_t *result_vec) {
    v128_t dots_f32x4 = wasm_f32x4_convert_u32x4(dots_vec->v128);
    v128_t query_sumsq_f32x4 = wasm_f32x4_splat((nk_f32_t)query_sumsq);
    v128_t products_f32x4 = wasm_f32x4_mul(query_sumsq_f32x4, wasm_f32x4_convert_u32x4(target_sumsqs_vec->v128));
    v128_t sqrt_products_f32x4 = wasm_f32x4_sqrt(products_f32x4);
    v128_t normalized_f32x4 = wasm_f32x4_div(dots_f32x4, sqrt_products_f32x4);
    v128_t angular_f32x4 = wasm_f32x4_sub(wasm_f32x4_splat(1.0f), normalized_f32x4);
    result_vec->v128 = wasm_f32x4_max(angular_f32x4, wasm_f32x4_splat(0.0f));
}

/** @brief Euclidean from_dot for u32 accumulators: cast to f32, then √(a² + b² − 2ab). 4 pairs. */
NK_INTERNAL void nk_euclidean_through_u32_from_dot_v128relaxed_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                                nk_b128_vec_t const *target_sumsqs_vec,
                                                                nk_b128_vec_t *result_vec) {
    v128_t dots_f32x4 = wasm_f32x4_convert_u32x4(dots_vec->v128);
    v128_t query_sumsq_f32x4 = wasm_f32x4_splat((nk_f32_t)query_sumsq);
    v128_t two_f32x4 = wasm_f32x4_splat(2.0f);
    v128_t sum_sq_f32x4 = wasm_f32x4_add(query_sumsq_f32x4, wasm_f32x4_convert_u32x4(target_sumsqs_vec->v128));
    v128_t dist_sq_f32x4 = wasm_f32x4_relaxed_nmadd(two_f32x4, dots_f32x4, sum_sq_f32x4);
    dist_sq_f32x4 = wasm_f32x4_max(dist_sq_f32x4, wasm_f32x4_splat(0.0f));
    result_vec->v128 = wasm_f32x4_sqrt(dist_sq_f32x4);
}

#pragma endregion Spatial From Dot Helpers
#pragma region I8 and U8 Integers

NK_PUBLIC void nk_sqeuclidean_u8_v128relaxed(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    v128_t sum_u32x4 = wasm_u32x4_splat(0);
    nk_u8_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    v128_t a_u8x16, b_u8x16;

nk_sqeuclidean_u8_v128relaxed_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec = {0}, b_vec = {0};
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        a_u8x16 = a_vec.v128;
        b_u8x16 = b_vec.v128;
        count_scalars = 0;
    }
    else {
        a_u8x16 = wasm_v128_load(a_scalars);
        b_u8x16 = wasm_v128_load(b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }

    // |a-b| via saturating subtraction: diff = (a ⊖ b) | (b ⊖ a)
    v128_t difference_u8x16 = wasm_v128_or(wasm_u8x16_sub_sat(a_u8x16, b_u8x16), wasm_u8x16_sub_sat(b_u8x16, a_u8x16));

    // Widen to u16 and square via extmul
    v128_t difference_low_u16x8 = wasm_u16x8_extend_low_u8x16(difference_u8x16);
    v128_t difference_high_u16x8 = wasm_u16x8_extend_high_u8x16(difference_u8x16);
    sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_extmul_low_i16x8(difference_low_u16x8, difference_low_u16x8));
    sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_extmul_high_i16x8(difference_low_u16x8, difference_low_u16x8));
    sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_extmul_low_i16x8(difference_high_u16x8, difference_high_u16x8));
    sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_extmul_high_i16x8(difference_high_u16x8, difference_high_u16x8));
    if (count_scalars) goto nk_sqeuclidean_u8_v128relaxed_cycle;

    *result = nk_reduce_add_u32x4_v128relaxed_(sum_u32x4);
}

NK_PUBLIC void nk_euclidean_u8_v128relaxed(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_squared;
    nk_sqeuclidean_u8_v128relaxed(a, b, n, &distance_squared);
    *result = nk_f32_sqrt_v128relaxed((nk_f32_t)distance_squared);
}

NK_PUBLIC void nk_angular_u8_v128relaxed(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    // Bias u8 [0,255] → i8 [-128,127] via XOR 0x80, then use the i8 magnitude+sign
    // decomposition for saturation-safe relaxed_dot.
    //
    // The XOR-only approach (passing raw u8 as first operand) causes vpmaddubsw saturation:
    // u8*i8 pairwise sums can reach 64770, exceeding i16 max (32767).
    // Biasing first ensures i8*u7 products stay in [-16256, 16129], pairs in [-32512, 32258].
    //
    // Let a' = a - 128, b' = b - 128 (via XOR 0x80).
    // Compute biased dots via relaxed_dot with i7 magnitude trick:
    //   a'·b' = relaxed_dot(a', b'&0x7F) - 128·Σ(a'[i] where b'[i]<0)
    // Then recover true unsigned dots:
    //   a·b = a'·b' + 128·(Σa + Σb) - n·16384
    //   a·a = a'·a' + 256·Σa - n·16384
    //   b·b = b'·b' + 256·Σb - n·16384
    nk_i64_t biased_ab = 0, biased_aa = 0, biased_bb = 0;
    nk_i64_t sum_a_total = 0, sum_b_total = 0;
    nk_size_t i = 0;

    // Windowed accumulation loop
    while (i + 16 <= n) {
        v128_t dot_ab_i32x4 = wasm_i32x4_splat(0);
        v128_t dot_aa_i32x4 = wasm_i32x4_splat(0);
        v128_t dot_bb_i32x4 = wasm_i32x4_splat(0);
        v128_t corr_ab_i16x8 = wasm_i16x8_splat(0);
        v128_t corr_aa_i16x8 = wasm_i16x8_splat(0);
        v128_t corr_bb_i16x8 = wasm_i16x8_splat(0);
        v128_t sum_a_u16x8 = wasm_u16x8_splat(0);
        v128_t sum_b_u16x8 = wasm_u16x8_splat(0);

        // Inner loop: accumulate 127 iterations before widening corrections
        // Overflow safety: max i16 lane = 127 × 254 = 32258 < 32767
        nk_size_t cycle = 0;
        for (; cycle < 127 && i + 16 <= n; ++cycle, i += 16) {
            v128_t a_u8x16 = wasm_v128_load(a + i);
            v128_t b_u8x16 = wasm_v128_load(b + i);

            // Bias to signed: a' = a ^ 0x80, b' = b ^ 0x80
            v128_t a_i8x16 = wasm_v128_xor(a_u8x16, wasm_i8x16_splat((char)0x80));
            v128_t b_i8x16 = wasm_v128_xor(b_u8x16, wasm_i8x16_splat((char)0x80));

            // Clear sign bit to get 7-bit unsigned magnitudes
            v128_t a_7bit_u8x16 = wasm_v128_and(a_i8x16, wasm_i8x16_splat(0x7F));
            v128_t b_7bit_u8x16 = wasm_v128_and(b_i8x16, wasm_i8x16_splat(0x7F));

            // Negative masks for correction
            v128_t a_neg_mask_i8x16 = wasm_i8x16_lt(a_i8x16, wasm_i8x16_splat(0));
            v128_t b_neg_mask_i8x16 = wasm_i8x16_lt(b_i8x16, wasm_i8x16_splat(0));

            // Three relaxed_dot calls on biased values
            dot_ab_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_i8x16, b_7bit_u8x16, dot_ab_i32x4);
            dot_aa_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_i8x16, a_7bit_u8x16, dot_aa_i32x4);
            dot_bb_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(b_i8x16, b_7bit_u8x16, dot_bb_i32x4);

            // Accumulate corrections in i16 (1 widening/iter instead of 2)
            v128_t a_where_b_neg = wasm_v128_and(a_i8x16, b_neg_mask_i8x16);
            v128_t a_where_a_neg = wasm_v128_and(a_i8x16, a_neg_mask_i8x16);
            v128_t b_where_b_neg = wasm_v128_and(b_i8x16, b_neg_mask_i8x16);
            corr_ab_i16x8 = wasm_i16x8_add(corr_ab_i16x8, wasm_i16x8_extadd_pairwise_i8x16(a_where_b_neg));
            corr_aa_i16x8 = wasm_i16x8_add(corr_aa_i16x8, wasm_i16x8_extadd_pairwise_i8x16(a_where_a_neg));
            corr_bb_i16x8 = wasm_i16x8_add(corr_bb_i16x8, wasm_i16x8_extadd_pairwise_i8x16(b_where_b_neg));

            // Unsigned sums for final unbias correction
            sum_a_u16x8 = wasm_i16x8_add(sum_a_u16x8, wasm_u16x8_extadd_pairwise_u8x16(a_u8x16));
            sum_b_u16x8 = wasm_i16x8_add(sum_b_u16x8, wasm_u16x8_extadd_pairwise_u8x16(b_u8x16));
        }

        // Deferred widening: i16/u16 → i32/u32 once per window
        v128_t corr_ab_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(corr_ab_i16x8);
        v128_t corr_aa_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(corr_aa_i16x8);
        v128_t corr_bb_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(corr_bb_i16x8);
        v128_t sum_a_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(sum_a_u16x8);
        v128_t sum_b_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(sum_b_u16x8);
        biased_ab += nk_reduce_add_i32x4_v128relaxed_(dot_ab_i32x4) -
                     128LL * nk_reduce_add_i32x4_v128relaxed_(corr_ab_i32x4);
        biased_aa += nk_reduce_add_i32x4_v128relaxed_(dot_aa_i32x4) -
                     128LL * nk_reduce_add_i32x4_v128relaxed_(corr_aa_i32x4);
        biased_bb += nk_reduce_add_i32x4_v128relaxed_(dot_bb_i32x4) -
                     128LL * nk_reduce_add_i32x4_v128relaxed_(corr_bb_i32x4);
        sum_a_total += nk_reduce_add_u32x4_v128relaxed_(sum_a_u32x4);
        sum_b_total += nk_reduce_add_u32x4_v128relaxed_(sum_b_u32x4);
    }

    // Scalar tail: compute biased products directly
    for (; i < n; i++) {
        nk_i32_t a_biased = (nk_i32_t)a[i] - 128;
        nk_i32_t b_biased = (nk_i32_t)b[i] - 128;
        biased_ab += (nk_i64_t)a_biased * b_biased;
        biased_aa += (nk_i64_t)a_biased * a_biased;
        biased_bb += (nk_i64_t)b_biased * b_biased;
        sum_a_total += a[i];
        sum_b_total += b[i];
    }

    // Recover true unsigned dots from biased:
    //   a·b = (a-128)·(b-128) + 128·Σa + 128·Σb - n·16384
    //   a·a = (a-128)·(a-128) + 256·Σa - n·16384
    //   b·b = (b-128)·(b-128) + 256·Σb - n·16384
    nk_i64_t n_correction = (nk_i64_t)n * 16384LL;
    nk_f64_t dot_ab = (nk_f64_t)(biased_ab + 128LL * (sum_a_total + sum_b_total) - n_correction);
    nk_f64_t norm_aa = (nk_f64_t)(biased_aa + 256LL * sum_a_total - n_correction);
    nk_f64_t norm_bb = (nk_f64_t)(biased_bb + 256LL * sum_b_total - n_correction);
    *result = (nk_f32_t)nk_angular_normalize_f64_v128relaxed_(dot_ab, norm_aa, norm_bb);
}

NK_PUBLIC void nk_sqeuclidean_i8_v128relaxed(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result) {
    // XOR with 0x80 to reinterpret signed as unsigned, then use unsigned algorithm.
    // |a-b|² is invariant under this uniform offset.
    v128_t sum_u32x4 = wasm_u32x4_splat(0);
    v128_t bias_u8x16 = wasm_u8x16_splat(0x80);
    nk_i8_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    v128_t a_u8x16, b_u8x16;

nk_sqeuclidean_i8_v128relaxed_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec = {0}, b_vec = {0};
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        a_u8x16 = wasm_v128_xor(a_vec.v128, bias_u8x16);
        b_u8x16 = wasm_v128_xor(b_vec.v128, bias_u8x16);
        count_scalars = 0;
    }
    else {
        a_u8x16 = wasm_v128_xor(wasm_v128_load(a_scalars), bias_u8x16);
        b_u8x16 = wasm_v128_xor(wasm_v128_load(b_scalars), bias_u8x16);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }

    v128_t difference_u8x16 = wasm_v128_or(wasm_u8x16_sub_sat(a_u8x16, b_u8x16), wasm_u8x16_sub_sat(b_u8x16, a_u8x16));
    v128_t difference_low_u16x8 = wasm_u16x8_extend_low_u8x16(difference_u8x16);
    v128_t difference_high_u16x8 = wasm_u16x8_extend_high_u8x16(difference_u8x16);
    sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_extmul_low_i16x8(difference_low_u16x8, difference_low_u16x8));
    sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_extmul_high_i16x8(difference_low_u16x8, difference_low_u16x8));
    sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_extmul_low_i16x8(difference_high_u16x8, difference_high_u16x8));
    sum_u32x4 = wasm_i32x4_add(sum_u32x4, wasm_i32x4_extmul_high_i16x8(difference_high_u16x8, difference_high_u16x8));
    if (count_scalars) goto nk_sqeuclidean_i8_v128relaxed_cycle;

    *result = nk_reduce_add_u32x4_v128relaxed_(sum_u32x4);
}

NK_PUBLIC void nk_euclidean_i8_v128relaxed(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_squared;
    nk_sqeuclidean_i8_v128relaxed(a, b, n, &distance_squared);
    *result = nk_f32_sqrt_v128relaxed((nk_f32_t)distance_squared);
}

NK_PUBLIC void nk_angular_i8_v128relaxed(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
    // Uses the same relaxed_dot decomposition as nk_dot_i8_v128relaxed:
    //   a·b = relaxed_dot(a, b&0x7F) - 128·Σ(a[i] where b[i]<0)
    //   a·a = relaxed_dot(a, a&0x7F) - 128·Σ(a[i] where a[i]<0)
    //   b·b = relaxed_dot(b, b&0x7F) - 128·Σ(b[i] where b[i]<0)
    nk_i64_t dot_ab_total = 0, dot_aa_total = 0, dot_bb_total = 0;
    nk_i64_t corr_ab_total = 0, corr_aa_total = 0, corr_bb_total = 0;
    nk_size_t i = 0;

    // Windowed accumulation loop
    while (i + 16 <= n) {
        v128_t dot_ab_i32x4 = wasm_i32x4_splat(0);
        v128_t dot_aa_i32x4 = wasm_i32x4_splat(0);
        v128_t dot_bb_i32x4 = wasm_i32x4_splat(0);
        v128_t corr_ab_i16x8 = wasm_i16x8_splat(0); // accumulate corrections in i16
        v128_t corr_aa_i16x8 = wasm_i16x8_splat(0);
        v128_t corr_bb_i16x8 = wasm_i16x8_splat(0);

        // Inner loop: accumulate 127 iterations before widening corrections
        // Overflow safety: max i16 lane magnitude = 127 × 254 = 32258 < 32767
        nk_size_t cycle = 0;
        for (; cycle < 127 && i + 16 <= n; ++cycle, i += 16) {
            v128_t a_i8x16 = wasm_v128_load(a + i);
            v128_t b_i8x16 = wasm_v128_load(b + i);

            // Clear sign bit to get 7-bit unsigned magnitudes
            v128_t a_7bit_u8x16 = wasm_v128_and(a_i8x16, wasm_i8x16_splat(0x7F));
            v128_t b_7bit_u8x16 = wasm_v128_and(b_i8x16, wasm_i8x16_splat(0x7F));

            // Negative masks for correction
            v128_t a_neg_mask_i8x16 = wasm_i8x16_lt(a_i8x16, wasm_i8x16_splat(0));
            v128_t b_neg_mask_i8x16 = wasm_i8x16_lt(b_i8x16, wasm_i8x16_splat(0));

            // Three relaxed_dot calls
            dot_ab_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_i8x16, b_7bit_u8x16, dot_ab_i32x4);
            dot_aa_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_i8x16, a_7bit_u8x16, dot_aa_i32x4);
            dot_bb_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(b_i8x16, b_7bit_u8x16, dot_bb_i32x4);

            // Accumulate corrections in i16 (1 widening/iter instead of 2)
            v128_t a_where_b_neg = wasm_v128_and(a_i8x16, b_neg_mask_i8x16);
            v128_t a_where_a_neg = wasm_v128_and(a_i8x16, a_neg_mask_i8x16);
            v128_t b_where_b_neg = wasm_v128_and(b_i8x16, b_neg_mask_i8x16);
            corr_ab_i16x8 = wasm_i16x8_add(corr_ab_i16x8, wasm_i16x8_extadd_pairwise_i8x16(a_where_b_neg));
            corr_aa_i16x8 = wasm_i16x8_add(corr_aa_i16x8, wasm_i16x8_extadd_pairwise_i8x16(a_where_a_neg));
            corr_bb_i16x8 = wasm_i16x8_add(corr_bb_i16x8, wasm_i16x8_extadd_pairwise_i8x16(b_where_b_neg));
        }

        // Deferred widening: i16 → i32 once per window
        v128_t corr_ab_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(corr_ab_i16x8);
        v128_t corr_aa_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(corr_aa_i16x8);
        v128_t corr_bb_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(corr_bb_i16x8);
        dot_ab_total += nk_reduce_add_i32x4_v128relaxed_(dot_ab_i32x4);
        dot_aa_total += nk_reduce_add_i32x4_v128relaxed_(dot_aa_i32x4);
        dot_bb_total += nk_reduce_add_i32x4_v128relaxed_(dot_bb_i32x4);
        corr_ab_total += nk_reduce_add_i32x4_v128relaxed_(corr_ab_i32x4);
        corr_aa_total += nk_reduce_add_i32x4_v128relaxed_(corr_aa_i32x4);
        corr_bb_total += nk_reduce_add_i32x4_v128relaxed_(corr_bb_i32x4);
    }

    // Scalar tail
    for (; i < n; i++) {
        dot_ab_total += (nk_i32_t)a[i] * (nk_i32_t)b[i];
        dot_aa_total += (nk_i32_t)a[i] * (nk_i32_t)a[i];
        dot_bb_total += (nk_i32_t)b[i] * (nk_i32_t)b[i];
    }

    // Apply correction: true_dot = relaxed_dot - 128 × correction
    // Scalar tail computes true products directly, so correction only applies to SIMD portion.
    nk_f64_t dot_ab = (nk_f64_t)(dot_ab_total - 128LL * corr_ab_total);
    nk_f64_t norm_aa = (nk_f64_t)(dot_aa_total - 128LL * corr_aa_total);
    nk_f64_t norm_bb = (nk_f64_t)(dot_bb_total - 128LL * corr_bb_total);
    *result = (nk_f32_t)nk_angular_normalize_f64_v128relaxed_(dot_ab, norm_aa, norm_bb);
}

#pragma endregion I8 and U8 Integers

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_SPATIAL_V128RELAXED_H
