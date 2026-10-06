/**
 *  @brief SIMD-accelerated Dot Products for WASM.
 *  @file include/numkong/dot/v128relaxed.h
 *  @author Ash Vardanian
 *  @date January 31, 2026
 *
 *  Requires Emscripten 3.1.27+ with `-msimd128 -mrelaxed-simd` flags.
 *
 *  Key optimizations:
 *  - Uses relaxed FMA (f32x4.relaxed_madd, f64x2.relaxed_madd) for 2x throughput
 *  - Smart i8/u8 dot products using algebraic decomposition + correction terms
 *  - F64 upcasting variant for improved numerical precision (NEON-style)
 *
 *  Smart i8 optimization:
 *    Decompose: b = b_7bit - 128 × signbit
 *    Therefore: a·b = a·b_7bit - 128 × sum(a[i] where b[i] < 0)
 *    Uses fast relaxed_dot_i8x16_i7x16 + SAD-like correction
 *
 *  Smart u8 optimization:
 *    Decompose: b = b_7bit + 128 × highbit
 *    Therefore: a·b = a·b_7bit + 128 × sum(a[i] where b[i] >= 128)
 *    Simpler than i8 (positive correction, can use shift instead of mul)
 */

#ifndef NK_DOT_V128RELAXED_H
#define NK_DOT_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/reduce/v128relaxed.h"
#include "numkong/cast/serial.h"
#include "numkong/cast/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

NK_INTERNAL nk_f64_t nk_dot_stable_sum_f64x2_v128relaxed_(v128_t sum_f64x2, v128_t compensation_f64x2) {
    v128_t tentative_sum_f64x2 = wasm_f64x2_add(sum_f64x2, compensation_f64x2);
    v128_t virtual_addend_f64x2 = wasm_f64x2_sub(tentative_sum_f64x2, sum_f64x2);
    v128_t rounding_error_f64x2 = wasm_f64x2_add(
        wasm_f64x2_sub(sum_f64x2, wasm_f64x2_sub(tentative_sum_f64x2, virtual_addend_f64x2)),
        wasm_f64x2_sub(compensation_f64x2, virtual_addend_f64x2));
    nk_f64_t lower_sum = wasm_f64x2_extract_lane(tentative_sum_f64x2, 0);
    nk_f64_t upper_sum = wasm_f64x2_extract_lane(tentative_sum_f64x2, 1);
    nk_f64_t lower_error = wasm_f64x2_extract_lane(rounding_error_f64x2, 0);
    nk_f64_t upper_error = wasm_f64x2_extract_lane(rounding_error_f64x2, 1);
    nk_f64_t tentative_sum = lower_sum + upper_sum;
    nk_f64_t virtual_addend = tentative_sum - lower_sum;
    nk_f64_t rounding_error = (lower_sum - (tentative_sum - virtual_addend)) + (upper_sum - virtual_addend);
    return tentative_sum + (lower_error + upper_error + rounding_error);
}

NK_PUBLIC void nk_dot_f32_v128relaxed(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    v128_t sum_f64x2 = wasm_f64x2_splat(0.0);
    nk_f32_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b64_vec_t a_f32_vec, b_f32_vec;

nk_dot_f32_v128relaxed_cycle:
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
    v128_t a_f32x2 = wasm_v128_load64_zero(&a_f32_vec.u64);
    v128_t b_f32x2 = wasm_v128_load64_zero(&b_f32_vec.u64);
    v128_t a_f64x2 = wasm_f64x2_promote_low_f32x4(a_f32x2);
    v128_t b_f64x2 = wasm_f64x2_promote_low_f32x4(b_f32x2);
    sum_f64x2 = wasm_f64x2_relaxed_madd(a_f64x2, b_f64x2, sum_f64x2);
    if (count_scalars) goto nk_dot_f32_v128relaxed_cycle;

    *result = nk_reduce_add_f64x2_v128relaxed_(sum_f64x2);
}

NK_PUBLIC void nk_dot_f16_v128relaxed(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);
    nk_f16_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b64_vec_t a_f16_vec, b_f16_vec;

nk_dot_f16_v128relaxed_cycle:
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
    nk_b128_vec_t a_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(a_f16_vec);
    nk_b128_vec_t b_f32_vec = nk_f16x4_to_f32x4_v128relaxed_(b_f16_vec);
    sum_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, b_f32_vec.v128, sum_f32x4);
    if (count_scalars) goto nk_dot_f16_v128relaxed_cycle;

    *result = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
}

NK_PUBLIC void nk_dot_bf16_v128relaxed(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);
    nk_bf16_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b64_vec_t a_bf16_vec, b_bf16_vec;

nk_dot_bf16_v128relaxed_cycle:
    if (count_scalars < 4) {
        nk_partial_load_b16x4_serial_(a_scalars, &a_bf16_vec, count_scalars);
        nk_partial_load_b16x4_serial_(b_scalars, &b_bf16_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b64_serial_(a_scalars, &a_bf16_vec);
        nk_load_b64_serial_(b_scalars, &b_bf16_vec);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    nk_b128_vec_t a_f32_vec = nk_bf16x4_to_f32x4_v128relaxed_(a_bf16_vec);
    nk_b128_vec_t b_f32_vec = nk_bf16x4_to_f32x4_v128relaxed_(b_bf16_vec);
    sum_f32x4 = wasm_f32x4_relaxed_madd(a_f32_vec.v128, b_f32_vec.v128, sum_f32x4);
    if (count_scalars) goto nk_dot_bf16_v128relaxed_cycle;

    *result = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
}

NK_PUBLIC void nk_dot_f64_v128relaxed(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    v128_t sum_f64x2 = wasm_f64x2_splat(0.0);
    v128_t compensation_f64x2 = wasm_f64x2_splat(0.0);
    nk_f64_t const *a_scalars = a, *b_scalars = b;
    nk_size_t count_scalars = n;
    nk_b128_vec_t a_vec, b_vec;

nk_dot_f64_v128relaxed_cycle:
    if (count_scalars < 2) {
        nk_partial_load_b64x2_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b64x2_serial_(b_scalars, &b_vec, count_scalars);
        count_scalars = 0;
    }
    else {
        nk_load_b128_serial_(a_scalars, &a_vec);
        nk_load_b128_serial_(b_scalars, &b_vec);
        a_scalars += 2, b_scalars += 2, count_scalars -= 2;
    }
    v128_t product_f64x2 = wasm_f64x2_mul(a_vec.v128, b_vec.v128);
    v128_t product_error_f64x2 = wasm_f64x2_sub(wasm_f64x2_relaxed_madd(a_vec.v128, b_vec.v128, wasm_f64x2_splat(0.0)),
                                                product_f64x2);
    v128_t tentative_sum_f64x2 = wasm_f64x2_add(sum_f64x2, product_f64x2);
    v128_t virtual_addend_f64x2 = wasm_f64x2_sub(tentative_sum_f64x2, sum_f64x2);
    v128_t sum_error_f64x2 = wasm_f64x2_add(
        wasm_f64x2_sub(sum_f64x2, wasm_f64x2_sub(tentative_sum_f64x2, virtual_addend_f64x2)),
        wasm_f64x2_sub(product_f64x2, virtual_addend_f64x2));
    sum_f64x2 = tentative_sum_f64x2;
    compensation_f64x2 = wasm_f64x2_add(compensation_f64x2, wasm_f64x2_add(sum_error_f64x2, product_error_f64x2));
    if (count_scalars) goto nk_dot_f64_v128relaxed_cycle;

    *result = nk_dot_stable_sum_f64x2_v128relaxed_(sum_f64x2, compensation_f64x2);
}

NK_PUBLIC void nk_dot_i8_v128relaxed(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i32_t *result) {
    nk_i64_t sum_total = 0;
    nk_size_t i = 0;

    // Windowed accumulation loop
    while (i + 16 <= n) {
        v128_t sum_i32x4 = wasm_i32x4_splat(0);
        v128_t correction_i16x8 = wasm_i16x8_splat(0);

        // Inner loop: accumulate 127 iterations before widening correction
        nk_size_t cycle = 0;
        for (; cycle < 127 && i + 16 <= n; ++cycle, i += 16) {
            v128_t a_i8x16 = wasm_v128_load(a + i);
            v128_t b_i8x16 = wasm_v128_load(b + i);

            // Extract sign bit: b_neg_mask = (b < 0) ? 0xFF : 0x00
            v128_t b_neg_mask_i8x16 = wasm_i8x16_lt(b_i8x16, wasm_i8x16_splat(0));

            // b_7bit = b & 0x7F (clears sign bit)
            v128_t b_7bit_i8x16 = wasm_v128_and(b_i8x16, wasm_i8x16_splat(0x7F));

            // Fast path: a · b_7bit
            sum_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_i8x16, b_7bit_i8x16, sum_i32x4);

            // Accumulate correction in i16 (only ONE extadd per iteration!)
            v128_t a_where_b_neg_i8x16 = wasm_v128_and(a_i8x16, b_neg_mask_i8x16);
            v128_t a_neg_i16x8 = wasm_i16x8_extadd_pairwise_i8x16(a_where_b_neg_i8x16);
            correction_i16x8 = wasm_i16x8_add(correction_i16x8, a_neg_i16x8);
        }

        // Widen correction once per window: i16 → i32
        v128_t corr_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(correction_i16x8);

        // Apply correction: sum -= 128 × correction
        v128_t corr_scaled_i32x4 = wasm_i32x4_mul(corr_i32x4, wasm_i32x4_splat(-128));
        sum_i32x4 = wasm_i32x4_add(sum_i32x4, corr_scaled_i32x4);

        // Reduce window to scalar
        sum_total += (nk_i32_t)nk_reduce_add_i32x4_v128relaxed_(sum_i32x4);
    }

    // Handle tail elements
    for (; i < n; i++) { sum_total += (nk_i32_t)a[i] * (nk_i32_t)b[i]; }

    *result = (nk_i32_t)sum_total;
}

NK_PUBLIC void nk_dot_u8_v128relaxed(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_i64_t biased_sum_total = 0;
    nk_i64_t sum_a_total = 0;
    nk_i64_t sum_b_total = 0;
    nk_size_t i = 0;

    // Bias u8 [0,255] → i8 [-128,127] before relaxed_dot to avoid the internal i16 saturation.
    while (i + 16 <= n) {
        v128_t biased_dot_i32x4 = wasm_i32x4_splat(0);
        v128_t correction_i16x8 = wasm_i16x8_splat(0);
        v128_t sum_a_u16x8 = wasm_u16x8_splat(0);
        v128_t sum_b_u16x8 = wasm_u16x8_splat(0);

        // Overflow safety:
        // - correction_i16x8 max lane magnitude is 127 * 128 = 16256 < 32767
        // - sum_a/sum_b max lane is 127 * 510 = 64770 < 65535
        nk_size_t cycle = 0;
        for (; cycle < 127 && i + 16 <= n; ++cycle, i += 16) {
            v128_t a_u8x16 = wasm_v128_load(a + i);
            v128_t b_u8x16 = wasm_v128_load(b + i);
            v128_t a_i8x16 = wasm_v128_xor(a_u8x16, wasm_i8x16_splat((char)0x80));
            v128_t b_i8x16 = wasm_v128_xor(b_u8x16, wasm_i8x16_splat((char)0x80));
            v128_t b_7bit_u8x16 = wasm_v128_and(b_i8x16, wasm_i8x16_splat(0x7F));
            v128_t b_neg_mask_i8x16 = wasm_i8x16_lt(b_i8x16, wasm_i8x16_splat(0));

            biased_dot_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_i8x16, b_7bit_u8x16, biased_dot_i32x4);
            correction_i16x8 = wasm_i16x8_add(
                correction_i16x8, wasm_i16x8_extadd_pairwise_i8x16(wasm_v128_and(a_i8x16, b_neg_mask_i8x16)));
            sum_a_u16x8 = wasm_i16x8_add(sum_a_u16x8, wasm_u16x8_extadd_pairwise_u8x16(a_u8x16));
            sum_b_u16x8 = wasm_i16x8_add(sum_b_u16x8, wasm_u16x8_extadd_pairwise_u8x16(b_u8x16));
        }

        v128_t correction_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(correction_i16x8);
        v128_t sum_a_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(sum_a_u16x8);
        v128_t sum_b_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(sum_b_u16x8);
        biased_sum_total += nk_reduce_add_i32x4_v128relaxed_(biased_dot_i32x4) -
                            128LL * nk_reduce_add_i32x4_v128relaxed_(correction_i32x4);
        sum_a_total += nk_reduce_add_u32x4_v128relaxed_(sum_a_u32x4);
        sum_b_total += nk_reduce_add_u32x4_v128relaxed_(sum_b_u32x4);
    }

    for (; i < n; i++) {
        nk_i32_t a_biased = (nk_i32_t)a[i] - 128;
        nk_i32_t b_biased = (nk_i32_t)b[i] - 128;
        biased_sum_total += (nk_i64_t)a_biased * b_biased;
        sum_a_total += a[i];
        sum_b_total += b[i];
    }

    biased_sum_total += 128LL * (sum_a_total + sum_b_total) - (nk_i64_t)n * 16384LL;
    *result = (nk_u32_t)biased_sum_total;
}

NK_PUBLIC void nk_dot_e2m3_v128relaxed(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f32_t *result) {
    // Integer dot product for e2m3 using relaxed SIMD: wasm_i32x4_relaxed_dot_i8x16_i7x16_add.
    // Every e2m3 value × 16 is an exact integer in [-120, +120].
    // The relaxed dot takes i8 × u7 (first signed, second unsigned [0,127]). Our magnitudes [0,120] fit.
    // Result = i32_dot / 256.0f (exact, no rounding error).
    //
    // 32-entry LUT split into two 16-entry halves for wasm_i8x16_relaxed_swizzle (indexes 0-15).
    v128_t lut_lower_u8x16 = wasm_i8x16_const(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    v128_t lut_upper_u8x16 = wasm_i8x16_const(32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120);
    v128_t magnitude_mask_u8x16 = wasm_u8x16_splat(0x1F);
    v128_t nibble_mask_u8x16 = wasm_u8x16_splat(0x0F);
    v128_t half_select_u8x16 = wasm_u8x16_splat(0x10);
    v128_t sign_mask_u8x16 = wasm_u8x16_splat(0x20);
    v128_t sum_i32x4 = wasm_i32x4_splat(0);
    v128_t a_e2m3_u8x16, b_e2m3_u8x16;

nk_dot_e2m3_v128relaxed_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        a_e2m3_u8x16 = a_vec.v128;
        b_e2m3_u8x16 = b_vec.v128;
        count_scalars = 0;
    }
    else {
        a_e2m3_u8x16 = wasm_v128_load(a_scalars);
        b_e2m3_u8x16 = wasm_v128_load(b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }

    // Extract 5-bit magnitude indices
    v128_t a_magnitude_u8x16 = wasm_v128_and(a_e2m3_u8x16, magnitude_mask_u8x16);
    v128_t b_magnitude_u8x16 = wasm_v128_and(b_e2m3_u8x16, magnitude_mask_u8x16);

    // Dual swizzle + bitselect for 32-entry LUT (a)
    v128_t a_shuffle_index_u8x16 = wasm_v128_and(a_magnitude_u8x16, nibble_mask_u8x16);
    v128_t a_lower_u8x16 = wasm_i8x16_relaxed_swizzle(lut_lower_u8x16, a_shuffle_index_u8x16);
    v128_t a_upper_u8x16 = wasm_i8x16_relaxed_swizzle(lut_upper_u8x16, a_shuffle_index_u8x16);
    v128_t a_upper_select_u8x16 = wasm_i8x16_eq(wasm_v128_and(a_magnitude_u8x16, half_select_u8x16), half_select_u8x16);
    v128_t a_unsigned_u8x16 = wasm_i8x16_relaxed_laneselect(a_upper_u8x16, a_lower_u8x16, a_upper_select_u8x16);

    // Dual swizzle + bitselect for 32-entry LUT (b)
    v128_t b_shuffle_index_u8x16 = wasm_v128_and(b_magnitude_u8x16, nibble_mask_u8x16);
    v128_t b_lower_u8x16 = wasm_i8x16_relaxed_swizzle(lut_lower_u8x16, b_shuffle_index_u8x16);
    v128_t b_upper_u8x16 = wasm_i8x16_relaxed_swizzle(lut_upper_u8x16, b_shuffle_index_u8x16);
    v128_t b_upper_select_u8x16 = wasm_i8x16_eq(wasm_v128_and(b_magnitude_u8x16, half_select_u8x16), half_select_u8x16);
    v128_t b_unsigned_u8x16 = wasm_i8x16_relaxed_laneselect(b_upper_u8x16, b_lower_u8x16, b_upper_select_u8x16);

    // Combined sign: (a ^ b) & 0x20 — nonzero means negative product
    // Apply sign to a (relaxed_dot wants i8 × u7: a_signed, b_unsigned)
    v128_t sign_combined_u8x16 = wasm_v128_and(wasm_v128_xor(a_e2m3_u8x16, b_e2m3_u8x16), sign_mask_u8x16);
    v128_t negate_mask_u8x16 = wasm_i8x16_eq(sign_combined_u8x16, sign_mask_u8x16);
    v128_t a_negated_u8x16 = wasm_i8x16_neg(a_unsigned_u8x16);
    v128_t a_signed_i8x16 = wasm_i8x16_relaxed_laneselect(a_negated_u8x16, a_unsigned_u8x16, negate_mask_u8x16);

    // relaxed_dot: a_signed[i8] × b_unsigned[u7] → i32 accumulate
    sum_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_signed_i8x16, b_unsigned_u8x16, sum_i32x4);

    if (count_scalars) goto nk_dot_e2m3_v128relaxed_cycle;
    *result = (nk_f32_t)nk_reduce_add_i32x4_v128relaxed_(sum_i32x4) / 256.0f;
}

NK_PUBLIC void nk_dot_e3m2_v128relaxed(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f32_t *result) {
    // Integer dot product for e3m2 using i16 arithmetic with widening multiply.
    // Every e3m2 value × 16 is an exact integer, but magnitudes reach 448, requiring i16.
    // Result = i32_dot / 256.0f (exact, no rounding error).
    //
    // 32-entry magnitude LUT split into low-byte halves for dual swizzle lookup.
    // High byte is 0 for indices 0-27 and 1 for indices 28-31, so a simple comparison
    // replaces the high-byte LUT entirely.
    //
    // Low-byte LUT entries (magnitude[i] & 0xFF):
    //   [0,1,2,3,4,5,6,7,8,10,12,14,16,20,24,28] lower half
    //   [32,40,48,56,64,80,96,112,128,160,192,224,0,64,128,192] upper half
    v128_t lut_lo_lower_u8x16 = wasm_i8x16_const(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28);
    v128_t lut_lo_upper_u8x16 = wasm_u8x16_const(32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 0, 64, 128, 192);
    v128_t magnitude_mask_u8x16 = wasm_u8x16_splat(0x1F);
    v128_t nibble_mask_u8x16 = wasm_u8x16_splat(0x0F);
    v128_t half_select_u8x16 = wasm_u8x16_splat(0x10);
    v128_t hi_threshold_u8x16 = wasm_u8x16_splat(28);
    v128_t sign_mask_u8x16 = wasm_u8x16_splat(0x20);
    v128_t sum_i32x4 = wasm_i32x4_splat(0);
    v128_t a_e3m2_u8x16, b_e3m2_u8x16;

nk_dot_e3m2_v128relaxed_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        a_e3m2_u8x16 = a_vec.v128;
        b_e3m2_u8x16 = b_vec.v128;
        count_scalars = 0;
    }
    else {
        a_e3m2_u8x16 = wasm_v128_load(a_scalars);
        b_e3m2_u8x16 = wasm_v128_load(b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }

    // Extract 5-bit magnitude indices
    v128_t a_magnitude_u8x16 = wasm_v128_and(a_e3m2_u8x16, magnitude_mask_u8x16);
    v128_t b_magnitude_u8x16 = wasm_v128_and(b_e3m2_u8x16, magnitude_mask_u8x16);

    // Dual swizzle + bitselect for 32-entry low-byte LUT (a)
    v128_t a_shuffle_index_u8x16 = wasm_v128_and(a_magnitude_u8x16, nibble_mask_u8x16);
    v128_t a_lower_u8x16 = wasm_i8x16_relaxed_swizzle(lut_lo_lower_u8x16, a_shuffle_index_u8x16);
    v128_t a_upper_u8x16 = wasm_i8x16_relaxed_swizzle(lut_lo_upper_u8x16, a_shuffle_index_u8x16);
    v128_t a_upper_select_u8x16 = wasm_i8x16_eq(wasm_v128_and(a_magnitude_u8x16, half_select_u8x16), half_select_u8x16);
    v128_t a_lo_bytes_u8x16 = wasm_i8x16_relaxed_laneselect(a_upper_u8x16, a_lower_u8x16, a_upper_select_u8x16);

    // High byte is 1 iff magnitude index >= 28 (values 256, 320, 384, 448), else 0
    v128_t a_hi_bytes_u8x16 = wasm_v128_and(wasm_u8x16_ge(a_magnitude_u8x16, hi_threshold_u8x16), wasm_u8x16_splat(1));

    // Dual swizzle + bitselect for 32-entry low-byte LUT (b)
    v128_t b_shuffle_index_u8x16 = wasm_v128_and(b_magnitude_u8x16, nibble_mask_u8x16);
    v128_t b_lower_u8x16 = wasm_i8x16_relaxed_swizzle(lut_lo_lower_u8x16, b_shuffle_index_u8x16);
    v128_t b_upper_u8x16 = wasm_i8x16_relaxed_swizzle(lut_lo_upper_u8x16, b_shuffle_index_u8x16);
    v128_t b_upper_select_u8x16 = wasm_i8x16_eq(wasm_v128_and(b_magnitude_u8x16, half_select_u8x16), half_select_u8x16);
    v128_t b_lo_bytes_u8x16 = wasm_i8x16_relaxed_laneselect(b_upper_u8x16, b_lower_u8x16, b_upper_select_u8x16);

    // High byte is 1 iff magnitude index >= 28
    v128_t b_hi_bytes_u8x16 = wasm_v128_and(wasm_u8x16_ge(b_magnitude_u8x16, hi_threshold_u8x16), wasm_u8x16_splat(1));

    // Combine low and high bytes into i16 via byte interleave shuffle (little-endian: low byte first)
    v128_t a_unsigned_low_i16x8 = wasm_i8x16_shuffle(a_lo_bytes_u8x16, a_hi_bytes_u8x16, 0, 16, 1, 17, 2, 18, 3, 19, 4,
                                                     20, 5, 21, 6, 22, 7, 23);
    v128_t a_unsigned_high_i16x8 = wasm_i8x16_shuffle(a_lo_bytes_u8x16, a_hi_bytes_u8x16, 8, 24, 9, 25, 10, 26, 11, 27,
                                                      12, 28, 13, 29, 14, 30, 15, 31);
    v128_t b_unsigned_low_i16x8 = wasm_i8x16_shuffle(b_lo_bytes_u8x16, b_hi_bytes_u8x16, 0, 16, 1, 17, 2, 18, 3, 19, 4,
                                                     20, 5, 21, 6, 22, 7, 23);
    v128_t b_unsigned_high_i16x8 = wasm_i8x16_shuffle(b_lo_bytes_u8x16, b_hi_bytes_u8x16, 8, 24, 9, 25, 10, 26, 11, 27,
                                                      12, 28, 13, 29, 14, 30, 15, 31);

    // Combined sign: XOR sign bits, negate only b (saves ~15 ops vs independent negation)
    v128_t sign_combined_u8x16 = wasm_v128_and(wasm_v128_xor(a_e3m2_u8x16, b_e3m2_u8x16), sign_mask_u8x16);
    v128_t negate_mask_u8x16 = wasm_i8x16_eq(sign_combined_u8x16, sign_mask_u8x16);
    v128_t negate_low_i16x8 = wasm_i8x16_shuffle(negate_mask_u8x16, negate_mask_u8x16, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5,
                                                 5, 6, 6, 7, 7);
    v128_t negate_high_i16x8 = wasm_i8x16_shuffle(negate_mask_u8x16, negate_mask_u8x16, 8, 8, 9, 9, 10, 10, 11, 11, 12,
                                                  12, 13, 13, 14, 14, 15, 15);
    b_unsigned_low_i16x8 = wasm_i16x8_relaxed_laneselect(wasm_i16x8_neg(b_unsigned_low_i16x8), b_unsigned_low_i16x8,
                                                         negate_low_i16x8);
    b_unsigned_high_i16x8 = wasm_i16x8_relaxed_laneselect(wasm_i16x8_neg(b_unsigned_high_i16x8), b_unsigned_high_i16x8,
                                                          negate_high_i16x8);

    // Widening multiply: i16×i16 → i32, accumulate (a is unsigned magnitude, b has combined sign)
    sum_i32x4 = wasm_i32x4_add(sum_i32x4, wasm_i32x4_extmul_low_i16x8(a_unsigned_low_i16x8, b_unsigned_low_i16x8));
    sum_i32x4 = wasm_i32x4_add(sum_i32x4, wasm_i32x4_extmul_high_i16x8(a_unsigned_low_i16x8, b_unsigned_low_i16x8));
    sum_i32x4 = wasm_i32x4_add(sum_i32x4, wasm_i32x4_extmul_low_i16x8(a_unsigned_high_i16x8, b_unsigned_high_i16x8));
    sum_i32x4 = wasm_i32x4_add(sum_i32x4, wasm_i32x4_extmul_high_i16x8(a_unsigned_high_i16x8, b_unsigned_high_i16x8));

    if (count_scalars) goto nk_dot_e3m2_v128relaxed_cycle;
    *result = (nk_f32_t)nk_reduce_add_i32x4_v128relaxed_(sum_i32x4) / 256.0f;
}

NK_PUBLIC void nk_dot_u1_v128relaxed(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result) {
    nk_u8_t const *a_bytes = (nk_u8_t const *)a;
    nk_u8_t const *b_bytes = (nk_u8_t const *)b;
    nk_size_t n_bytes = nk_size_divide_round_up_(n_bits, NK_BITS_PER_BYTE);

    nk_u32_t dot = 0;
    nk_size_t i = 0;

    // Windowed accumulation loop
    while (i + 16 <= n_bytes) {
        v128_t popcount_u8x16 = wasm_i8x16_splat(0);

        // Inner loop: accumulate 31 iterations in u8 before widening
        nk_size_t cycle = 0;
        for (; cycle < 31 && i + 16 <= n_bytes; ++cycle, i += 16) {
            v128_t a_u8x16 = wasm_v128_load(a_bytes + i);
            v128_t b_u8x16 = wasm_v128_load(b_bytes + i);

            // AND to find shared bits (dot product of binary vectors)
            v128_t and_u8x16 = wasm_v128_and(a_u8x16, b_u8x16);

            // Popcount each byte
            v128_t popcnt_u8x16 = wasm_i8x16_popcnt(and_u8x16);

            // Accumulate in u8 (safe: 31 × 8 = 248 < 255)
            popcount_u8x16 = wasm_i8x16_add(popcount_u8x16, popcnt_u8x16);
        }

        // Widen once per window: u8 → u16 → u32
        dot += nk_reduce_add_u8x16_v128relaxed_(popcount_u8x16);
    }

    // Handle tail bytes
    for (; i < n_bytes; i++) {
        nk_u8_t and_byte = a_bytes[i] & b_bytes[i];
        dot += nk_u1x8_popcount_(and_byte);
    }

    *result = dot;
}

/**
 *  Stateful GEMM kernels for batched dot products (4-way parallel accumulation).
 *  Used by nk_define_cross_packed_ / nk_define_cross_compensated_packed_ macros.
 */

typedef struct nk_dot_through_f32x4_state_v128relaxed_t_ {
    v128_t sum_f32x4;
} nk_dot_through_f32x4_state_v128relaxed_t_;

NK_INTERNAL void nk_dot_through_f32x4_init_v128relaxed_(nk_dot_through_f32x4_state_v128relaxed_t_ *state) {
    state->sum_f32x4 = wasm_f32x4_splat(0.0f);
}

NK_INTERNAL void nk_dot_through_f32x4_update_v128relaxed_(nk_dot_through_f32x4_state_v128relaxed_t_ *state,
                                                          nk_b128_vec_t a, nk_b128_vec_t b, nk_size_t depth_offset,
                                                          nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    state->sum_f32x4 = wasm_f32x4_relaxed_madd(a.v128, b.v128, state->sum_f32x4);
}

NK_INTERNAL void nk_dot_through_f32x4_finalize_v128relaxed_( //
    nk_dot_through_f32x4_state_v128relaxed_t_ const *state_a, nk_dot_through_f32x4_state_v128relaxed_t_ const *state_b,
    nk_dot_through_f32x4_state_v128relaxed_t_ const *state_c, nk_dot_through_f32x4_state_v128relaxed_t_ const *state_d,
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f32s[0] = nk_reduce_add_f32x4_v128relaxed_(state_a->sum_f32x4);
    result->f32s[1] = nk_reduce_add_f32x4_v128relaxed_(state_b->sum_f32x4);
    result->f32s[2] = nk_reduce_add_f32x4_v128relaxed_(state_c->sum_f32x4);
    result->f32s[3] = nk_reduce_add_f32x4_v128relaxed_(state_d->sum_f32x4);
}

typedef struct nk_dot_f32x2_state_v128relaxed_t {
    v128_t sum_f64x2;
} nk_dot_f32x2_state_v128relaxed_t;

NK_INTERNAL void nk_dot_f32x2_init_v128relaxed(nk_dot_f32x2_state_v128relaxed_t *state) {
    state->sum_f64x2 = wasm_f64x2_splat(0.0);
}

NK_INTERNAL void nk_dot_f32x2_update_v128relaxed(nk_dot_f32x2_state_v128relaxed_t *state, nk_b64_vec_t a,
                                                 nk_b64_vec_t b, nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    v128_t a_f32x2 = wasm_v128_load64_zero(&a.u64);
    v128_t b_f32x2 = wasm_v128_load64_zero(&b.u64);
    v128_t a_f64x2 = wasm_f64x2_promote_low_f32x4(a_f32x2);
    v128_t b_f64x2 = wasm_f64x2_promote_low_f32x4(b_f32x2);
    state->sum_f64x2 = wasm_f64x2_relaxed_madd(a_f64x2, b_f64x2, state->sum_f64x2);
}

NK_INTERNAL void nk_dot_f32x2_finalize_v128relaxed(                                                   //
    nk_dot_f32x2_state_v128relaxed_t const *state_a, nk_dot_f32x2_state_v128relaxed_t const *state_b, //
    nk_dot_f32x2_state_v128relaxed_t const *state_c, nk_dot_f32x2_state_v128relaxed_t const *state_d, //
    nk_size_t total_dimensions, nk_b256_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f64s[0] = nk_reduce_add_f64x2_v128relaxed_(state_a->sum_f64x2);
    result->f64s[1] = nk_reduce_add_f64x2_v128relaxed_(state_b->sum_f64x2);
    result->f64s[2] = nk_reduce_add_f64x2_v128relaxed_(state_c->sum_f64x2);
    result->f64s[3] = nk_reduce_add_f64x2_v128relaxed_(state_d->sum_f64x2);
}

typedef struct nk_dot_f64x2_state_v128relaxed_t {
    v128_t sum_f64x2;
    v128_t compensation_f64x2;
} nk_dot_f64x2_state_v128relaxed_t;

NK_INTERNAL void nk_dot_f64x2_init_v128relaxed(nk_dot_f64x2_state_v128relaxed_t *state) {
    state->sum_f64x2 = wasm_f64x2_splat(0.0);
    state->compensation_f64x2 = wasm_f64x2_splat(0.0);
}

NK_INTERNAL void nk_dot_f64x2_update_v128relaxed(nk_dot_f64x2_state_v128relaxed_t *state, nk_b128_vec_t a,
                                                 nk_b128_vec_t b, nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    v128_t product_f64x2 = wasm_f64x2_mul(a.v128, b.v128);
    v128_t product_error_f64x2 = wasm_f64x2_sub(wasm_f64x2_relaxed_madd(a.v128, b.v128, wasm_f64x2_splat(0.0)),
                                                product_f64x2);
    v128_t tentative_sum_f64x2 = wasm_f64x2_add(state->sum_f64x2, product_f64x2);
    v128_t virtual_addend_f64x2 = wasm_f64x2_sub(tentative_sum_f64x2, state->sum_f64x2);
    v128_t sum_error_f64x2 = wasm_f64x2_add(
        wasm_f64x2_sub(state->sum_f64x2, wasm_f64x2_sub(tentative_sum_f64x2, virtual_addend_f64x2)),
        wasm_f64x2_sub(product_f64x2, virtual_addend_f64x2));
    state->sum_f64x2 = tentative_sum_f64x2;
    state->compensation_f64x2 = wasm_f64x2_add(state->compensation_f64x2,
                                               wasm_f64x2_add(sum_error_f64x2, product_error_f64x2));
}

NK_INTERNAL void nk_dot_f64x2_finalize_v128relaxed(                                                   //
    nk_dot_f64x2_state_v128relaxed_t const *state_a, nk_dot_f64x2_state_v128relaxed_t const *state_b, //
    nk_dot_f64x2_state_v128relaxed_t const *state_c, nk_dot_f64x2_state_v128relaxed_t const *state_d, //
    nk_size_t total_dimensions, nk_b256_vec_t *result) {
    nk_unused_(total_dimensions);
    result->f64s[0] = nk_dot_stable_sum_f64x2_v128relaxed_(state_a->sum_f64x2, state_a->compensation_f64x2);
    result->f64s[1] = nk_dot_stable_sum_f64x2_v128relaxed_(state_b->sum_f64x2, state_b->compensation_f64x2);
    result->f64s[2] = nk_dot_stable_sum_f64x2_v128relaxed_(state_c->sum_f64x2, state_c->compensation_f64x2);
    result->f64s[3] = nk_dot_stable_sum_f64x2_v128relaxed_(state_d->sum_f64x2, state_d->compensation_f64x2);
}

NK_INTERNAL void nk_load_bf16x4_to_f32x4_v128relaxed_(void const *src, nk_b128_vec_t *dst) {
    nk_b64_vec_t raw;
    nk_copy_bytes_(&raw, src, 8);
    *dst = nk_bf16x4_to_f32x4_v128relaxed_(raw);
}

NK_INTERNAL void nk_partial_load_bf16x4_to_f32x4_v128relaxed_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    nk_b64_vec_t raw = {0};
    nk_copy_bytes_(&raw, src, n * sizeof(nk_bf16_t));
    *dst = nk_bf16x4_to_f32x4_v128relaxed_(raw);
}

NK_INTERNAL void nk_load_f16x4_to_f32x4_v128relaxed_(void const *src, nk_b128_vec_t *dst) {
    nk_b64_vec_t raw;
    nk_copy_bytes_(&raw, src, 8);
    *dst = nk_f16x4_to_f32x4_v128relaxed_(raw);
}

NK_INTERNAL void nk_partial_load_f16x4_to_f32x4_v128relaxed_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    nk_b64_vec_t raw = {0};
    nk_copy_bytes_(&raw, src, n * sizeof(nk_f16_t));
    *dst = nk_f16x4_to_f32x4_v128relaxed_(raw);
}

typedef struct nk_dot_i8x16_state_v128relaxed_t {
    v128_t product_sum_i32x4;    // relaxed_dot accumulator
    v128_t negative_sum_a_i32x4; // Σ(a[i] where b[i]<0), widened to i32
} nk_dot_i8x16_state_v128relaxed_t;

NK_INTERNAL void nk_dot_i8x16_init_v128relaxed(nk_dot_i8x16_state_v128relaxed_t *state) {
    state->product_sum_i32x4 = wasm_i32x4_splat(0);
    state->negative_sum_a_i32x4 = wasm_i32x4_splat(0);
}

NK_INTERNAL void nk_dot_i8x16_update_v128relaxed(nk_dot_i8x16_state_v128relaxed_t *state, nk_b128_vec_t a,
                                                 nk_b128_vec_t b, nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Bit-split: b = b_lo + (-128)·b_hi where b_lo = b & 0x7F ∈ [0,127], b_hi = b >> 7 ∈ {0,1}
    // So a·b = a·b_lo − 128·a·b_hi, both operands fit i7 for relaxed_dot
    v128_t b_lo_u8x16 = wasm_v128_and(b.v128, wasm_i8x16_splat(0x7F));
    v128_t b_hi_u8x16 = wasm_u8x16_shr(b.v128, 7);
    state->product_sum_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a.v128, b_lo_u8x16, state->product_sum_i32x4);
    state->negative_sum_a_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a.v128, b_hi_u8x16,
                                                                         state->negative_sum_a_i32x4);
}

NK_INTERNAL void nk_dot_i8x16_finalize_v128relaxed(                                                   //
    nk_dot_i8x16_state_v128relaxed_t const *state_a, nk_dot_i8x16_state_v128relaxed_t const *state_b, //
    nk_dot_i8x16_state_v128relaxed_t const *state_c, nk_dot_i8x16_state_v128relaxed_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // For each state: result = reduce(product_sum) − 128 × reduce(negative_sum_a)
    result->i32s[0] = nk_reduce_add_i32x4_v128relaxed_(state_a->product_sum_i32x4) -
                      128 * nk_reduce_add_i32x4_v128relaxed_(state_a->negative_sum_a_i32x4);
    result->i32s[1] = nk_reduce_add_i32x4_v128relaxed_(state_b->product_sum_i32x4) -
                      128 * nk_reduce_add_i32x4_v128relaxed_(state_b->negative_sum_a_i32x4);
    result->i32s[2] = nk_reduce_add_i32x4_v128relaxed_(state_c->product_sum_i32x4) -
                      128 * nk_reduce_add_i32x4_v128relaxed_(state_c->negative_sum_a_i32x4);
    result->i32s[3] = nk_reduce_add_i32x4_v128relaxed_(state_d->product_sum_i32x4) -
                      128 * nk_reduce_add_i32x4_v128relaxed_(state_d->negative_sum_a_i32x4);
}

typedef struct nk_dot_u8x16_state_v128relaxed_t {
    v128_t product_lo_i32x4; // relaxed_dot(a_signed, b_lo) accumulator
    v128_t product_hi_i32x4; // relaxed_dot(a_signed, b_hi) accumulator
} nk_dot_u8x16_state_v128relaxed_t;

NK_INTERNAL void nk_dot_u8x16_init_v128relaxed(nk_dot_u8x16_state_v128relaxed_t *state) {
    state->product_lo_i32x4 = wasm_i32x4_splat(0);
    state->product_hi_i32x4 = wasm_i32x4_splat(0);
}

NK_INTERNAL void nk_dot_u8x16_update_v128relaxed(nk_dot_u8x16_state_v128relaxed_t *state, nk_b128_vec_t a,
                                                 nk_b128_vec_t b, nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Bit-split b: b = b_lo + 128·b_hi, with a_signed = a ^ 0x80 = a - 128 (reinterpret u8 as i8)
    // Σ a·b = Σ(a_signed+128)·(b_lo+128·b_hi) = relaxed_dot(a_signed,b_lo) + 128·relaxed_dot(a_signed,b_hi) + 128·Σb
    v128_t a_signed_i8x16 = wasm_v128_xor(a.v128, wasm_i8x16_splat((signed char)0x80));
    v128_t b_lo_u8x16 = wasm_v128_and(b.v128, wasm_i8x16_splat(0x7F));
    v128_t b_hi_u8x16 = wasm_u8x16_shr(b.v128, 7);
    state->product_lo_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_signed_i8x16, b_lo_u8x16,
                                                                     state->product_lo_i32x4);
    state->product_hi_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_signed_i8x16, b_hi_u8x16,
                                                                     state->product_hi_i32x4);
}

NK_INTERNAL void nk_dot_u8x16_finalize_v128relaxed(                                                   //
    nk_dot_u8x16_state_v128relaxed_t const *state_a, nk_dot_u8x16_state_v128relaxed_t const *state_b, //
    nk_dot_u8x16_state_v128relaxed_t const *state_c, nk_dot_u8x16_state_v128relaxed_t const *state_d, //
    nk_size_t total_dimensions, nk_u32_t a_sum, nk_b128_vec_t b_sums, nk_b128_vec_t *result) {
    nk_unused_(a_sum);
    // Σ a·b = reduce(lo) + 128·reduce(hi) + 128·Σb
    result->u32s[0] = (nk_u32_t)(nk_reduce_add_i32x4_v128relaxed_(state_a->product_lo_i32x4) +
                                 128 * nk_reduce_add_i32x4_v128relaxed_(state_a->product_hi_i32x4) +
                                 128 * (nk_i32_t)b_sums.u32s[0]);
    result->u32s[1] = (nk_u32_t)(nk_reduce_add_i32x4_v128relaxed_(state_b->product_lo_i32x4) +
                                 128 * nk_reduce_add_i32x4_v128relaxed_(state_b->product_hi_i32x4) +
                                 128 * (nk_i32_t)b_sums.u32s[1]);
    result->u32s[2] = (nk_u32_t)(nk_reduce_add_i32x4_v128relaxed_(state_c->product_lo_i32x4) +
                                 128 * nk_reduce_add_i32x4_v128relaxed_(state_c->product_hi_i32x4) +
                                 128 * (nk_i32_t)b_sums.u32s[2]);
    result->u32s[3] = (nk_u32_t)(nk_reduce_add_i32x4_v128relaxed_(state_d->product_lo_i32x4) +
                                 128 * nk_reduce_add_i32x4_v128relaxed_(state_d->product_hi_i32x4) +
                                 128 * (nk_i32_t)b_sums.u32s[3]);
}

typedef struct nk_sum_u8x16_state_v128relaxed_t {
    v128_t sum_u32x4;
} nk_sum_u8x16_state_v128relaxed_t;

NK_INTERNAL void nk_sum_u8x16_init_v128relaxed(nk_sum_u8x16_state_v128relaxed_t *state) {
    state->sum_u32x4 = wasm_i32x4_splat(0);
}

NK_INTERNAL void nk_sum_u8x16_update_v128relaxed(nk_sum_u8x16_state_v128relaxed_t *state, nk_b128_vec_t v) {
    v128_t sum_u16x8 = wasm_u16x8_extadd_pairwise_u8x16(v.v128);
    v128_t sum_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(sum_u16x8);
    state->sum_u32x4 = wasm_i32x4_add(state->sum_u32x4, sum_u32x4);
}

NK_INTERNAL nk_u32_t nk_sum_u8x16_finalize_v128relaxed(nk_sum_u8x16_state_v128relaxed_t const *state, nk_size_t count) {
    nk_unused_(count);
    return nk_reduce_add_u32x4_v128relaxed_(state->sum_u32x4);
}

typedef struct nk_dot_e2m3x16_state_v128relaxed_t {
    v128_t sum_i32x4; // relaxed_dot accumulator (a_signed × b_unsigned)
} nk_dot_e2m3x16_state_v128relaxed_t;

NK_INTERNAL void nk_dot_e2m3x16_init_v128relaxed(nk_dot_e2m3x16_state_v128relaxed_t *state) {
    state->sum_i32x4 = wasm_i32x4_splat(0);
}

NK_INTERNAL void nk_dot_e2m3x16_update_v128relaxed(nk_dot_e2m3x16_state_v128relaxed_t *state, nk_b128_vec_t a,
                                                   nk_b128_vec_t b, nk_size_t depth_offset,
                                                   nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Same LUT-based approach as 1:1 dot, accumulating into state
    v128_t lut_lower_u8x16 = wasm_i8x16_const(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    v128_t lut_upper_u8x16 = wasm_i8x16_const(32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120);
    v128_t magnitude_mask_u8x16 = wasm_u8x16_splat(0x1F);
    v128_t nibble_mask_u8x16 = wasm_u8x16_splat(0x0F);
    v128_t half_select_u8x16 = wasm_u8x16_splat(0x10);
    v128_t sign_mask_u8x16 = wasm_u8x16_splat(0x20);

    // Extract magnitude indices
    v128_t a_mag_u8x16 = wasm_v128_and(a.v128, magnitude_mask_u8x16);
    v128_t b_mag_u8x16 = wasm_v128_and(b.v128, magnitude_mask_u8x16);

    // Dual swizzle LUT for a
    v128_t a_idx_u8x16 = wasm_v128_and(a_mag_u8x16, nibble_mask_u8x16);
    v128_t a_lo_u8x16 = wasm_i8x16_relaxed_swizzle(lut_lower_u8x16, a_idx_u8x16);
    v128_t a_hi_u8x16 = wasm_i8x16_relaxed_swizzle(lut_upper_u8x16, a_idx_u8x16);
    v128_t a_sel_u8x16 = wasm_i8x16_eq(wasm_v128_and(a_mag_u8x16, half_select_u8x16), half_select_u8x16);
    v128_t a_unsigned_u8x16 = wasm_i8x16_relaxed_laneselect(a_hi_u8x16, a_lo_u8x16, a_sel_u8x16);

    // Dual swizzle LUT for b
    v128_t b_idx_u8x16 = wasm_v128_and(b_mag_u8x16, nibble_mask_u8x16);
    v128_t b_lo_u8x16 = wasm_i8x16_relaxed_swizzle(lut_lower_u8x16, b_idx_u8x16);
    v128_t b_hi_u8x16 = wasm_i8x16_relaxed_swizzle(lut_upper_u8x16, b_idx_u8x16);
    v128_t b_sel_u8x16 = wasm_i8x16_eq(wasm_v128_and(b_mag_u8x16, half_select_u8x16), half_select_u8x16);
    v128_t b_unsigned_u8x16 = wasm_i8x16_relaxed_laneselect(b_hi_u8x16, b_lo_u8x16, b_sel_u8x16);

    // Combined sign → apply to a (relaxed_dot wants i8 × u7)
    v128_t sign_u8x16 = wasm_v128_and(wasm_v128_xor(a.v128, b.v128), sign_mask_u8x16);
    v128_t neg_mask_u8x16 = wasm_i8x16_eq(sign_u8x16, sign_mask_u8x16);
    v128_t a_neg_u8x16 = wasm_i8x16_neg(a_unsigned_u8x16);
    v128_t a_signed_i8x16 = wasm_i8x16_relaxed_laneselect(a_neg_u8x16, a_unsigned_u8x16, neg_mask_u8x16);

    // relaxed_dot accumulate
    state->sum_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_signed_i8x16, b_unsigned_u8x16, state->sum_i32x4);
}

NK_INTERNAL void nk_dot_e2m3x16_finalize_v128relaxed(                                                     //
    nk_dot_e2m3x16_state_v128relaxed_t const *state_a, nk_dot_e2m3x16_state_v128relaxed_t const *state_b, //
    nk_dot_e2m3x16_state_v128relaxed_t const *state_c, nk_dot_e2m3x16_state_v128relaxed_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // Standard 4-way reduce, divide by 256.0f (LUT values are scaled ×16 for each operand)
    nk_f32_t inv_256 = 1.0f / 256.0f;
    result->f32s[0] = (nk_f32_t)nk_reduce_add_i32x4_v128relaxed_(state_a->sum_i32x4) * inv_256;
    result->f32s[1] = (nk_f32_t)nk_reduce_add_i32x4_v128relaxed_(state_b->sum_i32x4) * inv_256;
    result->f32s[2] = (nk_f32_t)nk_reduce_add_i32x4_v128relaxed_(state_c->sum_i32x4) * inv_256;
    result->f32s[3] = (nk_f32_t)nk_reduce_add_i32x4_v128relaxed_(state_d->sum_i32x4) * inv_256;
}

typedef struct nk_dot_e3m2x16_state_v128relaxed_t {
    v128_t sum_i32x4;
} nk_dot_e3m2x16_state_v128relaxed_t;

NK_INTERNAL void nk_dot_e3m2x16_init_v128relaxed(nk_dot_e3m2x16_state_v128relaxed_t *state) {
    state->sum_i32x4 = wasm_i32x4_splat(0);
}

NK_INTERNAL void nk_dot_e3m2x16_update_v128relaxed(nk_dot_e3m2x16_state_v128relaxed_t *state, nk_b128_vec_t a,
                                                   nk_b128_vec_t b, nk_size_t depth_offset,
                                                   nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // ×4 scaled LUT — all values ≤ 112, fits u7 for relaxed_dot
    // Indices 0-11 rounded to nearest integer (max error ±0.5 in ×4 domain = ±0.125 in value)
    // Indices 12-31 exact
    v128_t lut_lower_u8x16 = wasm_i8x16_const(0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 5, 6, 7);
    v128_t lut_upper_u8x16 = wasm_i8x16_const(8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112);
    v128_t magnitude_mask_u8x16 = wasm_u8x16_splat(0x1F);
    v128_t nibble_mask_u8x16 = wasm_u8x16_splat(0x0F);
    v128_t half_select_u8x16 = wasm_u8x16_splat(0x10);
    v128_t sign_mask_u8x16 = wasm_u8x16_splat(0x20);

    v128_t a_mag_u8x16 = wasm_v128_and(a.v128, magnitude_mask_u8x16);
    v128_t b_mag_u8x16 = wasm_v128_and(b.v128, magnitude_mask_u8x16);

    // Dual swizzle LUT for a
    v128_t a_idx_u8x16 = wasm_v128_and(a_mag_u8x16, nibble_mask_u8x16);
    v128_t a_lo_u8x16 = wasm_i8x16_relaxed_swizzle(lut_lower_u8x16, a_idx_u8x16);
    v128_t a_hi_u8x16 = wasm_i8x16_relaxed_swizzle(lut_upper_u8x16, a_idx_u8x16);
    v128_t a_sel_u8x16 = wasm_i8x16_eq(wasm_v128_and(a_mag_u8x16, half_select_u8x16), half_select_u8x16);
    v128_t a_unsigned_u8x16 = wasm_i8x16_relaxed_laneselect(a_hi_u8x16, a_lo_u8x16, a_sel_u8x16);

    // Dual swizzle LUT for b
    v128_t b_idx_u8x16 = wasm_v128_and(b_mag_u8x16, nibble_mask_u8x16);
    v128_t b_lo_u8x16 = wasm_i8x16_relaxed_swizzle(lut_lower_u8x16, b_idx_u8x16);
    v128_t b_hi_u8x16 = wasm_i8x16_relaxed_swizzle(lut_upper_u8x16, b_idx_u8x16);
    v128_t b_sel_u8x16 = wasm_i8x16_eq(wasm_v128_and(b_mag_u8x16, half_select_u8x16), half_select_u8x16);
    v128_t b_unsigned_u8x16 = wasm_i8x16_relaxed_laneselect(b_hi_u8x16, b_lo_u8x16, b_sel_u8x16);

    // Combined sign → apply to a (relaxed_dot wants i8 × u7)
    v128_t sign_u8x16 = wasm_v128_and(wasm_v128_xor(a.v128, b.v128), sign_mask_u8x16);
    v128_t neg_mask_u8x16 = wasm_i8x16_eq(sign_u8x16, sign_mask_u8x16);
    v128_t a_neg_u8x16 = wasm_i8x16_neg(a_unsigned_u8x16);
    v128_t a_signed_i8x16 = wasm_i8x16_relaxed_laneselect(a_neg_u8x16, a_unsigned_u8x16, neg_mask_u8x16);

    state->sum_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_signed_i8x16, b_unsigned_u8x16, state->sum_i32x4);
}

NK_INTERNAL void nk_dot_e3m2x16_finalize_v128relaxed(                                                     //
    nk_dot_e3m2x16_state_v128relaxed_t const *state_a, nk_dot_e3m2x16_state_v128relaxed_t const *state_b, //
    nk_dot_e3m2x16_state_v128relaxed_t const *state_c, nk_dot_e3m2x16_state_v128relaxed_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    // ×4 per operand → ×16 product scaling → divide by 16
    nk_f32_t inv_16 = 1.0f / 16.0f;
    result->f32s[0] = (nk_f32_t)nk_reduce_add_i32x4_v128relaxed_(state_a->sum_i32x4) * inv_16;
    result->f32s[1] = (nk_f32_t)nk_reduce_add_i32x4_v128relaxed_(state_b->sum_i32x4) * inv_16;
    result->f32s[2] = (nk_f32_t)nk_reduce_add_i32x4_v128relaxed_(state_c->sum_i32x4) * inv_16;
    result->f32s[3] = (nk_f32_t)nk_reduce_add_i32x4_v128relaxed_(state_d->sum_i32x4) * inv_16;
}

typedef struct nk_dot_through_f32x4_state_v128relaxed_t_ nk_dot_e4m3x4_state_v128relaxed_t;
typedef struct nk_dot_through_f32x4_state_v128relaxed_t_ nk_dot_e5m2x4_state_v128relaxed_t;

NK_INTERNAL void nk_load_e4m3x4_to_f32x4_v128relaxed_(void const *src, nk_b128_vec_t *dst) {
    nk_b32_vec_t raw;
    nk_copy_bytes_(&raw, src, 4);
    *dst = nk_e4m3x4_to_f32x4_v128relaxed_(raw);
}

NK_INTERNAL void nk_partial_load_e4m3x4_to_f32x4_v128relaxed_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    nk_b32_vec_t raw = {0};
    nk_copy_bytes_(&raw, src, n * sizeof(nk_e4m3_t));
    *dst = nk_e4m3x4_to_f32x4_v128relaxed_(raw);
}

NK_INTERNAL void nk_load_e5m2x4_to_f32x4_v128relaxed_(void const *src, nk_b128_vec_t *dst) {
    nk_b32_vec_t raw;
    nk_copy_bytes_(&raw, src, 4);
    *dst = nk_e5m2x4_to_f32x4_v128relaxed_(raw);
}

NK_INTERNAL void nk_partial_load_e5m2x4_to_f32x4_v128relaxed_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    nk_b32_vec_t raw = {0};
    nk_copy_bytes_(&raw, src, n * sizeof(nk_e5m2_t));
    *dst = nk_e5m2x4_to_f32x4_v128relaxed_(raw);
}

NK_PUBLIC void nk_dot_e4m3_v128relaxed(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);

nk_dot_e4m3_v128relaxed_cycle:
    if (count_scalars < 4) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_e4m3x4_to_f32x4_v128relaxed_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_e4m3x4_to_f32x4_v128relaxed_(b_scalars, &b_vec, count_scalars);
        sum_f32x4 = wasm_f32x4_relaxed_madd(a_vec.v128, b_vec.v128, sum_f32x4);
        count_scalars = 0;
    }
    else {
        nk_b128_vec_t a_vec, b_vec;
        nk_load_e4m3x4_to_f32x4_v128relaxed_(a_scalars, &a_vec);
        nk_load_e4m3x4_to_f32x4_v128relaxed_(b_scalars, &b_vec);
        sum_f32x4 = wasm_f32x4_relaxed_madd(a_vec.v128, b_vec.v128, sum_f32x4);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    if (count_scalars) goto nk_dot_e4m3_v128relaxed_cycle;

    *result = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
}

NK_PUBLIC void nk_dot_e5m2_v128relaxed(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                       nk_f32_t *result) {
    v128_t sum_f32x4 = wasm_f32x4_splat(0.0f);

nk_dot_e5m2_v128relaxed_cycle:
    if (count_scalars < 4) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_e5m2x4_to_f32x4_v128relaxed_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_e5m2x4_to_f32x4_v128relaxed_(b_scalars, &b_vec, count_scalars);
        sum_f32x4 = wasm_f32x4_relaxed_madd(a_vec.v128, b_vec.v128, sum_f32x4);
        count_scalars = 0;
    }
    else {
        nk_b128_vec_t a_vec, b_vec;
        nk_load_e5m2x4_to_f32x4_v128relaxed_(a_scalars, &a_vec);
        nk_load_e5m2x4_to_f32x4_v128relaxed_(b_scalars, &b_vec);
        sum_f32x4 = wasm_f32x4_relaxed_madd(a_vec.v128, b_vec.v128, sum_f32x4);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    if (count_scalars) goto nk_dot_e5m2_v128relaxed_cycle;

    *result = nk_reduce_add_f32x4_v128relaxed_(sum_f32x4);
}

NK_PUBLIC void nk_dot_u4_v128relaxed(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result) {
    n = nk_size_round_up_to_multiple_(n, 2);
    nk_size_t n_bytes = n / 2;
    v128_t nibble_mask_u8x16 = wasm_u8x16_splat(0x0F);
    v128_t sum_i32x4 = wasm_i32x4_splat(0);
    v128_t a_u4x32, b_u4x32;

nk_dot_u4_v128relaxed_cycle:
    if (n_bytes < 16) {
        nk_b128_vec_t a_vec = {0}, b_vec = {0};
        nk_partial_load_b8x16_serial_(a, &a_vec, n_bytes);
        nk_partial_load_b8x16_serial_(b, &b_vec, n_bytes);
        a_u4x32 = a_vec.v128;
        b_u4x32 = b_vec.v128;
        n_bytes = 0;
    }
    else {
        a_u4x32 = wasm_v128_load(a);
        b_u4x32 = wasm_v128_load(b);
        a = (nk_u4x2_t const *)((nk_u8_t const *)a + 16);
        b = (nk_u4x2_t const *)((nk_u8_t const *)b + 16);
        n_bytes -= 16;
    }

    // Extract low and high nibbles
    v128_t a_low_u8x16 = wasm_v128_and(a_u4x32, nibble_mask_u8x16);
    v128_t a_high_u8x16 = wasm_v128_and(wasm_u16x8_shr(a_u4x32, 4), nibble_mask_u8x16);
    v128_t b_low_u8x16 = wasm_v128_and(b_u4x32, nibble_mask_u8x16);
    v128_t b_high_u8x16 = wasm_v128_and(wasm_u16x8_shr(b_u4x32, 4), nibble_mask_u8x16);

    // Values in [0,15] fit u7 slot directly
    sum_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_low_u8x16, b_low_u8x16, sum_i32x4);
    sum_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_high_u8x16, b_high_u8x16, sum_i32x4);
    if (n_bytes) goto nk_dot_u4_v128relaxed_cycle;

    *result = (nk_u32_t)nk_reduce_add_i32x4_v128relaxed_(sum_i32x4);
}

typedef struct nk_dot_u4x32_state_v128relaxed_t {
    v128_t sum_i32x4;
} nk_dot_u4x32_state_v128relaxed_t;

NK_INTERNAL void nk_dot_u4x32_init_v128relaxed(nk_dot_u4x32_state_v128relaxed_t *state) {
    state->sum_i32x4 = wasm_i32x4_splat(0);
}

NK_INTERNAL void nk_dot_u4x32_update_v128relaxed(nk_dot_u4x32_state_v128relaxed_t *state, nk_b128_vec_t a,
                                                 nk_b128_vec_t b, nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    v128_t nibble_mask_u8x16 = wasm_u8x16_splat(0x0F);
    v128_t a_low_u8x16 = wasm_v128_and(a.v128, nibble_mask_u8x16);
    v128_t a_high_u8x16 = wasm_v128_and(wasm_u16x8_shr(a.v128, 4), nibble_mask_u8x16);
    v128_t b_low_u8x16 = wasm_v128_and(b.v128, nibble_mask_u8x16);
    v128_t b_high_u8x16 = wasm_v128_and(wasm_u16x8_shr(b.v128, 4), nibble_mask_u8x16);
    state->sum_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_low_u8x16, b_low_u8x16, state->sum_i32x4);
    state->sum_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_high_u8x16, b_high_u8x16, state->sum_i32x4);
}

NK_INTERNAL void nk_dot_u4x32_finalize_v128relaxed(                                                   //
    nk_dot_u4x32_state_v128relaxed_t const *state_a, nk_dot_u4x32_state_v128relaxed_t const *state_b, //
    nk_dot_u4x32_state_v128relaxed_t const *state_c, nk_dot_u4x32_state_v128relaxed_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    result->u32s[0] = (nk_u32_t)nk_reduce_add_i32x4_v128relaxed_(state_a->sum_i32x4);
    result->u32s[1] = (nk_u32_t)nk_reduce_add_i32x4_v128relaxed_(state_b->sum_i32x4);
    result->u32s[2] = (nk_u32_t)nk_reduce_add_i32x4_v128relaxed_(state_c->sum_i32x4);
    result->u32s[3] = (nk_u32_t)nk_reduce_add_i32x4_v128relaxed_(state_d->sum_i32x4);
}

NK_PUBLIC void nk_dot_i4_v128relaxed(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result) {
    n = nk_size_round_up_to_multiple_(n, 2);
    nk_size_t n_bytes = n / 2;
    nk_u8_t const *a_bytes = (nk_u8_t const *)a;
    nk_u8_t const *b_bytes = (nk_u8_t const *)b;
    v128_t nibble_mask_u8x16 = wasm_u8x16_splat(0x0F);
    v128_t bias_mask_u8x16 = wasm_u8x16_splat(0x08);
    nk_i64_t cd_total = 0, cx_total = 0, dx_total = 0;
    nk_size_t i = 0;

    // Windowed accumulation loop
    while (i + 16 <= n_bytes) {
        v128_t sum_cd_i32x4 = wasm_i32x4_splat(0);
        v128_t sum_cx_low_u16x8 = wasm_u16x8_splat(0);
        v128_t sum_cx_high_u16x8 = wasm_u16x8_splat(0);
        v128_t sum_dx_low_u16x8 = wasm_u16x8_splat(0);
        v128_t sum_dx_high_u16x8 = wasm_u16x8_splat(0);

        // Inner loop: accumulate 128 iterations before widening
        // Overflow safety: max u16 lane = 128 × 30 = 3840 < 65535
        nk_size_t cycle = 0;
        for (; cycle < 128 && i + 16 <= n_bytes; ++cycle, i += 16) {
            v128_t a_i4x32 = wasm_v128_load(a_bytes + i);
            v128_t b_i4x32 = wasm_v128_load(b_bytes + i);

            // Extract nibbles
            v128_t a_low_u8x16 = wasm_v128_and(a_i4x32, nibble_mask_u8x16);
            v128_t a_high_u8x16 = wasm_v128_and(wasm_u16x8_shr(a_i4x32, 4), nibble_mask_u8x16);
            v128_t b_low_u8x16 = wasm_v128_and(b_i4x32, nibble_mask_u8x16);
            v128_t b_high_u8x16 = wasm_v128_and(wasm_u16x8_shr(b_i4x32, 4), nibble_mask_u8x16);

            // XOR with 8 to get biased values cx, dx in [0,15]
            v128_t c_low_u8x16 = wasm_v128_xor(a_low_u8x16, bias_mask_u8x16);
            v128_t c_high_u8x16 = wasm_v128_xor(a_high_u8x16, bias_mask_u8x16);
            v128_t d_low_u8x16 = wasm_v128_xor(b_low_u8x16, bias_mask_u8x16);
            v128_t d_high_u8x16 = wasm_v128_xor(b_high_u8x16, bias_mask_u8x16);

            // Compute biased dot products
            sum_cd_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(c_low_u8x16, d_low_u8x16, sum_cd_i32x4);
            sum_cd_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(c_high_u8x16, d_high_u8x16, sum_cd_i32x4);

            // Accumulate sums in u16 (1 widening/iter instead of 2)
            sum_cx_low_u16x8 = wasm_i16x8_add(sum_cx_low_u16x8, wasm_u16x8_extadd_pairwise_u8x16(c_low_u8x16));
            sum_cx_high_u16x8 = wasm_i16x8_add(sum_cx_high_u16x8, wasm_u16x8_extadd_pairwise_u8x16(c_high_u8x16));
            sum_dx_low_u16x8 = wasm_i16x8_add(sum_dx_low_u16x8, wasm_u16x8_extadd_pairwise_u8x16(d_low_u8x16));
            sum_dx_high_u16x8 = wasm_i16x8_add(sum_dx_high_u16x8, wasm_u16x8_extadd_pairwise_u8x16(d_high_u8x16));
        }

        // Deferred widening: u16 → u32 once per window
        v128_t sum_cx_u32x4 = wasm_i32x4_add(wasm_u32x4_extadd_pairwise_u16x8(sum_cx_low_u16x8),
                                             wasm_u32x4_extadd_pairwise_u16x8(sum_cx_high_u16x8));
        v128_t sum_dx_u32x4 = wasm_i32x4_add(wasm_u32x4_extadd_pairwise_u16x8(sum_dx_low_u16x8),
                                             wasm_u32x4_extadd_pairwise_u16x8(sum_dx_high_u16x8));
        cd_total += nk_reduce_add_i32x4_v128relaxed_(sum_cd_i32x4);
        cx_total += nk_reduce_add_u32x4_v128relaxed_(sum_cx_u32x4);
        dx_total += nk_reduce_add_u32x4_v128relaxed_(sum_dx_u32x4);
    }

    // SIMD portion covers i*2 elements (2 nibbles per byte)
    nk_i64_t n_simd_elements = (nk_i64_t)i * 2;

    // Scalar tail: use signed helpers directly (no bias/correction needed)
    nk_i64_t tail_dot = 0;
    for (; i < n_bytes; i++) {
        nk_i4x2_t ai = ((nk_i4x2_t const *)a_bytes)[i];
        nk_i4x2_t bi = ((nk_i4x2_t const *)b_bytes)[i];
        tail_dot += (nk_i32_t)nk_i4x2_low_(ai) * (nk_i32_t)nk_i4x2_low_(bi) +
                    (nk_i32_t)nk_i4x2_high_(ai) * (nk_i32_t)nk_i4x2_high_(bi);
    }

    // Apply algebraic correction to SIMD portion only, add unbiased scalar tail
    *result = (nk_i32_t)(cd_total - 8 * (cx_total + dx_total) + 64 * n_simd_elements + tail_dot);
}

typedef struct nk_dot_i4x32_state_v128relaxed_t {
    v128_t biased_product_sum_i32x4;
} nk_dot_i4x32_state_v128relaxed_t;

NK_INTERNAL void nk_dot_i4x32_init_v128relaxed(nk_dot_i4x32_state_v128relaxed_t *state) {
    state->biased_product_sum_i32x4 = wasm_i32x4_splat(0);
}

NK_INTERNAL void nk_dot_i4x32_update_v128relaxed(nk_dot_i4x32_state_v128relaxed_t *state, nk_b128_vec_t a,
                                                 nk_b128_vec_t b, nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    v128_t nibble_mask_u8x16 = wasm_u8x16_splat(0x0F);
    v128_t bias_mask_u8x16 = wasm_u8x16_splat(0x08);
    v128_t a_low_u8x16 = wasm_v128_xor(wasm_v128_and(a.v128, nibble_mask_u8x16), bias_mask_u8x16);
    v128_t a_high_u8x16 = wasm_v128_xor(wasm_v128_and(wasm_u16x8_shr(a.v128, 4), nibble_mask_u8x16), bias_mask_u8x16);
    v128_t b_low_u8x16 = wasm_v128_xor(wasm_v128_and(b.v128, nibble_mask_u8x16), bias_mask_u8x16);
    v128_t b_high_u8x16 = wasm_v128_xor(wasm_v128_and(wasm_u16x8_shr(b.v128, 4), nibble_mask_u8x16), bias_mask_u8x16);
    state->biased_product_sum_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_low_u8x16, b_low_u8x16,
                                                                             state->biased_product_sum_i32x4);
    state->biased_product_sum_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a_high_u8x16, b_high_u8x16,
                                                                             state->biased_product_sum_i32x4);
}

NK_INTERNAL void nk_dot_i4x32_finalize_v128relaxed(                                                   //
    nk_dot_i4x32_state_v128relaxed_t const *state_a, nk_dot_i4x32_state_v128relaxed_t const *state_b, //
    nk_dot_i4x32_state_v128relaxed_t const *state_c, nk_dot_i4x32_state_v128relaxed_t const *state_d, //
    nk_size_t total_dimensions,                                                                       //
    nk_i32_t a_sum, /* Row sum of A (signed sum of i4 values) */                                      //
    nk_b128_vec_t b_sums, /* 4 × i32 column sums of B */                                              //
    nk_b128_vec_t *result) {
    // Match x86 compensated i4 finalizers: result = biased_dot - 8*(a_sum + b_sum) - 64*depth_padded
    nk_i64_t depth_padded = (nk_i64_t)nk_size_round_up_to_multiple_(total_dimensions, 32);
    result->i32s[0] = nk_reduce_add_i32x4_v128relaxed_(state_a->biased_product_sum_i32x4) -
                      8 * ((nk_i64_t)a_sum + (nk_i64_t)b_sums.i32s[0]) - 64 * depth_padded;
    result->i32s[1] = nk_reduce_add_i32x4_v128relaxed_(state_b->biased_product_sum_i32x4) -
                      8 * ((nk_i64_t)a_sum + (nk_i64_t)b_sums.i32s[1]) - 64 * depth_padded;
    result->i32s[2] = nk_reduce_add_i32x4_v128relaxed_(state_c->biased_product_sum_i32x4) -
                      8 * ((nk_i64_t)a_sum + (nk_i64_t)b_sums.i32s[2]) - 64 * depth_padded;
    result->i32s[3] = nk_reduce_add_i32x4_v128relaxed_(state_d->biased_product_sum_i32x4) -
                      8 * ((nk_i64_t)a_sum + (nk_i64_t)b_sums.i32s[3]) - 64 * depth_padded;
}

typedef struct nk_sum_i4x32_state_v128relaxed_t {
    v128_t sum_i32x4;
} nk_sum_i4x32_state_v128relaxed_t;

NK_INTERNAL void nk_sum_i4x32_init_v128relaxed(nk_sum_i4x32_state_v128relaxed_t *state) {
    state->sum_i32x4 = wasm_i32x4_splat(0);
}

NK_INTERNAL void nk_sum_i4x32_update_v128relaxed(nk_sum_i4x32_state_v128relaxed_t *state, nk_b128_vec_t v) {
    v128_t nibble_mask_u8x16 = wasm_u8x16_splat(0x0F);
    v128_t bias_mask_u8x16 = wasm_u8x16_splat(0x08);
    v128_t low_u8x16 = wasm_v128_xor(wasm_v128_and(v.v128, nibble_mask_u8x16), bias_mask_u8x16);
    v128_t high_u8x16 = wasm_v128_xor(wasm_v128_and(wasm_u16x8_shr(v.v128, 4), nibble_mask_u8x16), bias_mask_u8x16);
    v128_t sum_low_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(wasm_u16x8_extadd_pairwise_u8x16(low_u8x16));
    v128_t sum_high_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(wasm_u16x8_extadd_pairwise_u8x16(high_u8x16));
    v128_t signed_sum_i32x4 = wasm_i32x4_sub(wasm_i32x4_add(sum_low_u32x4, sum_high_u32x4), wasm_i32x4_splat(64));
    state->sum_i32x4 = wasm_i32x4_add(state->sum_i32x4, signed_sum_i32x4);
}

NK_INTERNAL nk_i32_t nk_sum_i4x32_finalize_v128relaxed(nk_sum_i4x32_state_v128relaxed_t const *state, nk_size_t count) {
    nk_unused_(count);
    return nk_reduce_add_i32x4_v128relaxed_(state->sum_i32x4);
}

NK_PUBLIC void nk_dot_f32c_v128relaxed(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                       nk_f64c_t *result) {
    v128_t sum_real_f64x2 = wasm_f64x2_splat(0.0);
    v128_t sum_imag_f64x2 = wasm_f64x2_splat(0.0);
    v128_t sign_flip_i64x2 = wasm_i64x2_const(0, 0x8000000000000000ULL);

    nk_size_t idx_pairs = 0;
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        // Load [real, imag] as 64 bits, promote to f64x2
        v128_t a_f32x2 = wasm_v128_load64_zero((nk_f32_t const *)(a_pairs + idx_pairs));
        v128_t b_f32x2 = wasm_v128_load64_zero((nk_f32_t const *)(b_pairs + idx_pairs));
        v128_t a_real_imag_f64x2 = wasm_f64x2_promote_low_f32x4(a_f32x2);
        v128_t b_real_imag_f64x2 = wasm_f64x2_promote_low_f32x4(b_f32x2);

        // Swap b: [imag, real]
        v128_t b_swapped_f64x2 = wasm_i64x2_shuffle(b_real_imag_f64x2, b_real_imag_f64x2, 1, 0);

        // Accumulate: real part uses a*b directly, imag part uses a*b_swapped
        sum_real_f64x2 = wasm_f64x2_relaxed_madd(a_real_imag_f64x2, b_real_imag_f64x2, sum_real_f64x2);
        sum_imag_f64x2 = wasm_f64x2_relaxed_madd(a_real_imag_f64x2, b_swapped_f64x2, sum_imag_f64x2);
    }

    // Flip sign of lane 1 in sum_real: real = Σ(aᵣ*bᵣ) - Σ(aᵢ*bᵢ)
    sum_real_f64x2 = wasm_v128_xor(sum_real_f64x2, sign_flip_i64x2);

    // Finalize: real = sum_real[0] + sum_real[1], imag = sum_imag[0] + sum_imag[1]
    nk_f64_t real_part = wasm_f64x2_extract_lane(sum_real_f64x2, 0) + wasm_f64x2_extract_lane(sum_real_f64x2, 1);
    nk_f64_t imag_part = wasm_f64x2_extract_lane(sum_imag_f64x2, 0) + wasm_f64x2_extract_lane(sum_imag_f64x2, 1);
    result->real = real_part;
    result->imag = imag_part;
}

NK_PUBLIC void nk_vdot_f32c_v128relaxed(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                        nk_f64c_t *result) {
    v128_t sum_real_f64x2 = wasm_f64x2_splat(0.0);
    v128_t sum_imag_f64x2 = wasm_f64x2_splat(0.0);
    v128_t sign_flip_i64x2 = wasm_i64x2_const(0, 0x8000000000000000ULL);

    nk_size_t idx_pairs = 0;
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        v128_t a_f32x2 = wasm_v128_load64_zero((nk_f32_t const *)(a_pairs + idx_pairs));
        v128_t b_f32x2 = wasm_v128_load64_zero((nk_f32_t const *)(b_pairs + idx_pairs));
        v128_t a_real_imag_f64x2 = wasm_f64x2_promote_low_f32x4(a_f32x2);
        v128_t b_real_imag_f64x2 = wasm_f64x2_promote_low_f32x4(b_f32x2);
        v128_t b_swapped_f64x2 = wasm_i64x2_shuffle(b_real_imag_f64x2, b_real_imag_f64x2, 1, 0);

        sum_real_f64x2 = wasm_f64x2_relaxed_madd(a_real_imag_f64x2, b_real_imag_f64x2, sum_real_f64x2);
        sum_imag_f64x2 = wasm_f64x2_relaxed_madd(a_real_imag_f64x2, b_swapped_f64x2, sum_imag_f64x2);
    }

    // For vdot (conjugate dot): flip sign of imag lane 1 instead
    sum_imag_f64x2 = wasm_v128_xor(sum_imag_f64x2, sign_flip_i64x2);

    nk_f64_t real_part = wasm_f64x2_extract_lane(sum_real_f64x2, 0) + wasm_f64x2_extract_lane(sum_real_f64x2, 1);
    nk_f64_t imag_part = wasm_f64x2_extract_lane(sum_imag_f64x2, 0) + wasm_f64x2_extract_lane(sum_imag_f64x2, 1);
    result->real = real_part;
    result->imag = imag_part;
}

NK_PUBLIC void nk_dot_f64c_v128relaxed(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                       nk_f64c_t *result) {
    v128_t sum_real_f64x2 = wasm_f64x2_splat(0.0);
    v128_t sum_imag_f64x2 = wasm_f64x2_splat(0.0);
    v128_t sign_flip_i64x2 = wasm_i64x2_const(0, 0x8000000000000000ULL);

    nk_size_t idx_pairs = 0;
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        v128_t a_real_imag_f64x2 = wasm_v128_load((nk_f64_t const *)(a_pairs + idx_pairs));
        v128_t b_real_imag_f64x2 = wasm_v128_load((nk_f64_t const *)(b_pairs + idx_pairs));
        v128_t b_swapped_f64x2 = wasm_i64x2_shuffle(b_real_imag_f64x2, b_real_imag_f64x2, 1, 0);

        sum_real_f64x2 = wasm_f64x2_relaxed_madd(a_real_imag_f64x2, b_real_imag_f64x2, sum_real_f64x2);
        sum_imag_f64x2 = wasm_f64x2_relaxed_madd(a_real_imag_f64x2, b_swapped_f64x2, sum_imag_f64x2);
    }

    sum_real_f64x2 = wasm_v128_xor(sum_real_f64x2, sign_flip_i64x2);

    result->real = wasm_f64x2_extract_lane(sum_real_f64x2, 0) + wasm_f64x2_extract_lane(sum_real_f64x2, 1);
    result->imag = wasm_f64x2_extract_lane(sum_imag_f64x2, 0) + wasm_f64x2_extract_lane(sum_imag_f64x2, 1);
}

NK_PUBLIC void nk_vdot_f64c_v128relaxed(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                        nk_f64c_t *result) {
    v128_t sum_real_f64x2 = wasm_f64x2_splat(0.0);
    v128_t sum_imag_f64x2 = wasm_f64x2_splat(0.0);
    v128_t sign_flip_i64x2 = wasm_i64x2_const(0, 0x8000000000000000ULL);

    nk_size_t idx_pairs = 0;
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        v128_t a_real_imag_f64x2 = wasm_v128_load((nk_f64_t const *)(a_pairs + idx_pairs));
        v128_t b_real_imag_f64x2 = wasm_v128_load((nk_f64_t const *)(b_pairs + idx_pairs));
        v128_t b_swapped_f64x2 = wasm_i64x2_shuffle(b_real_imag_f64x2, b_real_imag_f64x2, 1, 0);

        sum_real_f64x2 = wasm_f64x2_relaxed_madd(a_real_imag_f64x2, b_real_imag_f64x2, sum_real_f64x2);
        sum_imag_f64x2 = wasm_f64x2_relaxed_madd(a_real_imag_f64x2, b_swapped_f64x2, sum_imag_f64x2);
    }

    sum_imag_f64x2 = wasm_v128_xor(sum_imag_f64x2, sign_flip_i64x2);

    result->real = wasm_f64x2_extract_lane(sum_real_f64x2, 0) + wasm_f64x2_extract_lane(sum_real_f64x2, 1);
    result->imag = wasm_f64x2_extract_lane(sum_imag_f64x2, 0) + wasm_f64x2_extract_lane(sum_imag_f64x2, 1);
}

typedef struct nk_dot_u1x128_state_v128relaxed_t {
    v128_t dot_count_u32x4;
} nk_dot_u1x128_state_v128relaxed_t;

NK_INTERNAL void nk_dot_u1x128_init_v128relaxed(nk_dot_u1x128_state_v128relaxed_t *state) {
    state->dot_count_u32x4 = wasm_u32x4_const(0, 0, 0, 0);
}

NK_INTERNAL void nk_dot_u1x128_update_v128relaxed(nk_dot_u1x128_state_v128relaxed_t *state, nk_b128_vec_t a,
                                                  nk_b128_vec_t b, nk_size_t depth_offset,
                                                  nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    v128_t and_u8x16 = wasm_v128_and(a.v128, b.v128);
    v128_t popcount_u8x16 = wasm_i8x16_popcnt(and_u8x16);
    v128_t popcount_u16x8 = wasm_u16x8_extadd_pairwise_u8x16(popcount_u8x16);
    v128_t popcount_u32x4 = wasm_u32x4_extadd_pairwise_u16x8(popcount_u16x8);
    state->dot_count_u32x4 = wasm_i32x4_add(state->dot_count_u32x4, popcount_u32x4);
}

NK_INTERNAL void nk_dot_u1x128_finalize_v128relaxed(                                                    //
    nk_dot_u1x128_state_v128relaxed_t const *state_a, nk_dot_u1x128_state_v128relaxed_t const *state_b, //
    nk_dot_u1x128_state_v128relaxed_t const *state_c, nk_dot_u1x128_state_v128relaxed_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    v128_t a_u32x4 = state_a->dot_count_u32x4, b_u32x4 = state_b->dot_count_u32x4;
    v128_t c_u32x4 = state_c->dot_count_u32x4, d_u32x4 = state_d->dot_count_u32x4;
    // Step 1: interleave pairs
    v128_t ab_lo_u32x4 = wasm_i32x4_shuffle(a_u32x4, b_u32x4, 0, 4, 1, 5); // a0 b0 a1 b1
    v128_t ab_hi_u32x4 = wasm_i32x4_shuffle(a_u32x4, b_u32x4, 2, 6, 3, 7); // a2 b2 a3 b3
    v128_t cd_lo_u32x4 = wasm_i32x4_shuffle(c_u32x4, d_u32x4, 0, 4, 1, 5); // c0 d0 c1 d1
    v128_t cd_hi_u32x4 = wasm_i32x4_shuffle(c_u32x4, d_u32x4, 2, 6, 3, 7); // c2 d2 c3 d3
    // Step 2: pairwise add
    v128_t sum_02_u32x4 = wasm_i32x4_add(ab_lo_u32x4, ab_hi_u32x4); // a02 b02 a13 b13
    v128_t sum_13_u32x4 = wasm_i32x4_add(cd_lo_u32x4, cd_hi_u32x4); // c02 d02 c13 d13
    // Step 3: final interleave
    v128_t even_u32x4 = wasm_i32x4_shuffle(sum_02_u32x4, sum_13_u32x4, 0, 1, 4, 5);
    v128_t odd_u32x4 = wasm_i32x4_shuffle(sum_02_u32x4, sum_13_u32x4, 2, 3, 6, 7);
    result->v128 = wasm_i32x4_add(even_u32x4, odd_u32x4); // [sum_a, sum_b, sum_c, sum_d]
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_DOT_V128RELAXED_H
