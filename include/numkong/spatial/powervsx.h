/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for Power VSX.
 *  @file include/numkong/spatial/powervsx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_powervsx_instructions Key Power VSX Spatial Instructions
 *
 *  Power ISA 3.0 (POWER9+) VSX instructions for distance computations:
 *
 *      Intrinsic                        Instruction           POWER9
 *      vec_madd(f32)                    XVMADDASP             5cy
 *      vec_mul(f32)                     XVMULSP               5cy
 *      vec_add(f32)                     XVADDSP               5cy
 *      vec_sub(f32)                     XVSUBSP               5cy
 *      vec_rsqrte(f32)                  XVRSQRTESP            5cy
 *      vec_sqrt(f32)                    XVSQRTSP              26cy
 *      vec_doublee                      XVCVSPDP              3cy  (f32 → f64 even elts)
 *      vec_xl_len                       LXVL                  5cy  (partial vector load)
 *      vec_extract_fp32_from_shorth     XVCVHPSP              5cy  (f16 → f32 high half)
 *      vec_extract_fp32_from_shortl     XVCVHPSP              5cy  (f16 → f32 low half)
 *      vec_msum(i8, u8, i32)            VMSUMMBM              5cy  (i8×u8 widening multiply-sum)
 *      vec_msum(u8, u8, u32)            VMSUMUBM              5cy  (u8×u8 widening multiply-sum)
 *      vec_unpackh(i8)                  VUPKHSB               2cy  (sign-extend high 8 i8 → i16x8)
 *      vec_unpackl(i8)                  VUPKLSB               2cy  (sign-extend low 8 i8 → i16x8)
 *
 *  For angular distance, `vec_rsqrte` provides ~12-bit precision. Two Newton-Raphson
 *  iterations achieve ~23-bit precision for f32, three iterations for f64.
 */
#ifndef NK_SPATIAL_POWERVSX_H
#define NK_SPATIAL_POWERVSX_H

#if NK_TARGET_POWER64_
#if NK_TARGET_POWERVSX

#include "numkong/types.h"
#include "numkong/dot/powervsx.h"    // `nk_hsum_*_powervsx_`, includes cast/powervsx.h
#include "numkong/scalar/powervsx.h" // `nk_f32_sqrt_powervsx`, `nk_f64_sqrt_powervsx`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

/**
 *  @brief Reciprocal square root of 4 floats with Newton-Raphson refinement.
 *
 *  Uses `vec_rsqrte` (~12-bit initial estimate) followed by two Newton-Raphson
 *  iterations, achieving ~23-bit precision sufficient for f32.
 *  NR step: rsqrt = rsqrt × (1.5 − 0.5 × x × rsqrt × rsqrt)
 */
NK_INTERNAL nk_vf32x4_t nk_rsqrt_f32x4_powervsx_(nk_vf32x4_t x) {
    nk_vf32x4_t half_f32x4 = vec_splats(0.5f);
    nk_vf32x4_t three_halves_f32x4 = vec_splats(1.5f);
    nk_vf32x4_t rsqrt_f32x4 = vec_rsqrte(x);
    // Iteration 1
    nk_vf32x4_t nr_f32x4 = vec_sub(three_halves_f32x4,
                                   vec_mul(half_f32x4, vec_mul(x, vec_mul(rsqrt_f32x4, rsqrt_f32x4))));
    rsqrt_f32x4 = vec_mul(rsqrt_f32x4, nr_f32x4);
    // Iteration 2
    nr_f32x4 = vec_sub(three_halves_f32x4, vec_mul(half_f32x4, vec_mul(x, vec_mul(rsqrt_f32x4, rsqrt_f32x4))));
    rsqrt_f32x4 = vec_mul(rsqrt_f32x4, nr_f32x4);
    return rsqrt_f32x4;
}

/**
 *  @brief Reciprocal square root of 2 doubles with Newton-Raphson refinement.
 *
 *  Uses `vec_rsqrte` (~12-bit estimate) followed by three Newton-Raphson
 *  iterations, achieving ~48-bit precision for f64.
 */
NK_INTERNAL nk_vf64x2_t nk_rsqrt_f64x2_powervsx_(nk_vf64x2_t x) {
    nk_vf64x2_t half_f64x2 = vec_splats(0.5);
    nk_vf64x2_t three_halves_f64x2 = vec_splats(1.5);
    nk_vf64x2_t rsqrt_f64x2 = vec_rsqrte(x);
    // Iteration 1
    nk_vf64x2_t nr_f64x2 = vec_sub(three_halves_f64x2,
                                   vec_mul(half_f64x2, vec_mul(x, vec_mul(rsqrt_f64x2, rsqrt_f64x2))));
    rsqrt_f64x2 = vec_mul(rsqrt_f64x2, nr_f64x2);
    // Iteration 2
    nr_f64x2 = vec_sub(three_halves_f64x2, vec_mul(half_f64x2, vec_mul(x, vec_mul(rsqrt_f64x2, rsqrt_f64x2))));
    rsqrt_f64x2 = vec_mul(rsqrt_f64x2, nr_f64x2);
    // Iteration 3
    nr_f64x2 = vec_sub(three_halves_f64x2, vec_mul(half_f64x2, vec_mul(x, vec_mul(rsqrt_f64x2, rsqrt_f64x2))));
    rsqrt_f64x2 = vec_mul(rsqrt_f64x2, nr_f64x2);
    return rsqrt_f64x2;
}

NK_INTERNAL nk_f32_t nk_angular_normalize_f32_powervsx_(nk_f32_t ab, nk_f32_t a2, nk_f32_t b2) {
    if (a2 == 0 && b2 == 0) return 0;
    if (ab == 0) return 1;
    nk_vf32x4_t squares_f32x4 = vec_splats(0.0f);
    squares_f32x4 = vec_insert(a2, squares_f32x4, 0);
    squares_f32x4 = vec_insert(b2, squares_f32x4, 1);
    nk_vf32x4_t rsqrts_f32x4 = nk_rsqrt_f32x4_powervsx_(squares_f32x4);
    nk_f32_t a2_rsqrt = vec_extract(rsqrts_f32x4, 0);
    nk_f32_t b2_rsqrt = vec_extract(rsqrts_f32x4, 1);
    nk_f32_t result = 1 - ab * a2_rsqrt * b2_rsqrt;
    return result > 0 ? result : 0;
}

NK_INTERNAL nk_f64_t nk_angular_normalize_f64_powervsx_(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {
    if (a2 == 0 && b2 == 0) return 0;
    if (ab == 0) return 1;
    nk_vf64x2_t squares_f64x2 = vec_splats(0.0);
    squares_f64x2 = vec_insert(a2, squares_f64x2, 0);
    squares_f64x2 = vec_insert(b2, squares_f64x2, 1);
    nk_vf64x2_t rsqrts_f64x2 = nk_rsqrt_f64x2_powervsx_(squares_f64x2);
    nk_f64_t a2_rsqrt = vec_extract(rsqrts_f64x2, 0);
    nk_f64_t b2_rsqrt = vec_extract(rsqrts_f64x2, 1);
    nk_f64_t result = 1 - ab * a2_rsqrt * b2_rsqrt;
    return result > 0 ? result : 0;
}

#pragma region F32 and F64 Floats

NK_PUBLIC void nk_sqeuclidean_f32_powervsx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    // Accumulate in f64 for numerical stability using vec_doublee/vec_doubleo (f32 → f64)
    nk_vf64x2_t sum_even_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf64x2_t sum_odd_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf32x4_t a_f32x4, b_f32x4;
    nk_size_t tail_bytes;

nk_sqeuclidean_f32_powervsx_cycle:
    if (n < 4) {
        tail_bytes = n * sizeof(nk_f32_t);
        a_f32x4 = vec_xl_len((nk_f32_t *)a, tail_bytes);
        b_f32x4 = vec_xl_len((nk_f32_t *)b, tail_bytes);
        n = 0;
    }
    else {
        a_f32x4 = vec_xl(0, a);
        b_f32x4 = vec_xl(0, b);
        a += 4, b += 4, n -= 4;
    }
    // Widen a and b to f64 before subtraction to avoid f32 precision loss in (a−b)
    nk_vf64x2_t a_even_f64x2 = vec_doublee(a_f32x4);
    nk_vf64x2_t b_even_f64x2 = vec_doublee(b_f32x4);
    nk_vf64x2_t diff_even_f64x2 = vec_sub(a_even_f64x2, b_even_f64x2);
    sum_even_f64x2 = vec_madd(diff_even_f64x2, diff_even_f64x2, sum_even_f64x2);
    nk_vf64x2_t a_odd_f64x2 = vec_doubleo(a_f32x4);
    nk_vf64x2_t b_odd_f64x2 = vec_doubleo(b_f32x4);
    nk_vf64x2_t diff_odd_f64x2 = vec_sub(a_odd_f64x2, b_odd_f64x2);
    sum_odd_f64x2 = vec_madd(diff_odd_f64x2, diff_odd_f64x2, sum_odd_f64x2);
    if (n) goto nk_sqeuclidean_f32_powervsx_cycle;

    nk_vf64x2_t total_f64x2 = vec_add(sum_even_f64x2, sum_odd_f64x2);
    *result = nk_hsum_f64x2_powervsx_(total_f64x2);
}

NK_PUBLIC void nk_euclidean_f32_powervsx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_sqeuclidean_f32_powervsx(a, b, n, result);
    *result = nk_f64_sqrt_powervsx(*result);
}

NK_PUBLIC void nk_angular_f32_powervsx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    // Accumulate in f64 for numerical stability using vec_doublee/vec_doubleo
    nk_vf64x2_t ab_even_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf64x2_t ab_odd_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf64x2_t a2_even_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf64x2_t a2_odd_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf64x2_t b2_even_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf64x2_t b2_odd_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf32x4_t a_f32x4, b_f32x4;
    nk_size_t tail_bytes;

nk_angular_f32_powervsx_cycle:
    if (n < 4) {
        tail_bytes = n * sizeof(nk_f32_t);
        a_f32x4 = vec_xl_len((nk_f32_t *)a, tail_bytes);
        b_f32x4 = vec_xl_len((nk_f32_t *)b, tail_bytes);
        n = 0;
    }
    else {
        a_f32x4 = vec_xl(0, a);
        b_f32x4 = vec_xl(0, b);
        a += 4, b += 4, n -= 4;
    }
    // Even elements (0, 2) → f64
    nk_vf64x2_t a_even_f64x2 = vec_doublee(a_f32x4);
    nk_vf64x2_t b_even_f64x2 = vec_doublee(b_f32x4);
    ab_even_f64x2 = vec_madd(a_even_f64x2, b_even_f64x2, ab_even_f64x2);
    a2_even_f64x2 = vec_madd(a_even_f64x2, a_even_f64x2, a2_even_f64x2);
    b2_even_f64x2 = vec_madd(b_even_f64x2, b_even_f64x2, b2_even_f64x2);
    // Odd elements (1, 3) → f64: rotate by 4 bytes
    nk_vf32x4_t a_rotated_f32x4 = vec_sld(a_f32x4, a_f32x4, 4);
    nk_vf32x4_t b_rotated_f32x4 = vec_sld(b_f32x4, b_f32x4, 4);
    nk_vf64x2_t a_odd_f64x2 = vec_doublee(a_rotated_f32x4);
    nk_vf64x2_t b_odd_f64x2 = vec_doublee(b_rotated_f32x4);
    ab_odd_f64x2 = vec_madd(a_odd_f64x2, b_odd_f64x2, ab_odd_f64x2);
    a2_odd_f64x2 = vec_madd(a_odd_f64x2, a_odd_f64x2, a2_odd_f64x2);
    b2_odd_f64x2 = vec_madd(b_odd_f64x2, b_odd_f64x2, b2_odd_f64x2);
    if (n) goto nk_angular_f32_powervsx_cycle;

    nk_f64_t ab = nk_hsum_f64x2_powervsx_(vec_add(ab_even_f64x2, ab_odd_f64x2));
    nk_f64_t a2 = nk_hsum_f64x2_powervsx_(vec_add(a2_even_f64x2, a2_odd_f64x2));
    nk_f64_t b2 = nk_hsum_f64x2_powervsx_(vec_add(b2_even_f64x2, b2_odd_f64x2));
    *result = nk_angular_normalize_f64_powervsx_(ab, a2, b2);
}

NK_PUBLIC void nk_sqeuclidean_f64_powervsx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_vf64x2_t sum_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf64x2_t a_f64x2, b_f64x2;
    nk_size_t tail_bytes;

nk_sqeuclidean_f64_powervsx_cycle:
    if (n < 2) {
        tail_bytes = n * sizeof(nk_f64_t);
        a_f64x2 = vec_xl_len((nk_f64_t *)a, tail_bytes);
        b_f64x2 = vec_xl_len((nk_f64_t *)b, tail_bytes);
        n = 0;
    }
    else {
        a_f64x2 = vec_xl(0, a);
        b_f64x2 = vec_xl(0, b);
        a += 2, b += 2, n -= 2;
    }
    nk_vf64x2_t diff_f64x2 = vec_sub(a_f64x2, b_f64x2);
    sum_f64x2 = vec_madd(diff_f64x2, diff_f64x2, sum_f64x2);
    if (n) goto nk_sqeuclidean_f64_powervsx_cycle;

    *result = nk_hsum_f64x2_powervsx_(sum_f64x2);
}

NK_PUBLIC void nk_euclidean_f64_powervsx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_sqeuclidean_f64_powervsx(a, b, n, result);
    *result = nk_f64_sqrt_powervsx(*result);
}

NK_PUBLIC void nk_angular_f64_powervsx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    // Dot2 (Ogita-Rump-Oishi) for cross-product ab (may have cancellation),
    // simple FMA for self-products a2/b2 (all positive, no cancellation)
    nk_vf64x2_t ab_sum_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf64x2_t ab_compensation_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf64x2_t a2_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf64x2_t b2_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf64x2_t a_f64x2, b_f64x2;
    nk_size_t tail_bytes;

nk_angular_f64_powervsx_cycle:
    if (n < 2) {
        tail_bytes = n * sizeof(nk_f64_t);
        a_f64x2 = vec_xl_len((nk_f64_t *)a, tail_bytes);
        b_f64x2 = vec_xl_len((nk_f64_t *)b, tail_bytes);
        n = 0;
    }
    else {
        a_f64x2 = vec_xl(0, a);
        b_f64x2 = vec_xl(0, b);
        a += 2, b += 2, n -= 2;
    }
    // TwoProd for ab: product = a×b, error = msub(a, b, product) captures rounding error
    nk_vf64x2_t product_f64x2 = vec_mul(a_f64x2, b_f64x2);
    nk_vf64x2_t product_error_f64x2 = vec_msub(a_f64x2, b_f64x2, product_f64x2);
    // TwoSum: (t, q) = TwoSum(sum, product) where t = sum + product rounded, q = error
    nk_vf64x2_t tentative_sum_f64x2 = vec_add(ab_sum_f64x2, product_f64x2);
    nk_vf64x2_t virtual_addend_f64x2 = vec_sub(tentative_sum_f64x2, ab_sum_f64x2);
    nk_vf64x2_t sum_error_f64x2 = vec_add(vec_sub(ab_sum_f64x2, vec_sub(tentative_sum_f64x2, virtual_addend_f64x2)),
                                          vec_sub(product_f64x2, virtual_addend_f64x2));
    ab_sum_f64x2 = tentative_sum_f64x2;
    ab_compensation_f64x2 = vec_add(ab_compensation_f64x2, vec_add(sum_error_f64x2, product_error_f64x2));
    // Simple FMA for self-products (no cancellation)
    a2_f64x2 = vec_madd(a_f64x2, a_f64x2, a2_f64x2);
    b2_f64x2 = vec_madd(b_f64x2, b_f64x2, b2_f64x2);
    if (n) goto nk_angular_f64_powervsx_cycle;

    *result = nk_angular_normalize_f64_powervsx_(nk_dot_stable_sum_f64x2_powervsx_(ab_sum_f64x2, ab_compensation_f64x2),
                                                 nk_hsum_f64x2_powervsx_(a2_f64x2), nk_hsum_f64x2_powervsx_(b2_f64x2));
}

#pragma endregion F32 and F64 Floats
#pragma region F16 and BF16 Floats

NK_PUBLIC void nk_sqeuclidean_bf16_powervsx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    // bf16 → f32 via merge with zero: places bf16 bits in upper 16 of each f32
    nk_vu16x8_t zero_u16x8 = vec_splats((nk_u16_t)0);
    nk_vf32x4_t sum_f32x4 = vec_splats(0.0f);
    nk_vu16x8_t a_u16x8, b_u16x8;
    nk_size_t tail_bytes;

nk_sqeuclidean_bf16_powervsx_cycle:
    if (n < 8) {
        tail_bytes = n * sizeof(nk_bf16_t);
        a_u16x8 = vec_xl_len((nk_u16_t *)a, tail_bytes);
        b_u16x8 = vec_xl_len((nk_u16_t *)b, tail_bytes);
        n = 0;
    }
    else {
        a_u16x8 = vec_xl(0, (nk_u16_t const *)a);
        b_u16x8 = vec_xl(0, (nk_u16_t const *)b);
        a += 8, b += 8, n -= 8;
    }
    nk_vf32x4_t a_high_f32x4 = (nk_vf32x4_t)vec_mergeh(zero_u16x8, a_u16x8);
    nk_vf32x4_t a_low_f32x4 = (nk_vf32x4_t)vec_mergel(zero_u16x8, a_u16x8);
    nk_vf32x4_t b_high_f32x4 = (nk_vf32x4_t)vec_mergeh(zero_u16x8, b_u16x8);
    nk_vf32x4_t b_low_f32x4 = (nk_vf32x4_t)vec_mergel(zero_u16x8, b_u16x8);
    nk_vf32x4_t diff_high_f32x4 = vec_sub(a_high_f32x4, b_high_f32x4);
    nk_vf32x4_t diff_low_f32x4 = vec_sub(a_low_f32x4, b_low_f32x4);
    sum_f32x4 = vec_madd(diff_high_f32x4, diff_high_f32x4, sum_f32x4);
    sum_f32x4 = vec_madd(diff_low_f32x4, diff_low_f32x4, sum_f32x4);
    if (n) goto nk_sqeuclidean_bf16_powervsx_cycle;
    *result = nk_hsum_f32x4_powervsx_(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_bf16_powervsx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_bf16_powervsx(a, b, n, result);
    *result = nk_f32_sqrt_powervsx(*result);
}

NK_PUBLIC void nk_angular_bf16_powervsx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_vu16x8_t zero_u16x8 = vec_splats((nk_u16_t)0);
    nk_vf32x4_t ab_f32x4 = vec_splats(0.0f);
    nk_vf32x4_t a2_f32x4 = vec_splats(0.0f);
    nk_vf32x4_t b2_f32x4 = vec_splats(0.0f);
    nk_vu16x8_t a_u16x8, b_u16x8;
    nk_size_t tail_bytes;

nk_angular_bf16_powervsx_cycle:
    if (n < 8) {
        tail_bytes = n * sizeof(nk_bf16_t);
        a_u16x8 = vec_xl_len((nk_u16_t *)a, tail_bytes);
        b_u16x8 = vec_xl_len((nk_u16_t *)b, tail_bytes);
        n = 0;
    }
    else {
        a_u16x8 = vec_xl(0, (nk_u16_t const *)a);
        b_u16x8 = vec_xl(0, (nk_u16_t const *)b);
        a += 8, b += 8, n -= 8;
    }
    nk_vf32x4_t a_high_f32x4 = (nk_vf32x4_t)vec_mergeh(zero_u16x8, a_u16x8);
    nk_vf32x4_t a_low_f32x4 = (nk_vf32x4_t)vec_mergel(zero_u16x8, a_u16x8);
    nk_vf32x4_t b_high_f32x4 = (nk_vf32x4_t)vec_mergeh(zero_u16x8, b_u16x8);
    nk_vf32x4_t b_low_f32x4 = (nk_vf32x4_t)vec_mergel(zero_u16x8, b_u16x8);
    ab_f32x4 = vec_madd(a_high_f32x4, b_high_f32x4, ab_f32x4);
    ab_f32x4 = vec_madd(a_low_f32x4, b_low_f32x4, ab_f32x4);
    a2_f32x4 = vec_madd(a_high_f32x4, a_high_f32x4, a2_f32x4);
    a2_f32x4 = vec_madd(a_low_f32x4, a_low_f32x4, a2_f32x4);
    b2_f32x4 = vec_madd(b_high_f32x4, b_high_f32x4, b2_f32x4);
    b2_f32x4 = vec_madd(b_low_f32x4, b_low_f32x4, b2_f32x4);
    if (n) goto nk_angular_bf16_powervsx_cycle;
    nk_f32_t ab = nk_hsum_f32x4_powervsx_(ab_f32x4);
    nk_f32_t a2 = nk_hsum_f32x4_powervsx_(a2_f32x4);
    nk_f32_t b2 = nk_hsum_f32x4_powervsx_(b2_f32x4);
    *result = nk_angular_normalize_f32_powervsx_(ab, a2, b2);
}

NK_PUBLIC void nk_sqeuclidean_f16_powervsx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    // f16 → f32 via POWER9 hardware XVCVHPSP (vec_extract_fp32_from_shorth/shortl)
    nk_vf32x4_t sum_f32x4 = vec_splats(0.0f);
    nk_vu16x8_t a_u16x8, b_u16x8;
    nk_size_t tail_bytes;

nk_sqeuclidean_f16_powervsx_cycle:
    if (n < 8) {
        tail_bytes = n * sizeof(nk_f16_t);
        a_u16x8 = vec_xl_len((nk_u16_t *)a, tail_bytes);
        b_u16x8 = vec_xl_len((nk_u16_t *)b, tail_bytes);
        n = 0;
    }
    else {
        a_u16x8 = vec_xl(0, (nk_u16_t const *)a);
        b_u16x8 = vec_xl(0, (nk_u16_t const *)b);
        a += 8, b += 8, n -= 8;
    }
    nk_vf32x4_t a_high_f32x4 = vec_extract_fp32_from_shorth(a_u16x8);
    nk_vf32x4_t a_low_f32x4 = vec_extract_fp32_from_shortl(a_u16x8);
    nk_vf32x4_t b_high_f32x4 = vec_extract_fp32_from_shorth(b_u16x8);
    nk_vf32x4_t b_low_f32x4 = vec_extract_fp32_from_shortl(b_u16x8);
    nk_vf32x4_t diff_high_f32x4 = vec_sub(a_high_f32x4, b_high_f32x4);
    nk_vf32x4_t diff_low_f32x4 = vec_sub(a_low_f32x4, b_low_f32x4);
    sum_f32x4 = vec_madd(diff_high_f32x4, diff_high_f32x4, sum_f32x4);
    sum_f32x4 = vec_madd(diff_low_f32x4, diff_low_f32x4, sum_f32x4);
    if (n) goto nk_sqeuclidean_f16_powervsx_cycle;
    *result = nk_hsum_f32x4_powervsx_(sum_f32x4);
}

NK_PUBLIC void nk_euclidean_f16_powervsx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_f16_powervsx(a, b, n, result);
    *result = nk_f32_sqrt_powervsx(*result);
}

NK_PUBLIC void nk_angular_f16_powervsx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    // f16 → f32 via POWER9 hardware XVCVHPSP
    nk_vf32x4_t ab_f32x4 = vec_splats(0.0f);
    nk_vf32x4_t a2_f32x4 = vec_splats(0.0f);
    nk_vf32x4_t b2_f32x4 = vec_splats(0.0f);
    nk_vu16x8_t a_u16x8, b_u16x8;
    nk_size_t tail_bytes;

nk_angular_f16_powervsx_cycle:
    if (n < 8) {
        tail_bytes = n * sizeof(nk_f16_t);
        a_u16x8 = vec_xl_len((nk_u16_t *)a, tail_bytes);
        b_u16x8 = vec_xl_len((nk_u16_t *)b, tail_bytes);
        n = 0;
    }
    else {
        a_u16x8 = vec_xl(0, (nk_u16_t const *)a);
        b_u16x8 = vec_xl(0, (nk_u16_t const *)b);
        a += 8, b += 8, n -= 8;
    }
    nk_vf32x4_t a_high_f32x4 = vec_extract_fp32_from_shorth(a_u16x8);
    nk_vf32x4_t a_low_f32x4 = vec_extract_fp32_from_shortl(a_u16x8);
    nk_vf32x4_t b_high_f32x4 = vec_extract_fp32_from_shorth(b_u16x8);
    nk_vf32x4_t b_low_f32x4 = vec_extract_fp32_from_shortl(b_u16x8);
    ab_f32x4 = vec_madd(a_high_f32x4, b_high_f32x4, ab_f32x4);
    ab_f32x4 = vec_madd(a_low_f32x4, b_low_f32x4, ab_f32x4);
    a2_f32x4 = vec_madd(a_high_f32x4, a_high_f32x4, a2_f32x4);
    a2_f32x4 = vec_madd(a_low_f32x4, a_low_f32x4, a2_f32x4);
    b2_f32x4 = vec_madd(b_high_f32x4, b_high_f32x4, b2_f32x4);
    b2_f32x4 = vec_madd(b_low_f32x4, b_low_f32x4, b2_f32x4);
    if (n) goto nk_angular_f16_powervsx_cycle;
    nk_f32_t ab = nk_hsum_f32x4_powervsx_(ab_f32x4);
    nk_f32_t a2 = nk_hsum_f32x4_powervsx_(a2_f32x4);
    nk_f32_t b2 = nk_hsum_f32x4_powervsx_(b2_f32x4);
    *result = nk_angular_normalize_f32_powervsx_(ab, a2, b2);
}

#pragma endregion F16 and BF16 Floats
#pragma region I8 and U8 Integers

NK_PUBLIC void nk_sqeuclidean_i8_powervsx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result) {
    // Power has no vabdq_s8. Widen i8 → i16 via vec_unpackh/vec_unpackl,
    // subtract in i16, then vec_msum(diff_i16, diff_i16, accumulator_i32) to square-accumulate.
    nk_vi32x4_t accumulator_i32x4 = vec_splats((nk_i32_t)0);
    nk_vi8x16_t a_i8x16, b_i8x16;
    nk_size_t tail_bytes;

nk_sqeuclidean_i8_powervsx_cycle:
    if (n < 16) {
        tail_bytes = n * sizeof(nk_i8_t);
        a_i8x16 = vec_xl_len((nk_i8_t *)a, tail_bytes);
        b_i8x16 = vec_xl_len((nk_i8_t *)b, tail_bytes);
        n = 0;
    }
    else {
        a_i8x16 = vec_xl(0, a);
        b_i8x16 = vec_xl(0, b);
        a += 16, b += 16, n -= 16;
    }
    // Widen high 8 bytes: i8 → i16
    nk_vi16x8_t a_high_i16x8 = vec_unpackh(a_i8x16);
    nk_vi16x8_t b_high_i16x8 = vec_unpackh(b_i8x16);
    nk_vi16x8_t diff_high_i16x8 = vec_sub(a_high_i16x8, b_high_i16x8);
    // vec_msum: multiply 8 i16 pairs and accumulate into 4 i32 lanes
    accumulator_i32x4 = vec_msum(diff_high_i16x8, diff_high_i16x8, accumulator_i32x4);
    // Widen low 8 bytes: i8 → i16
    nk_vi16x8_t a_low_i16x8 = vec_unpackl(a_i8x16);
    nk_vi16x8_t b_low_i16x8 = vec_unpackl(b_i8x16);
    nk_vi16x8_t diff_low_i16x8 = vec_sub(a_low_i16x8, b_low_i16x8);
    accumulator_i32x4 = vec_msum(diff_low_i16x8, diff_low_i16x8, accumulator_i32x4);
    if (n) goto nk_sqeuclidean_i8_powervsx_cycle;

    *result = (nk_u32_t)nk_hsum_i32x4_powervsx_(accumulator_i32x4);
}

NK_PUBLIC void nk_euclidean_i8_powervsx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq_u32;
    nk_sqeuclidean_i8_powervsx(a, b, n, &distance_sq_u32);
    *result = nk_f32_sqrt_powervsx((nk_f32_t)distance_sq_u32);
}

NK_PUBLIC void nk_angular_i8_powervsx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
    // Hybrid approach for 3-accumulator i8 angular distance:
    //   a·b: algebraic transform — VMSUMMBM(a, b⊕0x80) with correction −128·Σa
    //   a·a: abs-based unsigned   — VMSUMUBM(|a|, |a|), no correction needed
    //   b·b: abs-based unsigned   — VMSUMUBM(|b|, |b|), no correction needed
    // abs(-128)→-128 in i8 → 128 as u8 → 128²=16384=(-128)². Safe for all values.
    // 3 independent MSUM chains → excellent ILP on POWER9's dual-issue p01.
    nk_vu8x16_t const bias_u8x16 = vec_splats((nk_u8_t)0x80);
    nk_vi8x16_t const zeros_i8x16 = vec_splats((nk_i8_t)0);
    nk_vi32x4_t dot_product_i32x4 = vec_splats((nk_i32_t)0);
    nk_vu32x4_t a_norm_sq_u32x4 = vec_splats((nk_u32_t)0);
    nk_vu32x4_t b_norm_sq_u32x4 = vec_splats((nk_u32_t)0);
    nk_vu32x4_t sum_a_biased_u32x4 = vec_splats((nk_u32_t)0);
    nk_size_t count_padded = ((n + 15) / 16) * 16;
    nk_vi8x16_t a_i8x16, b_i8x16;
    nk_size_t tail_bytes;

nk_angular_i8_powervsx_cycle:
    if (n < 16) {
        tail_bytes = n * sizeof(nk_i8_t);
        a_i8x16 = vec_xl_len((nk_i8_t *)a, tail_bytes);
        b_i8x16 = vec_xl_len((nk_i8_t *)b, tail_bytes);
        n = 0;
    }
    else {
        a_i8x16 = vec_xl(0, a);
        b_i8x16 = vec_xl(0, b);
        a += 16, b += 16, n -= 16;
    }

    // Dot product: algebraic via VMSUMMBM(i8 × u8 → i32)
    nk_vu8x16_t b_biased_u8x16 = vec_xor((nk_vu8x16_t)b_i8x16, bias_u8x16);
    dot_product_i32x4 = vec_msum(a_i8x16, b_biased_u8x16, dot_product_i32x4);
    // Correction sum: Σ(a+128) via VSUM4UBS
    sum_a_biased_u32x4 = vec_sum4s(vec_xor((nk_vu8x16_t)a_i8x16, bias_u8x16), sum_a_biased_u32x4);
    // Norms: |a|² and |b|² via VMSUMUBM(u8 × u8 → u32) on absolute values
    nk_vu8x16_t a_abs_u8x16 = (nk_vu8x16_t)vec_max(a_i8x16, vec_sub(zeros_i8x16, a_i8x16));
    nk_vu8x16_t b_abs_u8x16 = (nk_vu8x16_t)vec_max(b_i8x16, vec_sub(zeros_i8x16, b_i8x16));
    a_norm_sq_u32x4 = vec_msum(a_abs_u8x16, a_abs_u8x16, a_norm_sq_u32x4);
    b_norm_sq_u32x4 = vec_msum(b_abs_u8x16, b_abs_u8x16, b_norm_sq_u32x4);

    if (n) goto nk_angular_i8_powervsx_cycle;

    // Correct the biased dot product: a·b = biased − 128·Σa = biased − 128·(Σ(a+128) − 128·count_padded)
    nk_i64_t correction = 128LL * (nk_i64_t)nk_hsum_u32x4_powervsx_(sum_a_biased_u32x4) -
                          16384LL * (nk_i64_t)count_padded;
    nk_i32_t dot_product_i32 = (nk_i32_t)((nk_i64_t)nk_hsum_i32x4_powervsx_(dot_product_i32x4) - correction);
    nk_u32_t a_norm_sq_u32 = nk_hsum_u32x4_powervsx_(a_norm_sq_u32x4);
    nk_u32_t b_norm_sq_u32 = nk_hsum_u32x4_powervsx_(b_norm_sq_u32x4);
    *result = nk_angular_normalize_f32_powervsx_((nk_f32_t)dot_product_i32, (nk_f32_t)a_norm_sq_u32,
                                                 (nk_f32_t)b_norm_sq_u32);
}

NK_PUBLIC void nk_sqeuclidean_u8_powervsx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    // Compute |a-b| without underflow: vec_sub(vec_max(a, b), vec_min(a, b))
    // Then square-accumulate via vec_msum(u8, u8, u32) → VMSUMUBM
    nk_vu32x4_t accumulator_u32x4 = vec_splats((nk_u32_t)0);
    nk_vu8x16_t a_u8x16, b_u8x16;
    nk_size_t tail_bytes;

nk_sqeuclidean_u8_powervsx_cycle:
    if (n < 16) {
        tail_bytes = n * sizeof(nk_u8_t);
        a_u8x16 = vec_xl_len((nk_u8_t *)a, tail_bytes);
        b_u8x16 = vec_xl_len((nk_u8_t *)b, tail_bytes);
        n = 0;
    }
    else {
        a_u8x16 = vec_xl(0, a);
        b_u8x16 = vec_xl(0, b);
        a += 16, b += 16, n -= 16;
    }
    nk_vu8x16_t diff_u8x16 = vec_sub(vec_max(a_u8x16, b_u8x16), vec_min(a_u8x16, b_u8x16));
    // VMSUMUBM: u8 × u8 → u32 accumulate
    accumulator_u32x4 = vec_msum(diff_u8x16, diff_u8x16, accumulator_u32x4);
    if (n) goto nk_sqeuclidean_u8_powervsx_cycle;

    *result = nk_hsum_u32x4_powervsx_(accumulator_u32x4);
}

NK_PUBLIC void nk_euclidean_u8_powervsx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq_u32;
    nk_sqeuclidean_u8_powervsx(a, b, n, &distance_sq_u32);
    *result = nk_f32_sqrt_powervsx((nk_f32_t)distance_sq_u32);
}

NK_PUBLIC void nk_angular_u8_powervsx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    // Triple accumulator in u32 using vec_msum(u8, u8, u32) → VMSUMUBM
    nk_vu32x4_t ab_u32x4 = vec_splats((nk_u32_t)0);
    nk_vu32x4_t aa_u32x4 = vec_splats((nk_u32_t)0);
    nk_vu32x4_t bb_u32x4 = vec_splats((nk_u32_t)0);
    nk_vu8x16_t a_u8x16, b_u8x16;
    nk_size_t tail_bytes;

nk_angular_u8_powervsx_cycle:
    if (n < 16) {
        tail_bytes = n * sizeof(nk_u8_t);
        a_u8x16 = vec_xl_len((nk_u8_t *)a, tail_bytes);
        b_u8x16 = vec_xl_len((nk_u8_t *)b, tail_bytes);
        n = 0;
    }
    else {
        a_u8x16 = vec_xl(0, a);
        b_u8x16 = vec_xl(0, b);
        a += 16, b += 16, n -= 16;
    }
    // VMSUMUBM: u8 × u8 → u32 accumulate
    ab_u32x4 = vec_msum(a_u8x16, b_u8x16, ab_u32x4);
    aa_u32x4 = vec_msum(a_u8x16, a_u8x16, aa_u32x4);
    bb_u32x4 = vec_msum(b_u8x16, b_u8x16, bb_u32x4);
    if (n) goto nk_angular_u8_powervsx_cycle;

    nk_u32_t ab = nk_hsum_u32x4_powervsx_(ab_u32x4);
    nk_u32_t aa = nk_hsum_u32x4_powervsx_(aa_u32x4);
    nk_u32_t bb = nk_hsum_u32x4_powervsx_(bb_u32x4);
    *result = nk_angular_normalize_f32_powervsx_((nk_f32_t)ab, (nk_f32_t)aa, (nk_f32_t)bb);
}

/** @brief Angular from_dot: computes 1 − dot × rsqrt(query_sumsq × target_sumsq) for 4 pairs in f64. */
NK_INTERNAL void nk_angular_through_f64_from_dot_powervsx_(nk_b256_vec_t const *dots_vec, nk_f64_t query_sumsq,
                                                           nk_b256_vec_t const *target_sumsqs_vec,
                                                           nk_b256_vec_t *result_vec) {
    nk_vf64x2_t dots_ab_f64x2 = dots_vec->vf64x2s[0];
    nk_vf64x2_t dots_cd_f64x2 = dots_vec->vf64x2s[1];
    nk_vf64x2_t query_f64x2 = vec_splats(query_sumsq);
    nk_vf64x2_t targets_ab_f64x2 = target_sumsqs_vec->vf64x2s[0];
    nk_vf64x2_t targets_cd_f64x2 = target_sumsqs_vec->vf64x2s[1];

    nk_vf64x2_t products_ab_f64x2 = vec_mul(query_f64x2, targets_ab_f64x2);
    nk_vf64x2_t products_cd_f64x2 = vec_mul(query_f64x2, targets_cd_f64x2);

    nk_vf64x2_t rsqrt_ab_f64x2 = nk_rsqrt_f64x2_powervsx_(products_ab_f64x2);
    nk_vf64x2_t rsqrt_cd_f64x2 = nk_rsqrt_f64x2_powervsx_(products_cd_f64x2);

    nk_vf64x2_t ones_f64x2 = vec_splats(1.0);
    nk_vf64x2_t zeros_f64x2 = vec_splats(0.0);
    nk_vf64x2_t result_ab_f64x2 = vec_max(vec_sub(ones_f64x2, vec_mul(dots_ab_f64x2, rsqrt_ab_f64x2)), zeros_f64x2);
    nk_vf64x2_t result_cd_f64x2 = vec_max(vec_sub(ones_f64x2, vec_mul(dots_cd_f64x2, rsqrt_cd_f64x2)), zeros_f64x2);

    nk_vu64x2_t prodzero_ab_u64x2 = (nk_vu64x2_t)vec_cmpeq(products_ab_f64x2, zeros_f64x2);
    nk_vu64x2_t dotzero_ab_u64x2 = (nk_vu64x2_t)vec_cmpeq(dots_ab_f64x2, zeros_f64x2);
    result_ab_f64x2 = vec_sel(result_ab_f64x2, zeros_f64x2, vec_and(prodzero_ab_u64x2, dotzero_ab_u64x2));
    result_ab_f64x2 = vec_sel(result_ab_f64x2, ones_f64x2, vec_andc(prodzero_ab_u64x2, dotzero_ab_u64x2));

    nk_vu64x2_t prodzero_cd_u64x2 = (nk_vu64x2_t)vec_cmpeq(products_cd_f64x2, zeros_f64x2);
    nk_vu64x2_t dotzero_cd_u64x2 = (nk_vu64x2_t)vec_cmpeq(dots_cd_f64x2, zeros_f64x2);
    result_cd_f64x2 = vec_sel(result_cd_f64x2, zeros_f64x2, vec_and(prodzero_cd_u64x2, dotzero_cd_u64x2));
    result_cd_f64x2 = vec_sel(result_cd_f64x2, ones_f64x2, vec_andc(prodzero_cd_u64x2, dotzero_cd_u64x2));

    result_vec->vf64x2s[0] = result_ab_f64x2;
    result_vec->vf64x2s[1] = result_cd_f64x2;
}

/** @brief Euclidean from_dot: computes √(query_sumsq + target_sumsq − 2×dot) for 4 pairs in f64. */
NK_INTERNAL void nk_euclidean_through_f64_from_dot_powervsx_(nk_b256_vec_t const *dots_vec, nk_f64_t query_sumsq,
                                                             nk_b256_vec_t const *target_sumsqs_vec,
                                                             nk_b256_vec_t *result_vec) {
    nk_vf64x2_t query_f64x2 = vec_splats(query_sumsq);
    nk_vf64x2_t neg_two_f64x2 = vec_splats(-2.0);
    nk_vf64x2_t zeros_f64x2 = vec_splats(0.0);

    nk_vf64x2_t sum_sq_ab_f64x2 = vec_add(query_f64x2, target_sumsqs_vec->vf64x2s[0]);
    nk_vf64x2_t sum_sq_cd_f64x2 = vec_add(query_f64x2, target_sumsqs_vec->vf64x2s[1]);
    nk_vf64x2_t dist_sq_ab_f64x2 = vec_max(vec_madd(neg_two_f64x2, dots_vec->vf64x2s[0], sum_sq_ab_f64x2), zeros_f64x2);
    nk_vf64x2_t dist_sq_cd_f64x2 = vec_max(vec_madd(neg_two_f64x2, dots_vec->vf64x2s[1], sum_sq_cd_f64x2), zeros_f64x2);

    result_vec->vf64x2s[0] = vec_sqrt(dist_sq_ab_f64x2);
    result_vec->vf64x2s[1] = vec_sqrt(dist_sq_cd_f64x2);
}

/** @brief Angular from_dot: computes 1 − dot × rsqrt(query_sumsq × target_sumsq) for 4 pairs in f32. */
NK_INTERNAL void nk_angular_through_f32_from_dot_powervsx_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
                                                           nk_b128_vec_t const *target_sumsqs_vec,
                                                           nk_b128_vec_t *result_vec) {
    nk_vf32x4_t dots_f32x4 = dots_vec->vf32x4;
    nk_vf32x4_t query_f32x4 = vec_splats(query_sumsq);
    nk_vf32x4_t products_f32x4 = vec_mul(query_f32x4, target_sumsqs_vec->vf32x4);
    nk_vf32x4_t rsqrt_f32x4 = nk_rsqrt_f32x4_powervsx_(products_f32x4);
    nk_vf32x4_t normalized_f32x4 = vec_mul(dots_f32x4, rsqrt_f32x4);
    nk_vf32x4_t angular_f32x4 = vec_sub(vec_splats(1.0f), normalized_f32x4);
    nk_vf32x4_t result_f32x4 = vec_max(angular_f32x4, vec_splats(0.0f));
    result_vec->vf32x4 = result_f32x4;
}

/** @brief Euclidean from_dot: computes √(query_sumsq + target_sumsq − 2×dot) for 4 pairs in f32. */
NK_INTERNAL void nk_euclidean_through_f32_from_dot_powervsx_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
                                                             nk_b128_vec_t const *target_sumsqs_vec,
                                                             nk_b128_vec_t *result_vec) {
    nk_vf32x4_t dots_f32x4 = dots_vec->vf32x4;
    nk_vf32x4_t query_f32x4 = vec_splats(query_sumsq);
    nk_vf32x4_t sum_sq_f32x4 = vec_add(query_f32x4, target_sumsqs_vec->vf32x4);
    // dist_sq = sum_sq − 2 × dot
    nk_vf32x4_t dist_sq_f32x4 = vec_madd(vec_splats(-2.0f), dots_f32x4, sum_sq_f32x4);
    // Clamp and sqrt
    dist_sq_f32x4 = vec_max(dist_sq_f32x4, vec_splats(0.0f));
    nk_vf32x4_t dist_f32x4 = vec_sqrt(dist_sq_f32x4);
    result_vec->vf32x4 = dist_f32x4;
}

/** @brief Angular from_dot for i32 accumulators: cast to f32, rsqrt+NR, clamp. 4 pairs. */
NK_INTERNAL void nk_angular_through_i32_from_dot_powervsx_(nk_b128_vec_t const *dots_vec, nk_i32_t query_sumsq,
                                                           nk_b128_vec_t const *target_sumsqs_vec,
                                                           nk_b128_vec_t *result_vec) {
    nk_vi32x4_t dots_i32x4 = dots_vec->vi32x4;
    nk_vf32x4_t dots_f32x4 = vec_ctf(dots_i32x4, 0);
    nk_vf32x4_t query_f32x4 = vec_splats((nk_f32_t)query_sumsq);
    nk_vi32x4_t targets_i32x4 = target_sumsqs_vec->vi32x4;
    nk_vf32x4_t products_f32x4 = vec_mul(query_f32x4, vec_ctf(targets_i32x4, 0));
    nk_vf32x4_t rsqrt_f32x4 = nk_rsqrt_f32x4_powervsx_(products_f32x4);
    nk_vf32x4_t normalized_f32x4 = vec_mul(dots_f32x4, rsqrt_f32x4);
    nk_vf32x4_t angular_f32x4 = vec_sub(vec_splats(1.0f), normalized_f32x4);
    nk_vf32x4_t result_f32x4 = vec_max(angular_f32x4, vec_splats(0.0f));
    result_vec->vf32x4 = result_f32x4;
}

/** @brief Euclidean from_dot for i32 accumulators: cast to f32, then √(a² + b² − 2ab). 4 pairs. */
NK_INTERNAL void nk_euclidean_through_i32_from_dot_powervsx_(nk_b128_vec_t const *dots_vec, nk_i32_t query_sumsq,
                                                             nk_b128_vec_t const *target_sumsqs_vec,
                                                             nk_b128_vec_t *result_vec) {
    nk_vi32x4_t dots_i32x4 = dots_vec->vi32x4;
    nk_vf32x4_t dots_f32x4 = vec_ctf(dots_i32x4, 0);
    nk_vf32x4_t query_f32x4 = vec_splats((nk_f32_t)query_sumsq);
    nk_vi32x4_t targets_i32x4 = target_sumsqs_vec->vi32x4;
    nk_vf32x4_t sum_sq_f32x4 = vec_add(query_f32x4, vec_ctf(targets_i32x4, 0));
    nk_vf32x4_t dist_sq_f32x4 = vec_madd(vec_splats(-2.0f), dots_f32x4, sum_sq_f32x4);
    dist_sq_f32x4 = vec_max(dist_sq_f32x4, vec_splats(0.0f));
    nk_vf32x4_t dist_f32x4 = vec_sqrt(dist_sq_f32x4);
    result_vec->vf32x4 = dist_f32x4;
}

/** @brief Angular from_dot for u32 accumulators: cast to f32, rsqrt+NR, clamp. 4 pairs. */
NK_INTERNAL void nk_angular_through_u32_from_dot_powervsx_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                           nk_b128_vec_t const *target_sumsqs_vec,
                                                           nk_b128_vec_t *result_vec) {
    nk_vu32x4_t dots_u32x4 = dots_vec->vu32x4;
    nk_vf32x4_t dots_f32x4 = vec_ctf(dots_u32x4, 0);
    nk_vf32x4_t query_f32x4 = vec_splats((nk_f32_t)query_sumsq);
    nk_vu32x4_t targets_u32x4 = target_sumsqs_vec->vu32x4;
    nk_vf32x4_t products_f32x4 = vec_mul(query_f32x4, vec_ctf(targets_u32x4, 0));
    nk_vf32x4_t rsqrt_f32x4 = nk_rsqrt_f32x4_powervsx_(products_f32x4);
    nk_vf32x4_t normalized_f32x4 = vec_mul(dots_f32x4, rsqrt_f32x4);
    nk_vf32x4_t angular_f32x4 = vec_sub(vec_splats(1.0f), normalized_f32x4);
    nk_vf32x4_t result_f32x4 = vec_max(angular_f32x4, vec_splats(0.0f));
    result_vec->vf32x4 = result_f32x4;
}

/** @brief Euclidean from_dot for u32 accumulators: cast to f32, then √(a² + b² − 2ab). 4 pairs. */
NK_INTERNAL void nk_euclidean_through_u32_from_dot_powervsx_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                             nk_b128_vec_t const *target_sumsqs_vec,
                                                             nk_b128_vec_t *result_vec) {
    nk_vu32x4_t dots_u32x4 = dots_vec->vu32x4;
    nk_vf32x4_t dots_f32x4 = vec_ctf(dots_u32x4, 0);
    nk_vf32x4_t query_f32x4 = vec_splats((nk_f32_t)query_sumsq);
    nk_vu32x4_t targets_u32x4 = target_sumsqs_vec->vu32x4;
    nk_vf32x4_t sum_sq_f32x4 = vec_add(query_f32x4, vec_ctf(targets_u32x4, 0));
    nk_vf32x4_t dist_sq_f32x4 = vec_madd(vec_splats(-2.0f), dots_f32x4, sum_sq_f32x4);
    dist_sq_f32x4 = vec_max(dist_sq_f32x4, vec_splats(0.0f));
    nk_vf32x4_t dist_f32x4 = vec_sqrt(dist_sq_f32x4);
    result_vec->vf32x4 = dist_f32x4;
}

#pragma endregion I8 and U8 Integers

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_POWERVSX
#endif // NK_TARGET_POWER64_
#endif // NK_SPATIAL_POWERVSX_H
