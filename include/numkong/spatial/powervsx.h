/**
 *  @file include/numkong/spatial/powervsx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *  @brief SIMD-accelerated spatial similarity measures for Power VSX.
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_powervsx_instructions Key Power VSX Spatial Instructions
 *
 *  Power ISA 3.0 (POWER9+) VSX instructions for distance computations:
 *
 *  @verbatim
 *  Intrinsic                        Instruction           POWER9
 *  vec_madd(f32)                    XVMADDASP             5cy
 *  vec_mul(f32)                     XVMULSP               5cy
 *  vec_add(f32)                     XVADDSP               5cy
 *  vec_sub(f32)                     XVSUBSP               5cy
 *  vec_rsqrte(f32)                  XVRSQRTESP            5cy
 *  vec_sqrt(f32)                    XVSQRTSP              26cy
 *  vec_doublee                      XVCVSPDP              3cy  (f32 → f64 even elts)
 *  vec_xl_len                       LXVL                  5cy  (partial vector load)
 *  vec_extract_fp32_from_shorth     XVCVHPSP              5cy  (f16 → f32 high half)
 *  vec_extract_fp32_from_shortl     XVCVHPSP              5cy  (f16 → f32 low half)
 *  vec_msum(i8, u8, i32)            VMSUMMBM              5cy  (i8×u8 widening multiply-sum)
 *  vec_msum(u8, u8, u32)            VMSUMUBM              5cy  (u8×u8 widening multiply-sum)
 *  vec_unpackh(i8)                  VUPKHSB               2cy  (sign-extend high 8 i8 → i16x8)
 *  vec_unpackl(i8)                  VUPKLSB               2cy  (sign-extend low 8 i8 → i16x8)
 *  @endverbatim
 *
 *  For angular distance, @c vec_rsqrte provides ~12-bit precision. Two Newton-Raphson iterations
 *  achieve ~23-bit precision for f32, three iterations for f64.
 */
#ifndef NUMKONG_SPATIAL_POWERVSX_H
#define NUMKONG_SPATIAL_POWERVSX_H

#if NUMKONG_ARCH_PPC64_
#if NUMKONG_TARGET_POWERVSX

#include "numkong/types.h"
#include "numkong/dot/powervsx.h" // `nk_hsum_*_powervsx_`, includes cast/powervsx.h

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
 *  Uses @c vec_rsqrte (~12-bit initial estimate) followed by two Newton-Raphson iterations,
 *  achieving ~23-bit precision sufficient for f32, and 0 for an infinite input.
 *  NR step: rsqrt = rsqrt × (1.5 − 0.5 × x × rsqrt × rsqrt)
 */
NUMKONG_INLINE nk_vf32x4_t nk_rsqrt_f32x4_powervsx_(nk_vf32x4_t x) {
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
    // The Newton steps turn the estimate of 0 for an infinite input into ∞ × 0 = NaN
    return vec_sel(rsqrt_f32x4, vec_splats(0.0f), vec_cmpeq(x, vec_splats(NUMKONG_F32_INF)));
}

/**
 *  @brief Reciprocal square root of 2 doubles with Newton-Raphson refinement.
 *
 *  Uses @c vec_rsqrte (~12-bit estimate) followed by three Newton-Raphson
 *  iterations, achieving ~48-bit precision for f64, and 0 for an infinite input.
 */
NUMKONG_INLINE nk_vf64x2_t nk_rsqrt_f64x2_powervsx_(nk_vf64x2_t x) {
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
    return vec_sel(rsqrt_f64x2, vec_splats(0.0), vec_cmpeq(x, vec_splats(NUMKONG_F64_INF)));
}

NUMKONG_INLINE nk_f32_t nk_angular_normalize_f32_powervsx_(nk_f32_t ab, nk_f32_t a2, nk_f32_t b2) {
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

NUMKONG_INLINE nk_f64_t nk_angular_normalize_f64_powervsx_(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {
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

NUMKONG_INLINE void nk_squared_distance_f32_powervsx_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                      nk_f64_t *result) {
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

NUMKONG_API nk_status_t nk_sqeuclidean_f32_powervsx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                    void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f32_powervsx_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_f32_powervsx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                  void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f32_powervsx_(a, b, n, result);
    *result = vec_extract(vec_sqrt(vec_splats(*result)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_f32_powervsx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                                void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
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
    return nk_success_k;
}

NUMKONG_INLINE void nk_squared_distance_f64_powervsx_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                      nk_f64_t *result) {
    nk_vf64x2_t sum_f64x2 = vec_splats((nk_f64_t)0), compensation_f64x2 = vec_splats((nk_f64_t)0);
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
    nk_dot2_f64x2_powervsx_(&sum_f64x2, &compensation_f64x2, diff_f64x2, diff_f64x2);
    if (n) goto nk_sqeuclidean_f64_powervsx_cycle;

    *result = nk_dot_stable_sum_f64x2_powervsx_(sum_f64x2, compensation_f64x2);
}

NUMKONG_API nk_status_t nk_sqeuclidean_f64_powervsx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                    void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f64_powervsx_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_f64_powervsx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                  void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f64_powervsx_(a, b, n, result);
    *result = vec_extract(vec_sqrt(vec_splats(*result)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_f64_powervsx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                                void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_vf64x2_t ab_sum_f64x2 = vec_splats((nk_f64_t)0), ab_compensation_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf64x2_t a2_sum_f64x2 = vec_splats((nk_f64_t)0), a2_compensation_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf64x2_t b2_sum_f64x2 = vec_splats((nk_f64_t)0), b2_compensation_f64x2 = vec_splats((nk_f64_t)0);
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
    nk_dot2_f64x2_powervsx_(&ab_sum_f64x2, &ab_compensation_f64x2, a_f64x2, b_f64x2);
    nk_dot2_f64x2_powervsx_(&a2_sum_f64x2, &a2_compensation_f64x2, a_f64x2, a_f64x2);
    nk_dot2_f64x2_powervsx_(&b2_sum_f64x2, &b2_compensation_f64x2, b_f64x2, b_f64x2);
    if (n) goto nk_angular_f64_powervsx_cycle;

    *result = nk_angular_normalize_f64_powervsx_(
        nk_dot_stable_sum_f64x2_powervsx_(ab_sum_f64x2, ab_compensation_f64x2),
        nk_dot_stable_sum_f64x2_powervsx_(a2_sum_f64x2, a2_compensation_f64x2),
        nk_dot_stable_sum_f64x2_powervsx_(b2_sum_f64x2, b2_compensation_f64x2));
    return nk_success_k;
}

#pragma endregion F32 and F64 Floats
#pragma region F16 and BF16 Floats

NUMKONG_INLINE void nk_squared_distance_bf16_powervsx_(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                       nk_f32_t *result) {
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

NUMKONG_API nk_status_t nk_sqeuclidean_bf16_powervsx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                     nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_bf16_powervsx_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_bf16_powervsx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                   nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_bf16_powervsx_(a, b, n, result);
    *result = vec_extract(vec_sqrt(vec_splats(*result)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_bf16_powervsx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                                 void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
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
    return nk_success_k;
}

NUMKONG_INLINE void nk_squared_distance_f16_powervsx_(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                      nk_f32_t *result) {
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

NUMKONG_API nk_status_t nk_sqeuclidean_f16_powervsx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                    void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f16_powervsx_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_f16_powervsx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                  void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f16_powervsx_(a, b, n, result);
    *result = vec_extract(vec_sqrt(vec_splats(*result)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_f16_powervsx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                                void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
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
    return nk_success_k;
}

#pragma endregion F16 and BF16 Floats
#pragma region I8 and U8 Integers

NUMKONG_INLINE void nk_squared_distance_i8_powervsx_(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n,
                                                     nk_u32_t *result) {
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

NUMKONG_API nk_status_t nk_sqeuclidean_i8_powervsx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result,
                                                   void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_i8_powervsx_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_i8_powervsx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                                 void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u32_t distance_sq_u32;
    nk_squared_distance_i8_powervsx_(a, b, n, &distance_sq_u32);
    *result = vec_extract(vec_sqrt(vec_splats((nk_f32_t)distance_sq_u32)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_i8_powervsx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result,
                                               void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // Hybrid approach for 3-accumulator i8 angular distance:
    //   a · b: algebraic transform — VMSUMMBM(a, b⊕0x80) with correction −128 · Σa
    //   a · a: abs-based unsigned   — VMSUMUBM(|a|, |a|), no correction needed
    //   b · b: abs-based unsigned   — VMSUMUBM(|b|, |b|), no correction needed
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

    // Correct the biased dot product: a · b = biased − 128 · Σa, which expands to
    // biased − 128 · (Σ(a+128) − 128 · count_padded).
    nk_i64_t correction = 128LL * (nk_i64_t)nk_hsum_u32x4_powervsx_(sum_a_biased_u32x4) -
                          16384LL * (nk_i64_t)count_padded;
    nk_i32_t dot_product_i32 = (nk_i32_t)((nk_i64_t)nk_hsum_i32x4_powervsx_(dot_product_i32x4) - correction);
    nk_u32_t a_norm_sq_u32 = nk_hsum_u32x4_powervsx_(a_norm_sq_u32x4);
    nk_u32_t b_norm_sq_u32 = nk_hsum_u32x4_powervsx_(b_norm_sq_u32x4);
    *result = nk_angular_normalize_f32_powervsx_((nk_f32_t)dot_product_i32, (nk_f32_t)a_norm_sq_u32,
                                                 (nk_f32_t)b_norm_sq_u32);
    return nk_success_k;
}

NUMKONG_INLINE void nk_squared_distance_u8_powervsx_(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n,
                                                     nk_u32_t *result) {
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

NUMKONG_API nk_status_t nk_sqeuclidean_u8_powervsx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                   void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_u8_powervsx_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_u8_powervsx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                                 void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u32_t distance_sq_u32;
    nk_squared_distance_u8_powervsx_(a, b, n, &distance_sq_u32);
    *result = vec_extract(vec_sqrt(vec_splats((nk_f32_t)distance_sq_u32)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_u8_powervsx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result,
                                               void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
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
    return nk_success_k;
}

/** Angular from_dot: computes 1 − dot × rsqrt(q) × rsqrt(t) for 4 pairs in f64, where q is
 *  @p query_sumsq and t each target's sum of squares, with the rules of the serial variant.
 *  Separate reciprocal square roots avoid overflowing the product of two finite-but-large norms. */
NUMKONG_INLINE void nk_angular_through_f64_from_dot_powervsx_(nk_b256_vec_t const *dots_vec, nk_f64_t query_sumsq,
                                                              nk_b256_vec_t const *target_sumsqs_vec,
                                                              nk_b256_vec_t *result_vec) {
    nk_vf64x2_t const zeros_f64x2 = vec_splats(0.0), ones_f64x2 = vec_splats(1.0);
    nk_vf64x2_t const dots_ab_f64x2 = dots_vec->vf64x2s[0], dots_cd_f64x2 = dots_vec->vf64x2s[1];
    nk_vf64x2_t const query_f64x2 = vec_splats(query_sumsq);
    nk_vf64x2_t const targets_ab_f64x2 = target_sumsqs_vec->vf64x2s[0];
    nk_vf64x2_t const targets_cd_f64x2 = target_sumsqs_vec->vf64x2s[1];

    nk_vf64x2_t const query_rsqrt_f64x2 = nk_rsqrt_f64x2_powervsx_(query_f64x2);
    nk_vf64x2_t const rsqrt_ab_f64x2 = vec_mul(query_rsqrt_f64x2, nk_rsqrt_f64x2_powervsx_(targets_ab_f64x2));
    nk_vf64x2_t const rsqrt_cd_f64x2 = vec_mul(query_rsqrt_f64x2, nk_rsqrt_f64x2_powervsx_(targets_cd_f64x2));
    nk_vf64x2_t angular_ab_f64x2 = vec_max(vec_sub(ones_f64x2, vec_mul(dots_ab_f64x2, rsqrt_ab_f64x2)), zeros_f64x2);
    nk_vf64x2_t angular_cd_f64x2 = vec_max(vec_sub(ones_f64x2, vec_mul(dots_cd_f64x2, rsqrt_cd_f64x2)), zeros_f64x2);

    // A zero norm or an exactly zero dot gives 1, and two zero norms give 0
    nk_vu64x2_t const query_zero_u64x2 = (nk_vu64x2_t)vec_cmpeq(query_f64x2, zeros_f64x2);
    nk_vu64x2_t const unit_ab_u64x2 = vec_or(
        (nk_vu64x2_t)vec_cmpeq(dots_ab_f64x2, zeros_f64x2),
        vec_or(query_zero_u64x2, (nk_vu64x2_t)vec_cmpeq(targets_ab_f64x2, zeros_f64x2)));
    nk_vu64x2_t const unit_cd_u64x2 = vec_or(
        (nk_vu64x2_t)vec_cmpeq(dots_cd_f64x2, zeros_f64x2),
        vec_or(query_zero_u64x2, (nk_vu64x2_t)vec_cmpeq(targets_cd_f64x2, zeros_f64x2)));
    angular_ab_f64x2 = vec_sel(angular_ab_f64x2, ones_f64x2, unit_ab_u64x2);
    angular_cd_f64x2 = vec_sel(angular_cd_f64x2, ones_f64x2, unit_cd_u64x2);
    angular_ab_f64x2 = vec_sel(angular_ab_f64x2, zeros_f64x2,
                               (nk_vu64x2_t)vec_cmpeq(vec_add(query_f64x2, targets_ab_f64x2), zeros_f64x2));
    angular_cd_f64x2 = vec_sel(angular_cd_f64x2, zeros_f64x2,
                               (nk_vu64x2_t)vec_cmpeq(vec_add(query_f64x2, targets_cd_f64x2), zeros_f64x2));

    // A NaN dot outranks the zero-norm cases
    result_vec->vf64x2s[0] = vec_sel(dots_ab_f64x2, angular_ab_f64x2,
                                     (nk_vu64x2_t)vec_cmpeq(dots_ab_f64x2, dots_ab_f64x2));
    result_vec->vf64x2s[1] = vec_sel(dots_cd_f64x2, angular_cd_f64x2,
                                     (nk_vu64x2_t)vec_cmpeq(dots_cd_f64x2, dots_cd_f64x2));
}

/** Euclidean from_dot: computes √(q + t − 2 × dot) for 4 pairs in f64, where q is @p query_sumsq
 *  and t each target's sum of squares. */
NUMKONG_INLINE void nk_euclidean_through_f64_from_dot_powervsx_(nk_b256_vec_t const *dots_vec, nk_f64_t query_sumsq,
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

/** Angular from_dot: computes 1 − dot × rsqrt(q) × rsqrt(t) for 4 pairs in f32, where q is
 *  @p query_sumsq and t each target's sum of squares, with the rules of the serial variant.
 *  Separate reciprocal square roots avoid overflowing the product of two finite-but-large norms. */
NUMKONG_INLINE void nk_angular_through_f32_from_dot_powervsx_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
                                                              nk_b128_vec_t const *target_sumsqs_vec,
                                                              nk_b128_vec_t *result_vec) {
    nk_vf32x4_t const zeros_f32x4 = vec_splats(0.0f), ones_f32x4 = vec_splats(1.0f), dots_f32x4 = dots_vec->vf32x4;
    nk_vf32x4_t const query_f32x4 = vec_splats(query_sumsq), targets_f32x4 = target_sumsqs_vec->vf32x4;
    nk_vf32x4_t const rsqrt_f32x4 = vec_mul(nk_rsqrt_f32x4_powervsx_(query_f32x4),
                                            nk_rsqrt_f32x4_powervsx_(targets_f32x4));
    nk_vf32x4_t angular_f32x4 = vec_max(vec_sub(ones_f32x4, vec_mul(dots_f32x4, rsqrt_f32x4)), zeros_f32x4);
    nk_vu32x4_t const unit_u32x4 = vec_or(
        (nk_vu32x4_t)vec_cmpeq(dots_f32x4, zeros_f32x4),
        vec_or((nk_vu32x4_t)vec_cmpeq(query_f32x4, zeros_f32x4), (nk_vu32x4_t)vec_cmpeq(targets_f32x4, zeros_f32x4)));
    angular_f32x4 = vec_sel(angular_f32x4, ones_f32x4, unit_u32x4);
    angular_f32x4 = vec_sel(angular_f32x4, zeros_f32x4,
                            (nk_vu32x4_t)vec_cmpeq(vec_add(query_f32x4, targets_f32x4), zeros_f32x4));
    // A NaN dot outranks the zero-norm cases
    result_vec->vf32x4 = vec_sel(dots_f32x4, angular_f32x4, (nk_vu32x4_t)vec_cmpeq(dots_f32x4, dots_f32x4));
}

/** Euclidean from_dot: computes √(q + t − 2 × dot) for 4 pairs in f32, where q is @p query_sumsq
 *  and t each target's sum of squares. */
NUMKONG_INLINE void nk_euclidean_through_f32_from_dot_powervsx_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
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

/** Angular from_dot for 4 i32 dots and u32 norms. @c vec_mule/vec_mulo make ab and d² exact in
 *  64 bits, and @c vec_floato rounds each gap ab − d² once, so FMA keeps the rest in f32. */
NUMKONG_INLINE void nk_angular_through_i32_from_dot_powervsx_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                              nk_b128_vec_t const *target_sumsqs_vec,
                                                              nk_b128_vec_t *result_vec) {
    nk_vf32x4_t const zeros_f32x4 = vec_splats(0.0f), ones_f32x4 = vec_splats(1.0f);
    nk_vi32x4_t const dots_i32x4 = dots_vec->vi32x4;
    nk_vu32x4_t const targets_u32x4 = target_sumsqs_vec->vu32x4, query_u32x4 = vec_splats(query_sumsq);
    nk_vu64x2_t const product_even_u64x2 = vec_mule(targets_u32x4, query_u32x4);
    nk_vu64x2_t const product_odd_u64x2 = vec_mulo(targets_u32x4, query_u32x4);
    // Cauchy–Schwarz keeps the gap non-negative
    nk_vu64x2_t const gap_even_u64x2 = vec_sub(product_even_u64x2, (nk_vu64x2_t)vec_mule(dots_i32x4, dots_i32x4));
    nk_vu64x2_t const gap_odd_u64x2 = vec_sub(product_odd_u64x2, (nk_vu64x2_t)vec_mulo(dots_i32x4, dots_i32x4));
    nk_vf32x4_t const gap_f32x4 = vec_mergeo(vec_floato(gap_even_u64x2), vec_floato(gap_odd_u64x2));
    nk_vf32x4_t const product_f32x4 = vec_mergeo(vec_floato(product_even_u64x2), vec_floato(product_odd_u64x2));
    nk_vf32x4_t const dots_f32x4 = vec_ctf(dots_i32x4, 0);
    nk_vf32x4_t const norm_f32x4 = vec_sqrt(product_f32x4);
    // A positive dot takes (ab − d²) / (ab + d × s), any other 1 + |d| / s
    nk_vu32x4_t const positive_u32x4 = (nk_vu32x4_t)vec_cmpgt(dots_i32x4, vec_splats(0));
    nk_vf32x4_t const numerator_f32x4 = vec_sel(vec_neg(dots_f32x4), gap_f32x4, positive_u32x4);
    nk_vf32x4_t const denominator_f32x4 = vec_sel(norm_f32x4, vec_madd(dots_f32x4, norm_f32x4, product_f32x4),
                                                  positive_u32x4);
    nk_vf32x4_t angular_f32x4 = vec_add(vec_div(numerator_f32x4, denominator_f32x4),
                                        vec_sel(ones_f32x4, zeros_f32x4, positive_u32x4));
    // A zero norm gives 1, and two zero norms give 0
    nk_vu32x4_t const target_zero_u32x4 = (nk_vu32x4_t)vec_cmpeq(targets_u32x4, vec_splats(0u));
    nk_vu32x4_t const query_zero_u32x4 = (nk_vu32x4_t)vec_cmpeq(query_u32x4, vec_splats(0u));
    angular_f32x4 = vec_sel(angular_f32x4, ones_f32x4, vec_or(target_zero_u32x4, query_zero_u32x4));
    result_vec->vf32x4 = vec_sel(angular_f32x4, zeros_f32x4, vec_and(target_zero_u32x4, query_zero_u32x4));
}

/** Euclidean from_dot for 4 i32 dots and u32 norms. The exact q + t − 2d spans 34 bits, so
 *  it sums as 16-bit halves that f32 holds exactly, rounding once without 64-bit lanes. */
NUMKONG_INLINE void nk_euclidean_through_i32_from_dot_powervsx_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                                nk_b128_vec_t const *target_sumsqs_vec,
                                                                nk_b128_vec_t *result_vec) {
    nk_vu32x4_t const shift_u32x4 = vec_splats(16u), low_mask_u32x4 = vec_splats(0xFFFFu);
    nk_vi32x4_t const dots_i32x4 = dots_vec->vi32x4;
    nk_vu32x4_t const targets_u32x4 = target_sumsqs_vec->vu32x4;
    // The arithmetic shift keeps d = (d >> 16) × 2¹⁶ + (d & 0xFFFF) for a negative dot
    nk_vi32x4_t const dots_high_i32x4 = vec_sra(dots_i32x4, shift_u32x4);
    nk_vi32x4_t const dots_low_i32x4 = vec_and(dots_i32x4, (nk_vi32x4_t)low_mask_u32x4);
    nk_vi32x4_t const high_i32x4 = vec_sub(
        (nk_vi32x4_t)vec_add(vec_sr(targets_u32x4, shift_u32x4), vec_splats(query_sumsq >> 16)),
        vec_add(dots_high_i32x4, dots_high_i32x4));
    nk_vi32x4_t const low_i32x4 = vec_sub(
        (nk_vi32x4_t)vec_add(vec_and(targets_u32x4, low_mask_u32x4), vec_splats(query_sumsq & 0xFFFFu)),
        vec_add(dots_low_i32x4, dots_low_i32x4));
    nk_vf32x4_t const distance_sq_f32x4 = vec_madd(vec_ctf(high_i32x4, 0), vec_splats(65536.0f), vec_ctf(low_i32x4, 0));
    result_vec->vf32x4 = vec_sqrt(distance_sq_f32x4);
}

/** Angular from_dot for 4 u32 dots and norms. @c vec_mule/vec_mulo make ab and d² exact in
 *  64 bits, and @c vec_floato rounds each gap ab − d² once, so FMA keeps the rest in f32. */
NUMKONG_INLINE void nk_angular_through_u32_from_dot_powervsx_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                              nk_b128_vec_t const *target_sumsqs_vec,
                                                              nk_b128_vec_t *result_vec) {
    nk_vf32x4_t const zeros_f32x4 = vec_splats(0.0f), ones_f32x4 = vec_splats(1.0f);
    nk_vu32x4_t const zeros_u32x4 = vec_splats(0u), dots_u32x4 = dots_vec->vu32x4;
    nk_vu32x4_t const targets_u32x4 = target_sumsqs_vec->vu32x4, query_u32x4 = vec_splats(query_sumsq);
    nk_vu64x2_t const product_even_u64x2 = vec_mule(targets_u32x4, query_u32x4);
    nk_vu64x2_t const product_odd_u64x2 = vec_mulo(targets_u32x4, query_u32x4);
    // Cauchy–Schwarz keeps the gap non-negative
    nk_vu64x2_t const gap_even_u64x2 = vec_sub(product_even_u64x2, vec_mule(dots_u32x4, dots_u32x4));
    nk_vu64x2_t const gap_odd_u64x2 = vec_sub(product_odd_u64x2, vec_mulo(dots_u32x4, dots_u32x4));
    nk_vf32x4_t const gap_f32x4 = vec_mergeo(vec_floato(gap_even_u64x2), vec_floato(gap_odd_u64x2));
    nk_vf32x4_t const product_f32x4 = vec_mergeo(vec_floato(product_even_u64x2), vec_floato(product_odd_u64x2));
    nk_vf32x4_t const norm_f32x4 = vec_sqrt(product_f32x4);
    // (ab − d²) / (ab + d × s) keeps the small angles that 1 − d / s cancels
    nk_vf32x4_t angular_f32x4 = vec_div(gap_f32x4, vec_madd(vec_ctf(dots_u32x4, 0), norm_f32x4, product_f32x4));
    // A zero norm or a zero dot gives 1, and two zero norms give 0
    nk_vu32x4_t const target_zero_u32x4 = (nk_vu32x4_t)vec_cmpeq(targets_u32x4, zeros_u32x4);
    nk_vu32x4_t const query_zero_u32x4 = (nk_vu32x4_t)vec_cmpeq(query_u32x4, zeros_u32x4);
    nk_vu32x4_t const unit_u32x4 = vec_or((nk_vu32x4_t)vec_cmpeq(dots_u32x4, zeros_u32x4),
                                          vec_or(target_zero_u32x4, query_zero_u32x4));
    angular_f32x4 = vec_sel(angular_f32x4, ones_f32x4, unit_u32x4);
    result_vec->vf32x4 = vec_sel(angular_f32x4, zeros_f32x4, vec_and(target_zero_u32x4, query_zero_u32x4));
}

/** Euclidean from_dot for 4 u32 dots and norms. The exact q + t − 2d spans 33 bits, so it sums
 *  as 16-bit halves that f32 holds exactly, rounding once without 64-bit lanes. */
NUMKONG_INLINE void nk_euclidean_through_u32_from_dot_powervsx_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                                nk_b128_vec_t const *target_sumsqs_vec,
                                                                nk_b128_vec_t *result_vec) {
    nk_vu32x4_t const shift_u32x4 = vec_splats(16u), low_mask_u32x4 = vec_splats(0xFFFFu);
    nk_vu32x4_t const dots_u32x4 = dots_vec->vu32x4, targets_u32x4 = target_sumsqs_vec->vu32x4;
    nk_vu32x4_t const dots_high_u32x4 = vec_sr(dots_u32x4, shift_u32x4);
    nk_vu32x4_t const dots_low_u32x4 = vec_and(dots_u32x4, low_mask_u32x4);
    nk_vi32x4_t const high_i32x4 = (nk_vi32x4_t)vec_sub(
        vec_add(vec_sr(targets_u32x4, shift_u32x4), vec_splats(query_sumsq >> 16)),
        vec_add(dots_high_u32x4, dots_high_u32x4));
    nk_vi32x4_t const low_i32x4 = (nk_vi32x4_t)vec_sub(
        vec_add(vec_and(targets_u32x4, low_mask_u32x4), vec_splats(query_sumsq & 0xFFFFu)),
        vec_add(dots_low_u32x4, dots_low_u32x4));
    nk_vf32x4_t const distance_sq_f32x4 = vec_madd(vec_ctf(high_i32x4, 0), vec_splats(65536.0f), vec_ctf(low_i32x4, 0));
    result_vec->vf32x4 = vec_sqrt(distance_sq_f32x4);
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

#endif // NUMKONG_TARGET_POWERVSX
#endif // NUMKONG_ARCH_PPC64_
#endif // NUMKONG_SPATIAL_POWERVSX_H
