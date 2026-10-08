/**
 *  @file include/numkong/spatial/sve.h
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief SIMD-accelerated spatial similarity measures for SVE.
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_sve_instructions ARM SVE Instructions
 *
 *  @verbatim
 *  Intrinsic      Instruction                V1
 *  svld1_f32      LD1W (Z.S, P/Z, [Xn])      4-6cy @ 2p
 *  svsub_f32_x    FSUB (Z.S, P/M, Z.S, Z.S)  3cy @ 2p
 *  svmla_f32_x    FMLA (Z.S, P/M, Z.S, Z.S)  4cy @ 2p
 *  svaddv_f32     FADDV (S, P, Z.S)          6cy @ 1p
 *  svdupq_n_f32   DUP (Z.S, #imm)            1cy @ 2p
 *  svwhilelt_b32  WHILELT (P.S, Xn, Xm)      2cy @ 1p
 *  svptrue_b32    PTRUE (P.S, pattern)       1cy @ 2p
 *  svcntw         CNTW (Xd)                  1cy @ 2p
 *  svld1_f64      LD1D (Z.D, P/Z, [Xn])      4-6cy @ 2p
 *  svsub_f64_x    FSUB (Z.D, P/M, Z.D, Z.D)  3cy @ 2p
 *  svmla_f64_x    FMLA (Z.D, P/M, Z.D, Z.D)  4cy @ 2p
 *  svaddv_f64     FADDV (D, P, Z.D)          6cy @ 1p
 *  @endverbatim
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5 and
 *  Apple M4+ use 128-bit. Code using svcntb() adapts automatically, but wider vectors process more
 *  elements per iteration with identical latencies.
 *
 *  Spatial operations like L2 distance and angular similarity benefit from SVE's fused multiply-add
 *  instructions. The FADDV reduction dominates the critical path for short vectors.
 */
#ifndef NUMKONG_SPATIAL_SVE_H
#define NUMKONG_SPATIAL_SVE_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_SVE_

#include "numkong/types.h"
#include "numkong/reduce/sve.h"   // `nk_svaddv_f64_`
#include "numkong/spatial/neon.h" // `nk_angular_normalize_f64_neon_`
#include "numkong/dot/sve.h"      // `nk_dot_stable_sum_f64_sve_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

/** Sums the squared differences of @p n F32 pairs, widened to F64. */
NUMKONG_INLINE void nk_squared_distance_f32_sve_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_size_t i = 0;
    svfloat64_t dist_sq_f64x = svdupq_n_f64(0.0, 0.0);
    for (; i < n; i += svcntw()) {
        svbool_t predicate_b32x = svwhilelt_b32_u64(i, n);
        svfloat32_t a_f32x = svld1_f32(predicate_b32x, a + i);
        svfloat32_t b_f32x = svld1_f32(predicate_b32x, b + i);
        nk_size_t remaining = n - i < svcntw() ? n - i : svcntw();

        // svcvt_f64_f32_x widens only even-indexed f32 elements; svext by 1 shifts odd into even.
        svbool_t pred_even_b64x = svwhilelt_b64_u64(0u, (remaining + 1) / 2);
        svfloat64_t a_even_f64x = svcvt_f64_f32_x(pred_even_b64x, a_f32x);
        svfloat64_t b_even_f64x = svcvt_f64_f32_x(pred_even_b64x, b_f32x);
        svfloat64_t diff_even_f64x = svsub_f64_x(pred_even_b64x, a_even_f64x, b_even_f64x);
        dist_sq_f64x = svmla_f64_m(pred_even_b64x, dist_sq_f64x, diff_even_f64x, diff_even_f64x);

        svbool_t pred_odd_b64x = svwhilelt_b64_u64(0u, remaining / 2);
        svfloat64_t a_odd_f64x = svcvt_f64_f32_x(pred_odd_b64x, svext_f32(a_f32x, a_f32x, 1));
        svfloat64_t b_odd_f64x = svcvt_f64_f32_x(pred_odd_b64x, svext_f32(b_f32x, b_f32x, 1));
        svfloat64_t diff_odd_f64x = svsub_f64_x(pred_odd_b64x, a_odd_f64x, b_odd_f64x);
        dist_sq_f64x = svmla_f64_m(pred_odd_b64x, dist_sq_f64x, diff_odd_f64x, diff_odd_f64x);
    }
    nk_f64_t dist_sq_f64 = nk_svaddv_f64_(svptrue_b64(), dist_sq_f64x);
    *result = dist_sq_f64;
}

#if NUMKONG_TARGET_SVE
NUMKONG_API nk_status_t nk_sqeuclidean_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f32_sve_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                             nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f32_sve_(a, b, n, result);
    *result = vget_lane_f64(vsqrt_f64(vdup_n_f64(*result)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t i = 0;
    svfloat64_t ab_f64x = svdupq_n_f64(0.0, 0.0);
    svfloat64_t a2_f64x = svdupq_n_f64(0.0, 0.0);
    svfloat64_t b2_f64x = svdupq_n_f64(0.0, 0.0);
    for (; i < n; i += svcntw()) {
        svbool_t predicate_b32x = svwhilelt_b32_u64(i, n);
        svfloat32_t a_f32x = svld1_f32(predicate_b32x, a + i);
        svfloat32_t b_f32x = svld1_f32(predicate_b32x, b + i);
        nk_size_t remaining = n - i < svcntw() ? n - i : svcntw();

        // svcvt_f64_f32_x widens only even-indexed f32 elements; svext by 1 shifts odd into even.
        svbool_t pred_even_b64x = svwhilelt_b64_u64(0u, (remaining + 1) / 2);
        svfloat64_t a_even_f64x = svcvt_f64_f32_x(pred_even_b64x, a_f32x);
        svfloat64_t b_even_f64x = svcvt_f64_f32_x(pred_even_b64x, b_f32x);
        ab_f64x = svmla_f64_m(pred_even_b64x, ab_f64x, a_even_f64x, b_even_f64x);
        a2_f64x = svmla_f64_m(pred_even_b64x, a2_f64x, a_even_f64x, a_even_f64x);
        b2_f64x = svmla_f64_m(pred_even_b64x, b2_f64x, b_even_f64x, b_even_f64x);

        svbool_t pred_odd_b64x = svwhilelt_b64_u64(0u, remaining / 2);
        svfloat64_t a_odd_f64x = svcvt_f64_f32_x(pred_odd_b64x, svext_f32(a_f32x, a_f32x, 1));
        svfloat64_t b_odd_f64x = svcvt_f64_f32_x(pred_odd_b64x, svext_f32(b_f32x, b_f32x, 1));
        ab_f64x = svmla_f64_m(pred_odd_b64x, ab_f64x, a_odd_f64x, b_odd_f64x);
        a2_f64x = svmla_f64_m(pred_odd_b64x, a2_f64x, a_odd_f64x, a_odd_f64x);
        b2_f64x = svmla_f64_m(pred_odd_b64x, b2_f64x, b_odd_f64x, b_odd_f64x);
    }

    nk_f64_t ab_f64 = nk_svaddv_f64_(svptrue_b64(), ab_f64x);
    nk_f64_t a2_f64 = nk_svaddv_f64_(svptrue_b64(), a2_f64x);
    nk_f64_t b2_f64 = nk_svaddv_f64_(svptrue_b64(), b2_f64x);
    *result = nk_angular_normalize_f64_neon_(ab_f64, a2_f64, b2_f64);
    return nk_success_k;
}

/** Sums the squared differences of @p n F64 pairs with Neumaier compensation. */
NUMKONG_INLINE void nk_squared_distance_f64_sve_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    // Neumaier compensated summation for numerical stability
    nk_size_t i = 0;
    svfloat64_t sum_f64x = svdupq_n_f64(0.0, 0.0);
    svfloat64_t compensation_f64x = svdupq_n_f64(0.0, 0.0);
    svbool_t predicate_all_b64x = svptrue_b64();
    do {
        svbool_t predicate_b64x = svwhilelt_b64_u64(i, n);
        svfloat64_t a_f64x = svld1_f64(predicate_b64x, a + i);
        svfloat64_t b_f64x = svld1_f64(predicate_b64x, b + i);
        svfloat64_t diff_f64x = svsub_f64_x(predicate_b64x, a_f64x, b_f64x);
        svfloat64_t diff_sq_f64x = svmul_f64_x(predicate_b64x, diff_f64x, diff_f64x);
        // Neumaier: t = sum + x
        svfloat64_t t_f64x = svadd_f64_m(predicate_b64x, sum_f64x, diff_sq_f64x);
        svfloat64_t abs_sum_f64x = svabs_f64_x(predicate_b64x, sum_f64x);
        // diff_sq is already non-negative (it's a square), so svabs is unnecessary
        svbool_t sum_ge_x_b64x = svcmpge_f64(predicate_b64x, abs_sum_f64x, diff_sq_f64x);
        // When |sum| >= |x|: comp += (sum - t) + x; when |x| > |sum|: comp += (x - t) + sum
        svfloat64_t comp_sum_large_f64x = svadd_f64_x(predicate_b64x, svsub_f64_x(predicate_b64x, sum_f64x, t_f64x),
                                                      diff_sq_f64x);
        svfloat64_t comp_x_large_f64x = svadd_f64_x(predicate_b64x, svsub_f64_x(predicate_b64x, diff_sq_f64x, t_f64x),
                                                    sum_f64x);
        svfloat64_t comp_update_f64x = svsel_f64(sum_ge_x_b64x, comp_sum_large_f64x, comp_x_large_f64x);
        compensation_f64x = svadd_f64_m(predicate_b64x, compensation_f64x, comp_update_f64x);
        sum_f64x = t_f64x;
        i += svcntd();
    } while (i < n);
    *result = nk_dot_stable_sum_f64_sve_(predicate_all_b64x, sum_f64x, compensation_f64x);
}

NUMKONG_API nk_status_t nk_sqeuclidean_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f64_sve_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                             nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_squared_distance_f64_sve_(a, b, n, result);
    *result = vget_lane_f64(vsqrt_f64(vdup_n_f64(*result)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // Dot2 (Ogita-Rump-Oishi) for cross-product ab (may have cancellation),
    // simple FMA for self-products a2/b2 (all positive, no cancellation)
    nk_size_t i = 0;
    svfloat64_t ab_sum_f64x = svdupq_n_f64(0.0, 0.0);
    svfloat64_t ab_compensation_f64x = svdupq_n_f64(0.0, 0.0);
    svfloat64_t a2_f64x = svdupq_n_f64(0.0, 0.0);
    svfloat64_t b2_f64x = svdupq_n_f64(0.0, 0.0);
    svbool_t predicate_all_b64x = svptrue_b64();
    do {
        svbool_t predicate_b64x = svwhilelt_b64_u64(i, n);
        svfloat64_t a_f64x = svld1_f64(predicate_b64x, a + i);
        svfloat64_t b_f64x = svld1_f64(predicate_b64x, b + i);
        // TwoProd for ab: product = a*b, error = a*b - product in one rounding
        svfloat64_t product_f64x = svmul_f64_x(predicate_b64x, a_f64x, b_f64x);
        svfloat64_t product_error_f64x = svnmls_f64_x(predicate_b64x, product_f64x, a_f64x, b_f64x);
        // TwoSum: (tentative_sum, sum_error) = TwoSum(sum, product)
        svfloat64_t tentative_sum_f64x = svadd_f64_m(predicate_b64x, ab_sum_f64x, product_f64x);
        svfloat64_t virtual_addend_f64x = svsub_f64_x(predicate_b64x, tentative_sum_f64x, ab_sum_f64x);
        svfloat64_t sum_error_f64x = svadd_f64_x(
            predicate_b64x,
            svsub_f64_x(predicate_b64x, ab_sum_f64x,
                        svsub_f64_x(predicate_b64x, tentative_sum_f64x, virtual_addend_f64x)),
            svsub_f64_x(predicate_b64x, product_f64x, virtual_addend_f64x));
        ab_sum_f64x = tentative_sum_f64x;
        ab_compensation_f64x = svadd_f64_m(predicate_b64x, ab_compensation_f64x,
                                           svadd_f64_x(predicate_b64x, sum_error_f64x, product_error_f64x));
        // Simple FMA for self-products (no cancellation)
        a2_f64x = svmla_f64_m(predicate_b64x, a2_f64x, a_f64x, a_f64x);
        b2_f64x = svmla_f64_m(predicate_b64x, b2_f64x, b_f64x, b_f64x);
        i += svcntd();
    } while (i < n);

    nk_f64_t ab_f64 = nk_dot_stable_sum_f64_sve_(predicate_all_b64x, ab_sum_f64x, ab_compensation_f64x);
    nk_f64_t a2_f64 = nk_svaddv_f64_(predicate_all_b64x, a2_f64x);
    nk_f64_t b2_f64 = nk_svaddv_f64_(predicate_all_b64x, b2_f64x);
    *result = nk_angular_normalize_f64_neon_(ab_f64, a2_f64, b2_f64);
    return nk_success_k;
}
#endif // NUMKONG_TARGET_SVE

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ARM64_SVE_
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_SPATIAL_SVE_H
