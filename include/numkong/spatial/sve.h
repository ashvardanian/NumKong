/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for SVE.
 *  @file include/numkong/spatial/sve.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_sve_instructions ARM SVE Instructions
 *
 *      Intrinsic      Instruction                V1
 *      svld1_f32      LD1W (Z.S, P/Z, [Xn])      4-6cy @ 2p
 *      svsub_f32_x    FSUB (Z.S, P/M, Z.S, Z.S)  3cy @ 2p
 *      svmla_f32_x    FMLA (Z.S, P/M, Z.S, Z.S)  4cy @ 2p
 *      svaddv_f32     FADDV (S, P, Z.S)          6cy @ 1p
 *      svdupq_n_f32   DUP (Z.S, #imm)            1cy @ 2p
 *      svwhilelt_b32  WHILELT (P.S, Xn, Xm)      2cy @ 1p
 *      svptrue_b32    PTRUE (P.S, pattern)       1cy @ 2p
 *      svcntw         CNTW (Xd)                  1cy @ 2p
 *      svld1_f64      LD1D (Z.D, P/Z, [Xn])      4-6cy @ 2p
 *      svsub_f64_x    FSUB (Z.D, P/M, Z.D, Z.D)  3cy @ 2p
 *      svmla_f64_x    FMLA (Z.D, P/M, Z.D, Z.D)  4cy @ 2p
 *      svaddv_f64     FADDV (D, P, Z.D)          6cy @ 1p
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5
 *  and Apple M4+ use 128-bit. Code using svcntb() adapts automatically, but wider vectors
 *  process more elements per iteration with identical latencies.
 *
 *  Spatial operations like L2 distance and angular similarity benefit from SVE's fused
 *  multiply-add instructions. The FADDV reduction dominates the critical path for short vectors.
 */
#ifndef NK_SPATIAL_SVE_H
#define NK_SPATIAL_SVE_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SVE

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // `nk_f64_sqrt_neon`
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

/** @brief Reciprocal square root of an f32 SVE vector via estimate + 2 Newton-Raphson steps.
 *
 *  Computes 1/sqrt(x) for each active lane. The initial estimate from `svrsqrte_f32`
 *  has ~8 bits of precision; each Newton-Raphson iteration via `svrsqrts_f32` roughly
 *  doubles the mantissa bits, giving ~23 bits (~full f32 precision) after 2 steps.
 *
 *  Marked `__arm_streaming_compatible` so the helper is callable from both streaming
 *  (SME) and non-streaming (SVE) contexts without mode transitions.
 *
 *  @param predicate Active-lane mask
 *  @param x         Input vector (must be positive for meaningful results)
 *  @return          Approximate 1/sqrt(x) with ~23-bit mantissa accuracy
 */
NK_INTERNAL svfloat32_t nk_rsqrt_f32x_sve_(svbool_t predicate_b32x, svfloat32_t x) NK_STREAMING_COMPATIBLE_ {
    svfloat32_t r = svrsqrte_f32(x);
    r = svmul_f32_x(predicate_b32x, r, svrsqrts_f32(svmul_f32_x(predicate_b32x, x, r), r));
    r = svmul_f32_x(predicate_b32x, r, svrsqrts_f32(svmul_f32_x(predicate_b32x, x, r), r));
    return r;
}

/** @brief Reciprocal square root of an f64 SVE vector via estimate + 3 Newton-Raphson steps.
 *
 *  Computes 1/sqrt(x) for each active lane. The initial estimate from `svrsqrte_f64`
 *  has ~8 bits of precision; three Newton-Raphson iterations via `svrsqrts_f64` yield
 *  ~52-bit mantissa accuracy (full f64 precision).
 *
 *  Marked `__arm_streaming_compatible` so the helper is callable from both streaming
 *  (SME) and non-streaming (SVE) contexts without mode transitions.
 *
 *  @param predicate_b32x Active-lane mask
 *  @param x         Input vector (must be positive for meaningful results)
 *  @return          Approximate 1/sqrt(x) with ~52-bit mantissa accuracy
 */
NK_INTERNAL svfloat64_t nk_rsqrt_f64x_sve_(svbool_t predicate_b64x, svfloat64_t x) NK_STREAMING_COMPATIBLE_ {
    svfloat64_t r = svrsqrte_f64(x);
    r = svmul_f64_x(predicate_b64x, r, svrsqrts_f64(svmul_f64_x(predicate_b64x, x, r), r));
    r = svmul_f64_x(predicate_b64x, r, svrsqrts_f64(svmul_f64_x(predicate_b64x, x, r), r));
    r = svmul_f64_x(predicate_b64x, r, svrsqrts_f64(svmul_f64_x(predicate_b64x, x, r), r));
    return r;
}

NK_PUBLIC void nk_sqeuclidean_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
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
    nk_f64_t dist_sq_f64 = svaddv_f64(svptrue_b64(), dist_sq_f64x);
    NK_UNPOISON(&dist_sq_f64, sizeof(dist_sq_f64));
    *result = dist_sq_f64;
}

NK_PUBLIC void nk_euclidean_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_sqeuclidean_f32_sve(a, b, n, result);
    *result = nk_f64_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
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

    nk_f64_t ab_f64 = svaddv_f64(svptrue_b64(), ab_f64x);
    nk_f64_t a2_f64 = svaddv_f64(svptrue_b64(), a2_f64x);
    nk_f64_t b2_f64 = svaddv_f64(svptrue_b64(), b2_f64x);
    NK_UNPOISON(&ab_f64, sizeof(ab_f64));
    NK_UNPOISON(&a2_f64, sizeof(a2_f64));
    NK_UNPOISON(&b2_f64, sizeof(b2_f64));
    *result = nk_angular_normalize_f64_neon_(ab_f64, a2_f64, b2_f64);
}

NK_PUBLIC void nk_sqeuclidean_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
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

NK_PUBLIC void nk_euclidean_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_sqeuclidean_f64_sve(a, b, n, result);
    *result = nk_f64_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
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
        // TwoProd for ab: product = a*b, error = fma(a,b,-product) = -(product - a*b)
        svfloat64_t product_f64x = svmul_f64_x(predicate_b64x, a_f64x, b_f64x);
        svfloat64_t product_error_f64x = svneg_f64_x(predicate_b64x,
                                                     svnmls_f64_x(predicate_b64x, product_f64x, a_f64x, b_f64x));
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
    nk_f64_t a2_f64 = svaddv_f64(predicate_all_b64x, a2_f64x);
    nk_f64_t b2_f64 = svaddv_f64(predicate_all_b64x, b2_f64x);
    NK_UNPOISON(&a2_f64, sizeof(a2_f64));
    NK_UNPOISON(&b2_f64, sizeof(b2_f64));
    *result = nk_angular_normalize_f64_neon_(ab_f64, a2_f64, b2_f64);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SVE
#endif // NK_TARGET_ARM64_
#endif // NK_SPATIAL_SVE_H
