/**
 *  @brief SIMD-accelerated Dot Products for SVE.
 *  @file include/numkong/dot/sve.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_sve_instructions ARM SVE Instructions
 *
 *      Intrinsic      Instruction                V1
 *      svld1_f32      LD1W (Z.S, P/Z, [Xn])      4-6cy @ 2p
 *      svld2_f32      LD2W (Z.S, P/Z, [Xn])      6-8cy @ 1p
 *      svmla_f32_x    FMLA (Z.S, P/M, Z.S, Z.S)  4cy @ 2p
 *      svmls_f32_x    FMLS (Z.S, P/M, Z.S, Z.S)  4cy @ 2p
 *      svaddv_f32     FADDV (S, P, Z.S)          6cy @ 1p
 *      svdup_f32      DUP (Z.S, #imm)            1cy @ 2p
 *      svwhilelt_b32  WHILELT (P.S, Xn, Xm)      2cy @ 1p
 *      svptrue_b32    PTRUE (P.S, pattern)       1cy @ 2p
 *      svcntw         CNTW (Xd)                  1cy @ 2p
 *      svcntd         CNTD (Xd)                  1cy @ 2p
 *      svld1_f64      LD1D (Z.D, P/Z, [Xn])      4-6cy @ 2p
 *      svld2_f64      LD2D (Z.D, P/Z, [Xn])      6-8cy @ 1p
 *      svmla_f64_x    FMLA (Z.D, P/M, Z.D, Z.D)  4cy @ 2p
 *      svmls_f64_x    FMLS (Z.D, P/M, Z.D, Z.D)  4cy @ 2p
 *      svaddv_f64     FADDV (D, P, Z.D)          6cy @ 1p
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5
 *  and Apple M4+ use 128-bit. Code using svcntb() adapts automatically, but wider vectors
 *  process more elements per iteration with identical latencies.
 *
 *  The FADDV horizontal reduction has higher latency (6cy) compared to vertical operations,
 *  making it beneficial to accumulate in vector registers and reduce only at the end.
 */
#ifndef NK_DOT_SVE_H
#define NK_DOT_SVE_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SVE

#include "numkong/types.h"      // `nk_f32_t`
#include "numkong/dot/serial.h" // `nk_u1x8_popcount_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

/** @brief Compensated horizontal sum of SVE f64 lanes via TwoSum tree reduction.
 *
 *  Uses svtbl to extract the upper half at each tree level. Out-of-range indices
 *  return 0 (SVE spec), which is harmless since only the lower half is meaningful
 *  after each halving stage.
 */
NK_INTERNAL nk_f64_t nk_dot_stable_sum_f64_sve_(svbool_t predicate_b64x, svfloat64_t sum, svfloat64_t compensation) {
    // Stage 0: TwoSum merge of sum + compensation (parallel across all active lanes)
    svfloat64_t tentative_sum_f64x = svadd_f64_x(predicate_b64x, sum, compensation);
    svfloat64_t virtual_addend_f64x = svsub_f64_x(predicate_b64x, tentative_sum_f64x, sum);
    svfloat64_t accumulated_error_f64x = svadd_f64_x(
        predicate_b64x,
        svsub_f64_x(predicate_b64x, sum, svsub_f64_x(predicate_b64x, tentative_sum_f64x, virtual_addend_f64x)),
        svsub_f64_x(predicate_b64x, compensation, virtual_addend_f64x));

    // Tree reduction: TwoSum halving at each level, log2(VL) iterations
    for (unsigned int half = (unsigned int)svcntd() / 2; half > 0; half >>= 1) {
        svuint64_t upper_indices_u64x = svadd_n_u64_x(predicate_b64x, svindex_u64(0, 1), half);
        svfloat64_t upper_sum_f64x = svtbl_f64(tentative_sum_f64x, upper_indices_u64x);
        svfloat64_t upper_error_f64x = svtbl_f64(accumulated_error_f64x, upper_indices_u64x);
        // TwoSum: lower_half + upper_half
        svfloat64_t halved_tentative_sum_f64x = svadd_f64_x(predicate_b64x, tentative_sum_f64x, upper_sum_f64x);
        svfloat64_t halved_virtual_addend_f64x = svsub_f64_x(predicate_b64x, halved_tentative_sum_f64x,
                                                             tentative_sum_f64x);
        svfloat64_t rounding_error_f64x = svadd_f64_x(
            predicate_b64x,
            svsub_f64_x(predicate_b64x, tentative_sum_f64x,
                        svsub_f64_x(predicate_b64x, halved_tentative_sum_f64x, halved_virtual_addend_f64x)),
            svsub_f64_x(predicate_b64x, upper_sum_f64x, halved_virtual_addend_f64x));
        tentative_sum_f64x = halved_tentative_sum_f64x;
        accumulated_error_f64x = svadd_f64_x(
            predicate_b64x, svadd_f64_x(predicate_b64x, accumulated_error_f64x, upper_error_f64x), rounding_error_f64x);
    }
    // Result is in lane 0
    svbool_t predicate_first_b64x = svwhilelt_b64_u64(0u, 1);
    return svlastb_f64(predicate_first_b64x, tentative_sum_f64x) +
           svlastb_f64(predicate_first_b64x, accumulated_error_f64x);
}

NK_PUBLIC void nk_dot_f32_sve(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                              nk_f64_t *result) {
    nk_size_t idx_scalars = 0;
    svfloat64_t ab_f64x = svdup_f64(0.);
    for (; idx_scalars < count_scalars; idx_scalars += svcntw()) {
        svbool_t predicate_b32x = svwhilelt_b32_u64(idx_scalars, count_scalars);
        svfloat32_t a_f32x = svld1_f32(predicate_b32x, a_scalars + idx_scalars);
        svfloat32_t b_f32x = svld1_f32(predicate_b32x, b_scalars + idx_scalars);
        nk_size_t remaining = count_scalars - idx_scalars < svcntw() ? count_scalars - idx_scalars : svcntw();

        // svcvt_f64_f32_x widens only even-indexed f32 elements; svext by 1 shifts odd into even.
        svbool_t pred_even_b64x = svwhilelt_b64_u64(0u, (remaining + 1) / 2);
        ab_f64x = svmla_f64_m(pred_even_b64x, ab_f64x, svcvt_f64_f32_x(pred_even_b64x, a_f32x),
                              svcvt_f64_f32_x(pred_even_b64x, b_f32x));

        svbool_t pred_odd_b64x = svwhilelt_b64_u64(0u, remaining / 2);
        ab_f64x = svmla_f64_m(pred_odd_b64x, ab_f64x, svcvt_f64_f32_x(pred_odd_b64x, svext_f32(a_f32x, a_f32x, 1)),
                              svcvt_f64_f32_x(pred_odd_b64x, svext_f32(b_f32x, b_f32x, 1)));
    }
    *result = svaddv_f64(svptrue_b64(), ab_f64x);
    NK_UNPOISON(result, sizeof(*result));
}

NK_PUBLIC void nk_dot_f32c_sve(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                               nk_f64c_t *results) {
    nk_size_t idx_pairs = 0;
    svfloat64_t ab_real_f64x = svdup_f64(0.);
    svfloat64_t ab_imag_f64x = svdup_f64(0.);
    for (; idx_pairs < count_pairs; idx_pairs += svcntw()) {
        svbool_t predicate_b32x = svwhilelt_b32_u64(idx_pairs, count_pairs);
        svfloat32x2_t a_f32x2 = svld2_f32(predicate_b32x, (nk_f32_t const *)(a_pairs + idx_pairs));
        svfloat32x2_t b_f32x2 = svld2_f32(predicate_b32x, (nk_f32_t const *)(b_pairs + idx_pairs));
        svfloat32_t a_real_f32x = svget2_f32(a_f32x2, 0);
        svfloat32_t a_imag_f32x = svget2_f32(a_f32x2, 1);
        svfloat32_t b_real_f32x = svget2_f32(b_f32x2, 0);
        svfloat32_t b_imag_f32x = svget2_f32(b_f32x2, 1);
        nk_size_t remaining = count_pairs - idx_pairs < svcntw() ? count_pairs - idx_pairs : svcntw();

        // svcvt_f64_f32_x widens only even-indexed f32 elements; svext by 1 shifts odd into even.
        svbool_t pred_even_b64x = svwhilelt_b64_u64(0u, (remaining + 1) / 2);
        svfloat64_t a_real_even_f64x = svcvt_f64_f32_x(pred_even_b64x, a_real_f32x);
        svfloat64_t a_imag_even_f64x = svcvt_f64_f32_x(pred_even_b64x, a_imag_f32x);
        svfloat64_t b_real_even_f64x = svcvt_f64_f32_x(pred_even_b64x, b_real_f32x);
        svfloat64_t b_imag_even_f64x = svcvt_f64_f32_x(pred_even_b64x, b_imag_f32x);
        ab_real_f64x = svmla_f64_m(pred_even_b64x, ab_real_f64x, a_real_even_f64x, b_real_even_f64x);
        ab_real_f64x = svmls_f64_m(pred_even_b64x, ab_real_f64x, a_imag_even_f64x, b_imag_even_f64x);
        ab_imag_f64x = svmla_f64_m(pred_even_b64x, ab_imag_f64x, a_real_even_f64x, b_imag_even_f64x);
        ab_imag_f64x = svmla_f64_m(pred_even_b64x, ab_imag_f64x, a_imag_even_f64x, b_real_even_f64x);

        svbool_t pred_odd_b64x = svwhilelt_b64_u64(0u, remaining / 2);
        svfloat64_t a_real_odd_f64x = svcvt_f64_f32_x(pred_odd_b64x, svext_f32(a_real_f32x, a_real_f32x, 1));
        svfloat64_t a_imag_odd_f64x = svcvt_f64_f32_x(pred_odd_b64x, svext_f32(a_imag_f32x, a_imag_f32x, 1));
        svfloat64_t b_real_odd_f64x = svcvt_f64_f32_x(pred_odd_b64x, svext_f32(b_real_f32x, b_real_f32x, 1));
        svfloat64_t b_imag_odd_f64x = svcvt_f64_f32_x(pred_odd_b64x, svext_f32(b_imag_f32x, b_imag_f32x, 1));
        ab_real_f64x = svmla_f64_m(pred_odd_b64x, ab_real_f64x, a_real_odd_f64x, b_real_odd_f64x);
        ab_real_f64x = svmls_f64_m(pred_odd_b64x, ab_real_f64x, a_imag_odd_f64x, b_imag_odd_f64x);
        ab_imag_f64x = svmla_f64_m(pred_odd_b64x, ab_imag_f64x, a_real_odd_f64x, b_imag_odd_f64x);
        ab_imag_f64x = svmla_f64_m(pred_odd_b64x, ab_imag_f64x, a_imag_odd_f64x, b_real_odd_f64x);
    }
    results->real = svaddv_f64(svptrue_b64(), ab_real_f64x);
    results->imag = svaddv_f64(svptrue_b64(), ab_imag_f64x);
    NK_UNPOISON(&results->real, sizeof(results->real));
    NK_UNPOISON(&results->imag, sizeof(results->imag));
}

NK_PUBLIC void nk_vdot_f32c_sve(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                nk_f64c_t *results) {
    nk_size_t idx_pairs = 0;
    svfloat64_t ab_real_f64x = svdup_f64(0.);
    svfloat64_t ab_imag_f64x = svdup_f64(0.);
    for (; idx_pairs < count_pairs; idx_pairs += svcntw()) {
        svbool_t predicate_b32x = svwhilelt_b32_u64(idx_pairs, count_pairs);
        svfloat32x2_t a_f32x2 = svld2_f32(predicate_b32x, (nk_f32_t const *)(a_pairs + idx_pairs));
        svfloat32x2_t b_f32x2 = svld2_f32(predicate_b32x, (nk_f32_t const *)(b_pairs + idx_pairs));
        svfloat32_t a_real_f32x = svget2_f32(a_f32x2, 0);
        svfloat32_t a_imag_f32x = svget2_f32(a_f32x2, 1);
        svfloat32_t b_real_f32x = svget2_f32(b_f32x2, 0);
        svfloat32_t b_imag_f32x = svget2_f32(b_f32x2, 1);
        nk_size_t remaining = count_pairs - idx_pairs < svcntw() ? count_pairs - idx_pairs : svcntw();

        // svcvt_f64_f32_x widens only even-indexed f32 elements; svext by 1 shifts odd into even.
        svbool_t pred_even_b64x = svwhilelt_b64_u64(0u, (remaining + 1) / 2);
        svfloat64_t a_real_even_f64x = svcvt_f64_f32_x(pred_even_b64x, a_real_f32x);
        svfloat64_t a_imag_even_f64x = svcvt_f64_f32_x(pred_even_b64x, a_imag_f32x);
        svfloat64_t b_real_even_f64x = svcvt_f64_f32_x(pred_even_b64x, b_real_f32x);
        svfloat64_t b_imag_even_f64x = svcvt_f64_f32_x(pred_even_b64x, b_imag_f32x);
        ab_real_f64x = svmla_f64_m(pred_even_b64x, ab_real_f64x, a_real_even_f64x, b_real_even_f64x);
        ab_real_f64x = svmla_f64_m(pred_even_b64x, ab_real_f64x, a_imag_even_f64x, b_imag_even_f64x);
        ab_imag_f64x = svmla_f64_m(pred_even_b64x, ab_imag_f64x, a_real_even_f64x, b_imag_even_f64x);
        ab_imag_f64x = svmls_f64_m(pred_even_b64x, ab_imag_f64x, a_imag_even_f64x, b_real_even_f64x);

        svbool_t pred_odd_b64x = svwhilelt_b64_u64(0u, remaining / 2);
        svfloat64_t a_real_odd_f64x = svcvt_f64_f32_x(pred_odd_b64x, svext_f32(a_real_f32x, a_real_f32x, 1));
        svfloat64_t a_imag_odd_f64x = svcvt_f64_f32_x(pred_odd_b64x, svext_f32(a_imag_f32x, a_imag_f32x, 1));
        svfloat64_t b_real_odd_f64x = svcvt_f64_f32_x(pred_odd_b64x, svext_f32(b_real_f32x, b_real_f32x, 1));
        svfloat64_t b_imag_odd_f64x = svcvt_f64_f32_x(pred_odd_b64x, svext_f32(b_imag_f32x, b_imag_f32x, 1));
        ab_real_f64x = svmla_f64_m(pred_odd_b64x, ab_real_f64x, a_real_odd_f64x, b_real_odd_f64x);
        ab_real_f64x = svmla_f64_m(pred_odd_b64x, ab_real_f64x, a_imag_odd_f64x, b_imag_odd_f64x);
        ab_imag_f64x = svmla_f64_m(pred_odd_b64x, ab_imag_f64x, a_real_odd_f64x, b_imag_odd_f64x);
        ab_imag_f64x = svmls_f64_m(pred_odd_b64x, ab_imag_f64x, a_imag_odd_f64x, b_real_odd_f64x);
    }
    results->real = svaddv_f64(svptrue_b64(), ab_real_f64x);
    results->imag = svaddv_f64(svptrue_b64(), ab_imag_f64x);
    NK_UNPOISON(&results->real, sizeof(results->real));
    NK_UNPOISON(&results->imag, sizeof(results->imag));
}

NK_PUBLIC void nk_dot_f64_sve(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                              nk_f64_t *result) {
    // Dot2 (Ogita-Rump-Oishi) compensated accumulation via TwoProd + TwoSum
    nk_size_t idx_scalars = 0;
    svfloat64_t sum_f64x = svdup_f64(0.);
    svfloat64_t compensation_f64x = svdup_f64(0.);
    do {
        svbool_t predicate_b64x = svwhilelt_b64_u64(idx_scalars, count_scalars);
        svfloat64_t a_f64x = svld1_f64(predicate_b64x, a_scalars + idx_scalars);
        svfloat64_t b_f64x = svld1_f64(predicate_b64x, b_scalars + idx_scalars);
        // TwoProd: product = a*b, error = -(product - a*b) negated
        svfloat64_t product_f64x = svmul_f64_x(predicate_b64x, a_f64x, b_f64x);
        svfloat64_t product_error_f64x = svneg_f64_x(predicate_b64x,
                                                     svnmls_f64_x(predicate_b64x, product_f64x, a_f64x, b_f64x));
        // TwoSum: tentative_sum = sum + product
        svfloat64_t tentative_sum_f64x = svadd_f64_m(predicate_b64x, sum_f64x, product_f64x);
        svfloat64_t virtual_addend_f64x = svsub_f64_x(predicate_b64x, tentative_sum_f64x, sum_f64x);
        svfloat64_t sum_error_f64x = svadd_f64_x(
            predicate_b64x,
            svsub_f64_x(predicate_b64x, sum_f64x, svsub_f64_x(predicate_b64x, tentative_sum_f64x, virtual_addend_f64x)),
            svsub_f64_x(predicate_b64x, product_f64x, virtual_addend_f64x));
        sum_f64x = tentative_sum_f64x;
        compensation_f64x = svadd_f64_m(predicate_b64x, compensation_f64x,
                                        svadd_f64_x(predicate_b64x, sum_error_f64x, product_error_f64x));
        idx_scalars += svcntd();
    } while (idx_scalars < count_scalars);
    *result = nk_dot_stable_sum_f64_sve_(svptrue_b64(), sum_f64x, compensation_f64x);
}

NK_PUBLIC void nk_dot_f64c_sve(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                               nk_f64c_t *results) {
    // Dot2 compensated accumulation for complex dot product: (a_real + i*a_imag)(b_real + i*b_imag)
    // real = a_real*b_real - a_imag*b_imag, imag = a_real*b_imag + a_imag*b_real
    nk_size_t idx_pairs = 0;
    svfloat64_t sum_real_f64x = svdup_f64(0.);
    svfloat64_t comp_real_f64x = svdup_f64(0.);
    svfloat64_t sum_imag_f64x = svdup_f64(0.);
    svfloat64_t comp_imag_f64x = svdup_f64(0.);
    do {
        svbool_t predicate_b64x = svwhilelt_b64_u64(idx_pairs, count_pairs);
        svfloat64x2_t a_f64x2 = svld2_f64(predicate_b64x, (nk_f64_t const *)(a_pairs + idx_pairs));
        svfloat64x2_t b_f64x2 = svld2_f64(predicate_b64x, (nk_f64_t const *)(b_pairs + idx_pairs));
        svfloat64_t a_real_f64x = svget2_f64(a_f64x2, 0);
        svfloat64_t a_imag_f64x = svget2_f64(a_f64x2, 1);
        svfloat64_t b_real_f64x = svget2_f64(b_f64x2, 0);
        svfloat64_t b_imag_f64x = svget2_f64(b_f64x2, 1);

        // TwoProd + TwoSum for real part: sum_real += a_real*b_real
        {
            svfloat64_t product_f64x = svmul_f64_x(predicate_b64x, a_real_f64x, b_real_f64x);
            svfloat64_t product_error_f64x = svneg_f64_x(
                predicate_b64x, svnmls_f64_x(predicate_b64x, product_f64x, a_real_f64x, b_real_f64x));
            svfloat64_t tentative_sum_f64x = svadd_f64_m(predicate_b64x, sum_real_f64x, product_f64x);
            svfloat64_t virtual_addend_f64x = svsub_f64_x(predicate_b64x, tentative_sum_f64x, sum_real_f64x);
            svfloat64_t sum_error_f64x = svadd_f64_x(
                predicate_b64x,
                svsub_f64_x(predicate_b64x, sum_real_f64x,
                            svsub_f64_x(predicate_b64x, tentative_sum_f64x, virtual_addend_f64x)),
                svsub_f64_x(predicate_b64x, product_f64x, virtual_addend_f64x));
            sum_real_f64x = tentative_sum_f64x;
            comp_real_f64x = svadd_f64_m(predicate_b64x, comp_real_f64x,
                                         svadd_f64_x(predicate_b64x, sum_error_f64x, product_error_f64x));
        }
        // TwoProd + TwoSum for real part: sum_real -= a_imag*b_imag
        {
            svfloat64_t product_f64x = svmul_f64_x(predicate_b64x, a_imag_f64x, b_imag_f64x);
            svfloat64_t product_error_f64x = svneg_f64_x(
                predicate_b64x, svnmls_f64_x(predicate_b64x, product_f64x, a_imag_f64x, b_imag_f64x));
            svfloat64_t neg_product_f64x = svneg_f64_x(predicate_b64x, product_f64x);
            svfloat64_t neg_product_error_f64x = svneg_f64_x(predicate_b64x, product_error_f64x);
            svfloat64_t tentative_sum_f64x = svadd_f64_m(predicate_b64x, sum_real_f64x, neg_product_f64x);
            svfloat64_t virtual_addend_f64x = svsub_f64_x(predicate_b64x, tentative_sum_f64x, sum_real_f64x);
            svfloat64_t sum_error_f64x = svadd_f64_x(
                predicate_b64x,
                svsub_f64_x(predicate_b64x, sum_real_f64x,
                            svsub_f64_x(predicate_b64x, tentative_sum_f64x, virtual_addend_f64x)),
                svsub_f64_x(predicate_b64x, neg_product_f64x, virtual_addend_f64x));
            sum_real_f64x = tentative_sum_f64x;
            comp_real_f64x = svadd_f64_m(predicate_b64x, comp_real_f64x,
                                         svadd_f64_x(predicate_b64x, sum_error_f64x, neg_product_error_f64x));
        }
        // TwoProd + TwoSum for imaginary part: sum_imag += a_real*b_imag
        {
            svfloat64_t product_f64x = svmul_f64_x(predicate_b64x, a_real_f64x, b_imag_f64x);
            svfloat64_t product_error_f64x = svneg_f64_x(
                predicate_b64x, svnmls_f64_x(predicate_b64x, product_f64x, a_real_f64x, b_imag_f64x));
            svfloat64_t tentative_sum_f64x = svadd_f64_m(predicate_b64x, sum_imag_f64x, product_f64x);
            svfloat64_t virtual_addend_f64x = svsub_f64_x(predicate_b64x, tentative_sum_f64x, sum_imag_f64x);
            svfloat64_t sum_error_f64x = svadd_f64_x(
                predicate_b64x,
                svsub_f64_x(predicate_b64x, sum_imag_f64x,
                            svsub_f64_x(predicate_b64x, tentative_sum_f64x, virtual_addend_f64x)),
                svsub_f64_x(predicate_b64x, product_f64x, virtual_addend_f64x));
            sum_imag_f64x = tentative_sum_f64x;
            comp_imag_f64x = svadd_f64_m(predicate_b64x, comp_imag_f64x,
                                         svadd_f64_x(predicate_b64x, sum_error_f64x, product_error_f64x));
        }
        // TwoProd + TwoSum for imaginary part: sum_imag += a_imag*b_real
        {
            svfloat64_t product_f64x = svmul_f64_x(predicate_b64x, a_imag_f64x, b_real_f64x);
            svfloat64_t product_error_f64x = svneg_f64_x(
                predicate_b64x, svnmls_f64_x(predicate_b64x, product_f64x, a_imag_f64x, b_real_f64x));
            svfloat64_t tentative_sum_f64x = svadd_f64_m(predicate_b64x, sum_imag_f64x, product_f64x);
            svfloat64_t virtual_addend_f64x = svsub_f64_x(predicate_b64x, tentative_sum_f64x, sum_imag_f64x);
            svfloat64_t sum_error_f64x = svadd_f64_x(
                predicate_b64x,
                svsub_f64_x(predicate_b64x, sum_imag_f64x,
                            svsub_f64_x(predicate_b64x, tentative_sum_f64x, virtual_addend_f64x)),
                svsub_f64_x(predicate_b64x, product_f64x, virtual_addend_f64x));
            sum_imag_f64x = tentative_sum_f64x;
            comp_imag_f64x = svadd_f64_m(predicate_b64x, comp_imag_f64x,
                                         svadd_f64_x(predicate_b64x, sum_error_f64x, product_error_f64x));
        }
        idx_pairs += svcntd();
    } while (idx_pairs < count_pairs);
    svbool_t predicate_all_b64x = svptrue_b64();
    results->real = nk_dot_stable_sum_f64_sve_(predicate_all_b64x, sum_real_f64x, comp_real_f64x);
    results->imag = nk_dot_stable_sum_f64_sve_(predicate_all_b64x, sum_imag_f64x, comp_imag_f64x);
}

NK_PUBLIC void nk_vdot_f64c_sve(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                nk_f64c_t *results) {
    // Dot2 compensated conjugate dot product: conj(a) · b = (a_real - i*a_imag)(b_real + i*b_imag)
    // real = a_real*b_real + a_imag*b_imag, imag = a_real*b_imag - a_imag*b_real
    nk_size_t idx_pairs = 0;
    svfloat64_t sum_real_f64x = svdup_f64(0.);
    svfloat64_t comp_real_f64x = svdup_f64(0.);
    svfloat64_t sum_imag_f64x = svdup_f64(0.);
    svfloat64_t comp_imag_f64x = svdup_f64(0.);
    do {
        svbool_t predicate_b64x = svwhilelt_b64_u64(idx_pairs, count_pairs);
        svfloat64x2_t a_f64x2 = svld2_f64(predicate_b64x, (nk_f64_t const *)(a_pairs + idx_pairs));
        svfloat64x2_t b_f64x2 = svld2_f64(predicate_b64x, (nk_f64_t const *)(b_pairs + idx_pairs));
        svfloat64_t a_real_f64x = svget2_f64(a_f64x2, 0);
        svfloat64_t a_imag_f64x = svget2_f64(a_f64x2, 1);
        svfloat64_t b_real_f64x = svget2_f64(b_f64x2, 0);
        svfloat64_t b_imag_f64x = svget2_f64(b_f64x2, 1);

        // TwoProd + TwoSum for real part: sum_real += a_real*b_real
        {
            svfloat64_t product_f64x = svmul_f64_x(predicate_b64x, a_real_f64x, b_real_f64x);
            svfloat64_t product_error_f64x = svneg_f64_x(
                predicate_b64x, svnmls_f64_x(predicate_b64x, product_f64x, a_real_f64x, b_real_f64x));
            svfloat64_t tentative_sum_f64x = svadd_f64_m(predicate_b64x, sum_real_f64x, product_f64x);
            svfloat64_t virtual_addend_f64x = svsub_f64_x(predicate_b64x, tentative_sum_f64x, sum_real_f64x);
            svfloat64_t sum_error_f64x = svadd_f64_x(
                predicate_b64x,
                svsub_f64_x(predicate_b64x, sum_real_f64x,
                            svsub_f64_x(predicate_b64x, tentative_sum_f64x, virtual_addend_f64x)),
                svsub_f64_x(predicate_b64x, product_f64x, virtual_addend_f64x));
            sum_real_f64x = tentative_sum_f64x;
            comp_real_f64x = svadd_f64_m(predicate_b64x, comp_real_f64x,
                                         svadd_f64_x(predicate_b64x, sum_error_f64x, product_error_f64x));
        }
        // TwoProd + TwoSum for real part: sum_real += a_imag*b_imag (conjugate: + not -)
        {
            svfloat64_t product_f64x = svmul_f64_x(predicate_b64x, a_imag_f64x, b_imag_f64x);
            svfloat64_t product_error_f64x = svneg_f64_x(
                predicate_b64x, svnmls_f64_x(predicate_b64x, product_f64x, a_imag_f64x, b_imag_f64x));
            svfloat64_t tentative_sum_f64x = svadd_f64_m(predicate_b64x, sum_real_f64x, product_f64x);
            svfloat64_t virtual_addend_f64x = svsub_f64_x(predicate_b64x, tentative_sum_f64x, sum_real_f64x);
            svfloat64_t sum_error_f64x = svadd_f64_x(
                predicate_b64x,
                svsub_f64_x(predicate_b64x, sum_real_f64x,
                            svsub_f64_x(predicate_b64x, tentative_sum_f64x, virtual_addend_f64x)),
                svsub_f64_x(predicate_b64x, product_f64x, virtual_addend_f64x));
            sum_real_f64x = tentative_sum_f64x;
            comp_real_f64x = svadd_f64_m(predicate_b64x, comp_real_f64x,
                                         svadd_f64_x(predicate_b64x, sum_error_f64x, product_error_f64x));
        }
        // TwoProd + TwoSum for imaginary part: sum_imag += a_real*b_imag
        {
            svfloat64_t product_f64x = svmul_f64_x(predicate_b64x, a_real_f64x, b_imag_f64x);
            svfloat64_t product_error_f64x = svneg_f64_x(
                predicate_b64x, svnmls_f64_x(predicate_b64x, product_f64x, a_real_f64x, b_imag_f64x));
            svfloat64_t tentative_sum_f64x = svadd_f64_m(predicate_b64x, sum_imag_f64x, product_f64x);
            svfloat64_t virtual_addend_f64x = svsub_f64_x(predicate_b64x, tentative_sum_f64x, sum_imag_f64x);
            svfloat64_t sum_error_f64x = svadd_f64_x(
                predicate_b64x,
                svsub_f64_x(predicate_b64x, sum_imag_f64x,
                            svsub_f64_x(predicate_b64x, tentative_sum_f64x, virtual_addend_f64x)),
                svsub_f64_x(predicate_b64x, product_f64x, virtual_addend_f64x));
            sum_imag_f64x = tentative_sum_f64x;
            comp_imag_f64x = svadd_f64_m(predicate_b64x, comp_imag_f64x,
                                         svadd_f64_x(predicate_b64x, sum_error_f64x, product_error_f64x));
        }
        // TwoProd + TwoSum for imaginary part: sum_imag -= a_imag*b_real (conjugate: - not +)
        {
            svfloat64_t product_f64x = svmul_f64_x(predicate_b64x, a_imag_f64x, b_real_f64x);
            svfloat64_t product_error_f64x = svneg_f64_x(
                predicate_b64x, svnmls_f64_x(predicate_b64x, product_f64x, a_imag_f64x, b_real_f64x));
            svfloat64_t neg_product_f64x = svneg_f64_x(predicate_b64x, product_f64x);
            svfloat64_t neg_product_error_f64x = svneg_f64_x(predicate_b64x, product_error_f64x);
            svfloat64_t tentative_sum_f64x = svadd_f64_m(predicate_b64x, sum_imag_f64x, neg_product_f64x);
            svfloat64_t virtual_addend_f64x = svsub_f64_x(predicate_b64x, tentative_sum_f64x, sum_imag_f64x);
            svfloat64_t sum_error_f64x = svadd_f64_x(
                predicate_b64x,
                svsub_f64_x(predicate_b64x, sum_imag_f64x,
                            svsub_f64_x(predicate_b64x, tentative_sum_f64x, virtual_addend_f64x)),
                svsub_f64_x(predicate_b64x, neg_product_f64x, virtual_addend_f64x));
            sum_imag_f64x = tentative_sum_f64x;
            comp_imag_f64x = svadd_f64_m(predicate_b64x, comp_imag_f64x,
                                         svadd_f64_x(predicate_b64x, sum_error_f64x, neg_product_error_f64x));
        }
        idx_pairs += svcntd();
    } while (idx_pairs < count_pairs);
    svbool_t predicate_all_b64x = svptrue_b64();
    results->real = nk_dot_stable_sum_f64_sve_(predicate_all_b64x, sum_real_f64x, comp_real_f64x);
    results->imag = nk_dot_stable_sum_f64_sve_(predicate_all_b64x, sum_imag_f64x, comp_imag_f64x);
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
#endif // NK_DOT_SVE_H
