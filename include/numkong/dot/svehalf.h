/**
 *  @brief SIMD-accelerated Dot Products for SVE FP16.
 *  @file include/numkong/dot/sve.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_svehalf_instructions ARM SVE+FP16 Instructions
 *
 *      Intrinsic      Instruction                V1
 *      svld1_f16      LD1H (Z.H, P/Z, [Xn])      4-6cy @ 2p
 *      svld2_f16      LD2H (Z.H, P/Z, [Xn])      6-8cy @ 1p
 *      svmla_f16_x    FMLA (Z.H, P/M, Z.H, Z.H)  4cy @ 2p
 *      svmls_f16_x    FMLS (Z.H, P/M, Z.H, Z.H)  4cy @ 2p
 *      svaddv_f16     FADDV (H, P, Z.H)          6cy @ 1p
 *      svdup_f16      DUP (Z.H, #imm)            1cy @ 2p
 *      svwhilelt_b16  WHILELT (P.H, Xn, Xm)      2cy @ 1p
 *      svptrue_b16    PTRUE (P.H, pattern)       1cy @ 2p
 *      svcnth         CNTH (Xd)                  1cy @ 2p
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5
 *  and Apple M4+ use 128-bit. Code using svcntb() adapts automatically, but wider vectors
 *  process more elements per iteration with identical latencies.
 *
 *  FP16 operations double the element count per vector compared to FP32, providing higher
 *  throughput at the cost of reduced precision. The FADDV reduction remains the bottleneck.
 */
#ifndef NK_DOT_SVEHALF_H
#define NK_DOT_SVEHALF_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SVEHALF

#include "numkong/types.h"      // `nk_f16_t`
#include "numkong/dot/serial.h" // `nk_u1x8_popcount_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+fp16")
#endif

NK_PUBLIC void nk_dot_f16_svehalf(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    nk_size_t idx_scalars = 0;
    svfloat32_t ab_f32x = svdup_f32(0);
    do {
        svbool_t predicate_b16x = svwhilelt_b16_u64(idx_scalars, count_scalars);
        svfloat16_t a_f16x = svld1_f16(predicate_b16x, (nk_f16_for_arm_simd_t const *)(a_scalars) + idx_scalars);
        svfloat16_t b_f16x = svld1_f16(predicate_b16x, (nk_f16_for_arm_simd_t const *)(b_scalars) + idx_scalars);
        nk_size_t remaining = count_scalars - idx_scalars < svcnth() ? count_scalars - idx_scalars : svcnth();

        // svcvt_f32_f16_x widens only even-indexed f16 elements; svext by 1 shifts odd into even.
        svbool_t pred_even_b32x = svwhilelt_b32_u64(0u, (remaining + 1) / 2);
        ab_f32x = svmla_f32_m(pred_even_b32x, ab_f32x, svcvt_f32_f16_x(pred_even_b32x, a_f16x),
                              svcvt_f32_f16_x(pred_even_b32x, b_f16x));

        svbool_t pred_odd_b32x = svwhilelt_b32_u64(0u, remaining / 2);
        ab_f32x = svmla_f32_m(pred_odd_b32x, ab_f32x, svcvt_f32_f16_x(pred_odd_b32x, svext_f16(a_f16x, a_f16x, 1)),
                              svcvt_f32_f16_x(pred_odd_b32x, svext_f16(b_f16x, b_f16x, 1)));

        idx_scalars += svcnth();
    } while (idx_scalars < count_scalars);
    *result = svaddv_f32(svptrue_b32(), ab_f32x);
    NK_UNPOISON(result, sizeof(*result));
}

NK_PUBLIC void nk_dot_f16c_svehalf(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f32c_t *results) {
    nk_size_t idx_scalars = 0;
    svfloat32_t ab_real_f32x = svdup_f32(0);
    svfloat32_t ab_imag_f32x = svdup_f32(0);
    do {
        svbool_t predicate_b16x = svwhilelt_b16_u64(idx_scalars, count_pairs);
        svfloat16x2_t a_f16x2x = svld2_f16(predicate_b16x, (nk_f16_for_arm_simd_t const *)(a_pairs) + idx_scalars * 2);
        svfloat16x2_t b_f16x2x = svld2_f16(predicate_b16x, (nk_f16_for_arm_simd_t const *)(b_pairs) + idx_scalars * 2);
        svfloat16_t ar_f16x = svget2_f16(a_f16x2x, 0), ai_f16x = svget2_f16(a_f16x2x, 1);
        svfloat16_t br_f16x = svget2_f16(b_f16x2x, 0), bi_f16x = svget2_f16(b_f16x2x, 1);
        nk_size_t remaining = count_pairs - idx_scalars < svcnth() ? count_pairs - idx_scalars : svcnth();

        // Even-indexed elements of each deinterleaved component
        svbool_t pred_even_b32x = svwhilelt_b32_u64(0u, (remaining + 1) / 2);
        svfloat32_t ar_even_f32x = svcvt_f32_f16_x(pred_even_b32x, ar_f16x);
        svfloat32_t ai_even_f32x = svcvt_f32_f16_x(pred_even_b32x, ai_f16x);
        svfloat32_t br_even_f32x = svcvt_f32_f16_x(pred_even_b32x, br_f16x);
        svfloat32_t bi_even_f32x = svcvt_f32_f16_x(pred_even_b32x, bi_f16x);
        ab_real_f32x = svmla_f32_m(pred_even_b32x, ab_real_f32x, ar_even_f32x, br_even_f32x);
        ab_real_f32x = svmls_f32_m(pred_even_b32x, ab_real_f32x, ai_even_f32x, bi_even_f32x);
        ab_imag_f32x = svmla_f32_m(pred_even_b32x, ab_imag_f32x, ar_even_f32x, bi_even_f32x);
        ab_imag_f32x = svmla_f32_m(pred_even_b32x, ab_imag_f32x, ai_even_f32x, br_even_f32x);

        // Odd-indexed elements via svext shift-by-1
        svbool_t pred_odd_b32x = svwhilelt_b32_u64(0u, remaining / 2);
        svfloat32_t ar_odd_f32x = svcvt_f32_f16_x(pred_odd_b32x, svext_f16(ar_f16x, ar_f16x, 1));
        svfloat32_t ai_odd_f32x = svcvt_f32_f16_x(pred_odd_b32x, svext_f16(ai_f16x, ai_f16x, 1));
        svfloat32_t br_odd_f32x = svcvt_f32_f16_x(pred_odd_b32x, svext_f16(br_f16x, br_f16x, 1));
        svfloat32_t bi_odd_f32x = svcvt_f32_f16_x(pred_odd_b32x, svext_f16(bi_f16x, bi_f16x, 1));
        ab_real_f32x = svmla_f32_m(pred_odd_b32x, ab_real_f32x, ar_odd_f32x, br_odd_f32x);
        ab_real_f32x = svmls_f32_m(pred_odd_b32x, ab_real_f32x, ai_odd_f32x, bi_odd_f32x);
        ab_imag_f32x = svmla_f32_m(pred_odd_b32x, ab_imag_f32x, ar_odd_f32x, bi_odd_f32x);
        ab_imag_f32x = svmla_f32_m(pred_odd_b32x, ab_imag_f32x, ai_odd_f32x, br_odd_f32x);

        idx_scalars += svcnth();
    } while (idx_scalars < count_pairs);
    results->real = svaddv_f32(svptrue_b32(), ab_real_f32x);
    results->imag = svaddv_f32(svptrue_b32(), ab_imag_f32x);
    NK_UNPOISON(&results->real, sizeof(results->real));
    NK_UNPOISON(&results->imag, sizeof(results->imag));
}

NK_PUBLIC void nk_vdot_f16c_svehalf(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f32c_t *results) {
    nk_size_t idx_scalars = 0;
    svfloat32_t ab_real_f32x = svdup_f32(0);
    svfloat32_t ab_imag_f32x = svdup_f32(0);
    do {
        svbool_t predicate_b16x = svwhilelt_b16_u64(idx_scalars, count_pairs);
        svfloat16x2_t a_f16x2x = svld2_f16(predicate_b16x, (nk_f16_for_arm_simd_t const *)(a_pairs) + idx_scalars * 2);
        svfloat16x2_t b_f16x2x = svld2_f16(predicate_b16x, (nk_f16_for_arm_simd_t const *)(b_pairs) + idx_scalars * 2);
        svfloat16_t ar_f16x = svget2_f16(a_f16x2x, 0), ai_f16x = svget2_f16(a_f16x2x, 1);
        svfloat16_t br_f16x = svget2_f16(b_f16x2x, 0), bi_f16x = svget2_f16(b_f16x2x, 1);
        nk_size_t remaining = count_pairs - idx_scalars < svcnth() ? count_pairs - idx_scalars : svcnth();

        // Even-indexed elements
        svbool_t pred_even_b32x = svwhilelt_b32_u64(0u, (remaining + 1) / 2);
        svfloat32_t ar_even_f32x = svcvt_f32_f16_x(pred_even_b32x, ar_f16x);
        svfloat32_t ai_even_f32x = svcvt_f32_f16_x(pred_even_b32x, ai_f16x);
        svfloat32_t br_even_f32x = svcvt_f32_f16_x(pred_even_b32x, br_f16x);
        svfloat32_t bi_even_f32x = svcvt_f32_f16_x(pred_even_b32x, bi_f16x);
        ab_real_f32x = svmla_f32_m(pred_even_b32x, ab_real_f32x, ar_even_f32x, br_even_f32x);
        ab_real_f32x = svmla_f32_m(pred_even_b32x, ab_real_f32x, ai_even_f32x, bi_even_f32x);
        ab_imag_f32x = svmla_f32_m(pred_even_b32x, ab_imag_f32x, ar_even_f32x, bi_even_f32x);
        ab_imag_f32x = svmls_f32_m(pred_even_b32x, ab_imag_f32x, ai_even_f32x, br_even_f32x);

        // Odd-indexed elements via svext shift-by-1
        svbool_t pred_odd_b32x = svwhilelt_b32_u64(0u, remaining / 2);
        svfloat32_t ar_odd_f32x = svcvt_f32_f16_x(pred_odd_b32x, svext_f16(ar_f16x, ar_f16x, 1));
        svfloat32_t ai_odd_f32x = svcvt_f32_f16_x(pred_odd_b32x, svext_f16(ai_f16x, ai_f16x, 1));
        svfloat32_t br_odd_f32x = svcvt_f32_f16_x(pred_odd_b32x, svext_f16(br_f16x, br_f16x, 1));
        svfloat32_t bi_odd_f32x = svcvt_f32_f16_x(pred_odd_b32x, svext_f16(bi_f16x, bi_f16x, 1));
        ab_real_f32x = svmla_f32_m(pred_odd_b32x, ab_real_f32x, ar_odd_f32x, br_odd_f32x);
        ab_real_f32x = svmla_f32_m(pred_odd_b32x, ab_real_f32x, ai_odd_f32x, bi_odd_f32x);
        ab_imag_f32x = svmla_f32_m(pred_odd_b32x, ab_imag_f32x, ar_odd_f32x, bi_odd_f32x);
        ab_imag_f32x = svmls_f32_m(pred_odd_b32x, ab_imag_f32x, ai_odd_f32x, br_odd_f32x);

        idx_scalars += svcnth();
    } while (idx_scalars < count_pairs);
    results->real = svaddv_f32(svptrue_b32(), ab_real_f32x);
    results->imag = svaddv_f32(svptrue_b32(), ab_imag_f32x);
    NK_UNPOISON(&results->real, sizeof(results->real));
    NK_UNPOISON(&results->imag, sizeof(results->imag));
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SVEHALF
#endif // NK_TARGET_ARM64_
#endif // NK_DOT_SVEHALF_H
