/**
 *  @file include/numkong/trigonometry/svehalf.h
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief SIMD-accelerated trigonometric functions for SVE FP16.
 *
 *  @sa include/numkong/trigonometry.h
 *
 *  @section trigonometry_svehalf_instructions ARM SVE+FP16 Instructions
 *
 *  @verbatim
 *  Intrinsic        Instruction
 *  svmla_f16_x      FMLA (Z.H, P/M, Z.H, Z.H)
 *  svmls_n_f16_x    FMLS (Z.H, P/M, Z.H, Z.H)
 *  svrintn_f16_x    FRINTN (Z.H, P/M, Z.H)
 *  svdivr_n_f16_m   FDIVR (Z.H, P/M, Z.H, Z.H)
 *  svacgt_n_f16     FACGT (P.H, P/Z, Z.H, Z.H)
 *  svcvt_f32_f16_x  FCVT (Z.S, P/M, Z.H)
 *  @endverbatim
 *
 *  Sine, cosine and arctangent evaluate in F16 on every lane, with polynomials fitted for F16.
 *  Sine and cosine reduce in F16 while every lane stays within |x| ≤ 256, and in F32 otherwise.
 *  The F32 reduction widens even and odd F16 lanes apart and interleaves them back.
 */
#ifndef NUMKONG_TRIGONOMETRY_SVEHALF_H
#define NUMKONG_TRIGONOMETRY_SVEHALF_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_SVEHALF

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+fp16")
#endif

/** Sine of F16 angles already reduced to about [-π/2, π/2], odd in the reduced angle. */
NUMKONG_INLINE svfloat16_t nk_sin_reduced_f16x_svehalf_(svfloat16_t reduced_f16x) {
    svbool_t const predicate_b16x = svptrue_b16();
    // Degree-7 odd polynomial with a unit linear term, coefficients searched in F16 arithmetic
    svfloat16_t const squared_f16x = svmul_f16_x(predicate_b16x, reduced_f16x, reduced_f16x);
    svfloat16_t polynomial_f16x = svmla_f16_x(predicate_b16x, svdup_n_f16(+0.00824737548828125), squared_f16x,
                                              svdup_n_f16(-0.00018310546875));
    polynomial_f16x = svmla_f16_x(predicate_b16x, svdup_n_f16(-0.16650390625), polynomial_f16x, squared_f16x);
    svfloat16_t const cubed_f16x = svmul_f16_x(predicate_b16x, reduced_f16x, squared_f16x);
    return svmla_f16_x(predicate_b16x, reduced_f16x, cubed_f16x, polynomial_f16x);
}

/** Subtracts @p multiples_f16x of π from @p angles_f16x, for F16 angles up to 256. */
NUMKONG_INLINE svfloat16_t nk_reduce_pi_f16x_svehalf_(svfloat16_t angles_f16x, svfloat16_t multiples_f16x) {
    svbool_t const predicate_b16x = svptrue_b16();
    // π in three F16 parts, the last scaled by 2¹² to stay clear of F16 subnormals
    svfloat16_t reduced_f16x = svmls_n_f16_x(predicate_b16x, angles_f16x, multiples_f16x, 3.140625);
    reduced_f16x = svmls_n_f16_x(predicate_b16x, reduced_f16x, multiples_f16x, 0.0009675025939941406);
    svfloat16_t const scaled_multiples_f16x = svmul_n_f16_x(predicate_b16x, multiples_f16x, 0.000244140625);
    return svmls_n_f16_x(predicate_b16x, reduced_f16x, scaled_multiples_f16x, 0.0006184577941894531);
}

/** Subtracts @p multiples_f32x of π from @p angles_f32x, for F16 angles of any magnitude. */
NUMKONG_INLINE svfloat32_t nk_reduce_pi_f32x_svehalf_(svfloat32_t angles_f32x, svfloat32_t multiples_f32x) {
    svbool_t const predicate_b32x = svptrue_b32();
    svfloat32_t const reduced_f32x = svmls_n_f32_x(predicate_b32x, angles_f32x, multiples_f32x, 3.140625f);
    return svmls_n_f32_x(predicate_b32x, reduced_f32x, multiples_f32x, 9.676535897e-4f);
}

/** Flips the sign of @p values_f32x in the lanes where @p flips_i32x is odd. */
NUMKONG_INLINE svfloat32_t nk_flip_odd_f32x_svehalf_(svfloat32_t values_f32x, svint32_t flips_i32x) {
    svbool_t const predicate_b32x = svptrue_b32();
    svuint32_t const signs_u32x = svlsl_n_u32_x(predicate_b32x, svreinterpret_u32_s32(flips_i32x), 31);
    return svreinterpret_f32_u32(sveor_u32_x(predicate_b32x, svreinterpret_u32_f32(values_f32x), signs_u32x));
}

/** Flips the sign of @p values_f16x in the lanes where @p flips_i16x is odd. */
NUMKONG_INLINE svfloat16_t nk_flip_odd_f16x_svehalf_(svfloat16_t values_f16x, svint16_t flips_i16x) {
    svbool_t const predicate_b16x = svptrue_b16();
    svuint16_t const signs_u16x = svlsl_n_u16_x(predicate_b16x, svreinterpret_u16_s16(flips_i16x), 15);
    return svreinterpret_f16_u16(sveor_u16_x(predicate_b16x, svreinterpret_u16_f16(values_f16x), signs_u16x));
}

/** Widens the even F16 lanes of @p angles_f16x to F32, reduces them by the nearest multiple of π,
 *  negated where it is odd, and narrows them back into the even lanes. */
NUMKONG_INLINE svfloat16_t nk_sin_reduce_even_f16x_svehalf_(svfloat16_t angles_f16x) {
    svbool_t const predicate_b32x = svptrue_b32();
    svfloat32_t const angles_f32x = svcvt_f32_f16_x(predicate_b32x, angles_f16x);
    svfloat32_t const quotients_f32x = svmul_n_f32_x(predicate_b32x, angles_f32x, 0.31830988618379067154f);
    svfloat32_t const multiples_f32x = svrintn_f32_x(predicate_b32x, quotients_f32x);
    svfloat32_t const reduced_f32x = nk_reduce_pi_f32x_svehalf_(angles_f32x, multiples_f32x);
    svint32_t const flips_i32x = svcvt_s32_f32_x(predicate_b32x, multiples_f32x);
    return svcvt_f16_f32_x(predicate_b32x, nk_flip_odd_f32x_svehalf_(reduced_f32x, flips_i32x));
}

/** Widens the even F16 lanes of @p angles_f16x to F32, reduces them by the nearest odd multiple of
 *  π/2, signed so its sine is their cosine, and narrows them back into the even lanes. */
NUMKONG_INLINE svfloat16_t nk_cos_reduce_even_f16x_svehalf_(svfloat16_t angles_f16x) {
    svbool_t const predicate_b32x = svptrue_b32();
    svfloat32_t const angles_f32x = svcvt_f32_f16_x(predicate_b32x, angles_f16x);
    svfloat32_t const quotients_f32x = svmad_n_f32_x(predicate_b32x, angles_f32x, svdup_n_f32(0.31830988618379067154f),
                                                     -0.5f);
    svfloat32_t const multiples_f32x = svrintn_f32_x(predicate_b32x, quotients_f32x);
    svfloat32_t const offsets_f32x = svadd_n_f32_x(predicate_b32x, multiples_f32x, 0.5f);
    svfloat32_t const reduced_f32x = nk_reduce_pi_f32x_svehalf_(angles_f32x, offsets_f32x);
    svint32_t const flips_i32x = svnot_s32_x(predicate_b32x, svcvt_s32_f32_x(predicate_b32x, multiples_f32x));
    return svcvt_f16_f32_x(predicate_b32x, nk_flip_odd_f32x_svehalf_(reduced_f32x, flips_i32x));
}

/** Sine of F16 angles within one F16 ULP. */
NUMKONG_INLINE svfloat16_t nk_sin_f16x_svehalf_(svfloat16_t angles_f16x) {
    svbool_t const predicate_b16x = svptrue_b16();
    svfloat16_t reduced_f16x;
    // The F16 reduction holds one ULP only up to |x| ≤ 256
    if (svptest_any(predicate_b16x, svacgt_n_f16(predicate_b16x, angles_f16x, 256))) {
        svfloat16_t const even_f16x = nk_sin_reduce_even_f16x_svehalf_(angles_f16x);
        svfloat16_t const odd_f16x = nk_sin_reduce_even_f16x_svehalf_(svext_f16(angles_f16x, angles_f16x, 1));
        reduced_f16x = svtrn1_f16(even_f16x, odd_f16x);
    }
    else {
        svfloat16_t const quotients_f16x = svmul_n_f16_x(predicate_b16x, angles_f16x, 0.31830988618379067154);
        svfloat16_t const multiples_f16x = svrintn_f16_x(predicate_b16x, quotients_f16x);
        reduced_f16x = nk_reduce_pi_f16x_svehalf_(angles_f16x, multiples_f16x);
        reduced_f16x = nk_flip_odd_f16x_svehalf_(reduced_f16x, svcvt_s16_f16_x(predicate_b16x, multiples_f16x));
    }
    return nk_sin_reduced_f16x_svehalf_(reduced_f16x);
}

/** Cosine of F16 angles within one F16 ULP. */
NUMKONG_INLINE svfloat16_t nk_cos_f16x_svehalf_(svfloat16_t angles_f16x) {
    svbool_t const predicate_b16x = svptrue_b16();
    svfloat16_t reduced_f16x;
    // The F16 reduction holds one ULP only up to |x| ≤ 256
    if (svptest_any(predicate_b16x, svacgt_n_f16(predicate_b16x, angles_f16x, 256))) {
        svfloat16_t const even_f16x = nk_cos_reduce_even_f16x_svehalf_(angles_f16x);
        svfloat16_t const odd_f16x = nk_cos_reduce_even_f16x_svehalf_(svext_f16(angles_f16x, angles_f16x, 1));
        reduced_f16x = svtrn1_f16(even_f16x, odd_f16x);
    }
    else {
        svfloat16_t const quotients_f16x = svmad_n_f16_x(predicate_b16x, angles_f16x,
                                                         svdup_n_f16(0.31830988618379067154), -0.5);
        svfloat16_t const multiples_f16x = svrintn_f16_x(predicate_b16x, quotients_f16x);
        svfloat16_t const offsets_f16x = svadd_n_f16_x(predicate_b16x, multiples_f16x, 0.5);
        svint16_t const flips_i16x = svnot_s16_x(predicate_b16x, svcvt_s16_f16_x(predicate_b16x, multiples_f16x));
        reduced_f16x = nk_reduce_pi_f16x_svehalf_(angles_f16x, offsets_f16x);
        reduced_f16x = nk_flip_odd_f16x_svehalf_(reduced_f16x, flips_i16x);
    }
    return nk_sin_reduced_f16x_svehalf_(reduced_f16x);
}

/** Arctangent of F16 values within one F16 ULP. */
NUMKONG_INLINE svfloat16_t nk_atan_f16x_svehalf_(svfloat16_t values_f16x) {
    svbool_t const predicate_b16x = svptrue_b16();

    // Fold |x| > 1 into [0, 1] through atan(x) = π/2 - atan(1/x)
    svfloat16_t const magnitudes_f16x = svabs_f16_x(predicate_b16x, values_f16x);
    svbool_t const folded_b16x = svacgt_n_f16(predicate_b16x, values_f16x, 1);
    svfloat16_t const reduced_f16x = svdivr_n_f16_m(folded_b16x, magnitudes_f16x, 1);

    // Folded lanes add the low part of π/2 first, as its high part alone is half an F16 ULP off
    svfloat16_t const signed_f16x = svneg_f16_m(reduced_f16x, folded_b16x, reduced_f16x);
    svfloat16_t const bases_f16x = svsubr_n_f16_m(folded_b16x, reduced_f16x, 0.0004837512969970703);

    // Degree-7 odd polynomial with a unit linear term, coefficients searched in F16 arithmetic
    svfloat16_t const squared_f16x = svmul_f16_x(predicate_b16x, reduced_f16x, reduced_f16x);
    svfloat16_t polynomial_f16x = svmla_f16_x(predicate_b16x, svdup_n_f16(+0.1600341796875), squared_f16x,
                                              svdup_n_f16(-0.046722412109375));
    polynomial_f16x = svmla_f16_x(predicate_b16x, svdup_n_f16(-0.328125), polynomial_f16x, squared_f16x);
    svfloat16_t const cubed_f16x = svmul_f16_x(predicate_b16x, signed_f16x, squared_f16x);
    svfloat16_t results_f16x = svmla_f16_x(predicate_b16x, bases_f16x, cubed_f16x, polynomial_f16x);
    results_f16x = svadd_n_f16_m(folded_b16x, results_f16x, 1.5703125);

    // Results are non-negative, so XOR-ing the input's sign bit copies it
    svuint16_t const signs_u16x = svand_n_u16_x(predicate_b16x, svreinterpret_u16_f16(values_f16x), 0x8000);
    return svreinterpret_f16_u16(sveor_u16_x(predicate_b16x, svreinterpret_u16_f16(results_f16x), signs_u16x));
}

NUMKONG_API nk_status_t nk_trig_sin_f16_svehalf(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f16_for_arm_simd_t const *inputs = (nk_f16_for_arm_simd_t const *)ins;
    nk_f16_for_arm_simd_t *outputs = (nk_f16_for_arm_simd_t *)outs;
    for (nk_size_t i = 0; i < n; i += svcnth()) {
        svbool_t const predicate_b16x = svwhilelt_b16_u64(i, n);
        svfloat16_t const angles_f16x = svld1_f16(predicate_b16x, inputs + i);
        svst1_f16(predicate_b16x, outputs + i, nk_sin_f16x_svehalf_(angles_f16x));
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_trig_cos_f16_svehalf(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f16_for_arm_simd_t const *inputs = (nk_f16_for_arm_simd_t const *)ins;
    nk_f16_for_arm_simd_t *outputs = (nk_f16_for_arm_simd_t *)outs;
    for (nk_size_t i = 0; i < n; i += svcnth()) {
        svbool_t const predicate_b16x = svwhilelt_b16_u64(i, n);
        svfloat16_t const angles_f16x = svld1_f16(predicate_b16x, inputs + i);
        svst1_f16(predicate_b16x, outputs + i, nk_cos_f16x_svehalf_(angles_f16x));
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_trig_atan_f16_svehalf(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f16_for_arm_simd_t const *inputs = (nk_f16_for_arm_simd_t const *)ins;
    nk_f16_for_arm_simd_t *outputs = (nk_f16_for_arm_simd_t *)outs;
    for (nk_size_t i = 0; i < n; i += svcnth()) {
        svbool_t const predicate_b16x = svwhilelt_b16_u64(i, n);
        svfloat16_t const values_f16x = svld1_f16(predicate_b16x, inputs + i);
        svst1_f16(predicate_b16x, outputs + i, nk_atan_f16x_svehalf_(values_f16x));
    }
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_SVEHALF
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_TRIGONOMETRY_SVEHALF_H
