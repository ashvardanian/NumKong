/**
 *  @file include/numkong/each/sme.h
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief SIMD-accelerated elementwise helpers for Arm SME.
 *
 *  @sa include/numkong/each.h
 *
 *  Streaming-mode SVE register helpers shared by the SME kernels: the base-2 exponent in F32, its
 *  F16 polynomial fragment, and the integer U8 weight exponent.
 */
#ifndef NK_EACH_SME_H
#define NK_EACH_SME_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SME

#include <arm_sme.h>

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme")
#endif

/** @brief Vectorized `2^x` (SME streaming SVE); matches `nk_f32_exp2_serial_` to polynomial precision. */
NK_HELPER_INLINE svfloat32_t nk_exp2_f32x_sme_(svfloat32_t x_f32x) NK_STREAMING_ {
    svbool_t const predicate_all_b32x = svptrue_b32();
    x_f32x = svmax_f32_x(predicate_all_b32x, svmin_f32_x(predicate_all_b32x, x_f32x, svdup_f32(127.0f)),
                         svdup_f32(-125.0f));
    svfloat32_t const n_f32x = svrintn_f32_x(predicate_all_b32x, x_f32x);
    svfloat32_t const r_f32x = svsub_f32_x(predicate_all_b32x, x_f32x, n_f32x);
    svfloat32_t p_f32x = svdup_f32(9.61812910e-3f);
    p_f32x = svmad_f32_x(predicate_all_b32x, p_f32x, r_f32x, svdup_f32(5.55041087e-2f));
    p_f32x = svmad_f32_x(predicate_all_b32x, p_f32x, r_f32x, svdup_f32(2.40226507e-1f));
    p_f32x = svmad_f32_x(predicate_all_b32x, p_f32x, r_f32x, svdup_f32(6.93147181e-1f));
    p_f32x = svmad_f32_x(predicate_all_b32x, p_f32x, r_f32x, svdup_f32(1.0f));
    svint32_t n_i32x = svcvt_s32_f32_x(predicate_all_b32x, n_f32x);
    n_i32x = svlsl_n_s32_x(predicate_all_b32x, svadd_n_s32_x(predicate_all_b32x, n_i32x, 127), 23);
    return svmul_f32_x(predicate_all_b32x, p_f32x, svreinterpret_f32_s32(n_i32x));
}

/**
 *  @brief Degree-3 evaluation of `2^r` over the reduced fraction `r ∈ [-0.5, 0.5]`, 32 lanes at a time: the family
 *         coefficients with the degree-4 term dropped, which falls below the F16 resolution of the weights it feeds.
 */
NK_HELPER_INLINE svfloat16_t nk_exp2_polynomial_f16x_sme_(svfloat16_t reduced_f16x) NK_STREAMING_ {
    svbool_t const predicate_all_b16x = svptrue_b16();
    svfloat16_t poly_f16x = svdup_f16((__fp16)5.55041087e-2f);
    poly_f16x = svmad_f16_x(predicate_all_b16x, poly_f16x, reduced_f16x, svdup_f16((__fp16)2.40226507e-1f));
    poly_f16x = svmad_f16_x(predicate_all_b16x, poly_f16x, reduced_f16x, svdup_f16((__fp16)6.93147181e-1f));
    poly_f16x = svmad_f16_x(predicate_all_b16x, poly_f16x, reduced_f16x, svdup_f16((__fp16)1.0f));
    return poly_f16x;
}

/**
 *  @brief I-BERT-style integer `2^t`: takes a Q15 exponent in `[−10·2^15, 0]` and returns `round(2^t · 255)` as a U8
 *         weight in each I32 lane, through a degree-3 Q14 polynomial and a lane-variable shift, with no float.
 */
NK_HELPER_INLINE svint32_t nk_exp2_u8_i32x_sme_(svint32_t t_q15_i32x) NK_STREAMING_ {
    svbool_t const predicate_all_b32x = svptrue_b32();
    svint32_t const whole_i32x = svasr_n_s32_x(predicate_all_b32x, t_q15_i32x, 15); // floor, in [-10, 0]
    svint32_t const fraction_i32x = svand_n_s32_x(predicate_all_b32x, t_q15_i32x, 0x7FFF);
    svint32_t poly_i32x = svdup_s32(1296); // Chebyshev-fit 2^r coefficients in Q14, degree 3
    poly_i32x = svadd_n_s32_x(
        predicate_all_b32x,
        svasr_n_s32_x(predicate_all_b32x, svmul_s32_x(predicate_all_b32x, fraction_i32x, poly_i32x), 15), 3678);
    poly_i32x = svadd_n_s32_x(
        predicate_all_b32x,
        svasr_n_s32_x(predicate_all_b32x, svmul_s32_x(predicate_all_b32x, fraction_i32x, poly_i32x), 15), 11410);
    poly_i32x = svadd_n_s32_x(
        predicate_all_b32x,
        svasr_n_s32_x(predicate_all_b32x, svmul_s32_x(predicate_all_b32x, fraction_i32x, poly_i32x), 15), 16382);
    svint32_t const scaled_i32x = svmul_n_s32_x(predicate_all_b32x, poly_i32x, 255); // Q14 of 2^r · 255
    svuint32_t const shift_u32x = svreinterpret_u32_s32(svsubr_n_s32_x(predicate_all_b32x, whole_i32x, 14));
    svint32_t const bias_i32x = svlsl_s32_x(predicate_all_b32x, svdup_s32(1),
                                            svreinterpret_u32_s32(svsubr_n_s32_x(predicate_all_b32x, whole_i32x, 13)));
    return svasr_s32_x(predicate_all_b32x, svadd_s32_x(predicate_all_b32x, scaled_i32x, bias_i32x), shift_u32x);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SME
#endif // NK_TARGET_ARM64_
#endif // NK_EACH_SME_H
