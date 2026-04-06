/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for SVE BF16.
 *  @file include/numkong/spatial/sve.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_svebfdot_instructions ARM SVE+BF16 Instructions
 *
 *      Intrinsic      Instruction                V1
 *      svld1_bf16     LD1H (Z.H, P/Z, [Xn])      4-6cy @ 2p
 *      svld1_u16      LD1H (Z.H, P/Z, [Xn])      4-6cy @ 2p
 *      svbfdot_f32    BFDOT (Z.S, Z.H, Z.H)      4cy @ 2p
 *      svmla_f32_x    FMLA (Z.S, P/M, Z.S, Z.S)  4cy @ 2p
 *      svsub_f32_x    FSUB (Z.S, P/M, Z.S, Z.S)  3cy @ 2p
 *      svaddv_f32     FADDV (S, P, Z.S)          6cy @ 1p
 *      svunpklo_u32   UUNPKLO (Z.S, Z.H)         2cy @ 2p
 *      svunpkhi_u32   UUNPKHI (Z.S, Z.H)         2cy @ 2p
 *      svlsl_n_u32_x  LSL (Z.S, P/M, Z.S, #imm)  2cy @ 2p
 *      svwhilelt_b16  WHILELT (P.H, Xn, Xm)      2cy @ 1p
 *      svwhilelt_b32  WHILELT (P.S, Xn, Xm)      2cy @ 1p
 *      svcnth         CNTH (Xd)                  1cy @ 2p
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5
 *  and Apple M4+ use 128-bit. Code using svcntb() adapts automatically, but wider vectors
 *  process more elements per iteration with identical latencies.
 *
 *  The BFDOT instruction fuses two BF16 multiplications with FP32 accumulation, providing
 *  efficient BF16 dot products without explicit conversion overhead.
 */
#ifndef NK_SPATIAL_SVEBFDOT_H
#define NK_SPATIAL_SVEBFDOT_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SVEBFDOT

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // `nk_f32_sqrt_neon`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+bf16")
#endif

NK_PUBLIC void nk_sqeuclidean_bf16_svebfdot(nk_bf16_t const *a_enum, nk_bf16_t const *b_enum, nk_size_t n,
                                            nk_f32_t *result) {
    nk_size_t i = 0;
    svfloat32_t d2_low_f32x = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t d2_high_f32x = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    nk_u16_t const *a = (nk_u16_t const *)(a_enum);
    nk_u16_t const *b = (nk_u16_t const *)(b_enum);
    do {
        svbool_t predicate_b16x = svwhilelt_b16_u64(i, n);
        svuint16_t a_u16x = svld1_u16(predicate_b16x, a + i);
        svuint16_t b_u16x = svld1_u16(predicate_b16x, b + i);

        // There is no `bf16` subtraction in SVE, so we need to convert to `u32` and shift.
        svbool_t predicate_low_b32x = svwhilelt_b32_u64(i, n);
        svbool_t predicate_high_b32x = svwhilelt_b32_u64(i + svcnth() / 2, n);
        svfloat32_t a_low_f32x = svreinterpret_f32_u32(svlsl_n_u32_x(predicate_low_b32x, svunpklo_u32(a_u16x), 16));
        svfloat32_t a_high_f32x = svreinterpret_f32_u32(svlsl_n_u32_x(predicate_high_b32x, svunpkhi_u32(a_u16x), 16));
        svfloat32_t b_low_f32x = svreinterpret_f32_u32(svlsl_n_u32_x(predicate_low_b32x, svunpklo_u32(b_u16x), 16));
        svfloat32_t b_high_f32x = svreinterpret_f32_u32(svlsl_n_u32_x(predicate_high_b32x, svunpkhi_u32(b_u16x), 16));

        svfloat32_t a_minus_b_low_f32x = svsub_f32_x(predicate_low_b32x, a_low_f32x, b_low_f32x);
        svfloat32_t a_minus_b_high_f32x = svsub_f32_x(predicate_high_b32x, a_high_f32x, b_high_f32x);
        d2_low_f32x = svmla_f32_m(predicate_low_b32x, d2_low_f32x, a_minus_b_low_f32x, a_minus_b_low_f32x);
        d2_high_f32x = svmla_f32_m(predicate_high_b32x, d2_high_f32x, a_minus_b_high_f32x, a_minus_b_high_f32x);
        i += svcnth();
    } while (i < n);
    nk_f32_t d2_low = svaddv_f32(svptrue_b32(), d2_low_f32x);
    nk_f32_t d2_high = svaddv_f32(svptrue_b32(), d2_high_f32x);
    NK_UNPOISON(&d2_low, sizeof(d2_low));
    NK_UNPOISON(&d2_high, sizeof(d2_high));
    nk_f32_t d2 = d2_low + d2_high;
    *result = d2;
}
NK_PUBLIC void nk_euclidean_bf16_svebfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_bf16_svebfdot(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_bf16_svebfdot(nk_bf16_t const *a_enum, nk_bf16_t const *b_enum, nk_size_t n,
                                        nk_f32_t *result) {
    nk_size_t i = 0;
    svfloat32_t ab_f32x = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t a2_f32x = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t b2_f32x = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    nk_bf16_for_arm_simd_t const *a = (nk_bf16_for_arm_simd_t const *)(a_enum);
    nk_bf16_for_arm_simd_t const *b = (nk_bf16_for_arm_simd_t const *)(b_enum);
    do {
        svbool_t predicate_b16x = svwhilelt_b16_u64(i, n);
        svbfloat16_t a_bf16x = svld1_bf16(predicate_b16x, a + i);
        svbfloat16_t b_bf16x = svld1_bf16(predicate_b16x, b + i);
        ab_f32x = svbfdot_f32(ab_f32x, a_bf16x, b_bf16x);
        a2_f32x = svbfdot_f32(a2_f32x, a_bf16x, a_bf16x);
        b2_f32x = svbfdot_f32(b2_f32x, b_bf16x, b_bf16x);
        i += svcnth();
    } while (i < n);

    nk_f32_t ab = svaddv_f32(svptrue_b32(), ab_f32x);
    nk_f32_t a2 = svaddv_f32(svptrue_b32(), a2_f32x);
    nk_f32_t b2 = svaddv_f32(svptrue_b32(), b2_f32x);
    NK_UNPOISON(&ab, sizeof(ab));
    NK_UNPOISON(&a2, sizeof(a2));
    NK_UNPOISON(&b2, sizeof(b2));
    *result = nk_angular_normalize_f32_neon_(ab, a2, b2);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SVEBFDOT
#endif // NK_TARGET_ARM64_
#endif // NK_SPATIAL_SVEBFDOT_H
