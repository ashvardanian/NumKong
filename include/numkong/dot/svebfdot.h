/**
 *  @brief SIMD-accelerated Dot Products for SVE BF16.
 *  @file include/numkong/dot/svebfdot.h
 *  @author Ash Vardanian
 *  @date March 16, 2026
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_svebfdot_instructions ARM SVE+BF16 Instructions
 *
 *      Intrinsic      Instruction            V1
 *      svld1_bf16     LD1H (Z.H, P/Z, [Xn])  4-6cy @ 2p
 *      svbfdot_f32    BFDOT (Z.S, Z.H, Z.H)  4cy @ 2p
 *      svaddv_f32     FADDV (S, P, Z.S)      6cy @ 1p
 *      svdup_f32      DUP (Z.S, #imm)        1cy @ 2p
 *      svwhilelt_b16  WHILELT (P.H, Xn, Xm)  2cy @ 1p
 *      svcnth         CNTH (Xd)              1cy @ 2p
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5
 *  and Apple M4+ use 128-bit. Code using svcnth() adapts automatically, but wider vectors
 *  process more elements per iteration with identical latencies.
 *
 *  The BFDOT instruction fuses two BF16 multiplications with FP32 accumulation per lane,
 *  providing 4x the throughput of convert-then-FMA sequences. Each BFDOT processes
 *  pairs of BF16 values, accumulating directly into FP32 without explicit conversion.
 */
#ifndef NK_DOT_SVEBFDOT_H
#define NK_DOT_SVEBFDOT_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SVEBFDOT

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+bf16")
#endif

NK_PUBLIC void nk_dot_bf16_svebfdot(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                    nk_f32_t *result) {
    nk_size_t idx_scalars = 0;
    svfloat32_t sum_f32x = svdup_f32(0);
    nk_bf16_for_arm_simd_t const *a = (nk_bf16_for_arm_simd_t const *)(a_scalars);
    nk_bf16_for_arm_simd_t const *b = (nk_bf16_for_arm_simd_t const *)(b_scalars);
    do {
        svbool_t predicate_b16x = svwhilelt_b16_u64(idx_scalars, count_scalars);
        svbfloat16_t a_bf16x = svld1_bf16(predicate_b16x, a + idx_scalars);
        svbfloat16_t b_bf16x = svld1_bf16(predicate_b16x, b + idx_scalars);
        sum_f32x = svbfdot_f32(sum_f32x, a_bf16x, b_bf16x);
        idx_scalars += svcnth();
    } while (idx_scalars < count_scalars);
    *result = svaddv_f32(svptrue_b32(), sum_f32x);
    NK_UNPOISON(result, sizeof(*result));
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
#endif // NK_DOT_SVEBFDOT_H
