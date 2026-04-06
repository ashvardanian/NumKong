/**
 *  @brief SIMD-accelerated Dot Products for SVE SDOT.
 *  @file include/numkong/dot/svesdot.h
 *  @author Ash Vardanian
 *  @date April 3, 2026
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_svesdot_instructions ARM SVE+DotProd Instructions
 *
 *      Intrinsic      Instruction             V1
 *      svld1_s8       LD1B (Z.B, P/Z, [Xn])   4-6cy @ 2p
 *      svld1_u8       LD1B (Z.B, P/Z, [Xn])   4-6cy @ 2p
 *      svdot_s32      SDOT (Z.S, Z.B, Z.B)    3cy @ 2p
 *      svdot_u32      UDOT (Z.S, Z.B, Z.B)    3cy @ 2p
 *      svaddv_s32     SADDV (D, P, Z.S)       6cy @ 1p
 *      svaddv_u32     UADDV (D, P, Z.S)       6cy @ 1p
 *      svdup_s32      DUP (Z.S, #imm)         1cy @ 2p
 *      svwhilelt_b8   WHILELT (P.B, Xn, Xm)   2cy @ 1p
 *      svcntb         CNTB (Xd)               1cy @ 2p
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5
 *  and Apple M4+ use 128-bit. Code using svcntb() adapts automatically, but wider vectors
 *  process more elements per iteration with identical latencies.
 *
 *  The SDOT/UDOT instructions fuse four int8 multiplications with int32 accumulation per lane,
 *  providing the same 4-way dot product as NEON SDOT but with scalable vector widths.
 *  On 256-bit SVE, this processes 32 int8 elements per instruction vs NEON's fixed 16.
 */
#ifndef NK_DOT_SVESDOT_H
#define NK_DOT_SVESDOT_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SVESDOT

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+dotprod"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+dotprod")
#endif

NK_PUBLIC void nk_dot_i8_svesdot(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_i32_t *result) {
    nk_size_t idx_scalars = 0;
    svint32_t sum_i32x = svdup_s32(0);
    do {
        svbool_t predicate_b8x = svwhilelt_b8_u64(idx_scalars, count_scalars);
        svint8_t a_i8x = svld1_s8(predicate_b8x, a_scalars + idx_scalars);
        svint8_t b_i8x = svld1_s8(predicate_b8x, b_scalars + idx_scalars);
        sum_i32x = svdot_s32(sum_i32x, a_i8x, b_i8x);
        idx_scalars += svcntb();
    } while (idx_scalars < count_scalars);
    *result = (nk_i32_t)svaddv_s32(svptrue_b32(), sum_i32x);
    NK_UNPOISON(result, sizeof(*result));
}

NK_PUBLIC void nk_dot_u8_svesdot(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_u32_t *result) {
    nk_size_t idx_scalars = 0;
    svuint32_t sum_u32x = svdup_u32(0);
    do {
        svbool_t predicate_b8x = svwhilelt_b8_u64(idx_scalars, count_scalars);
        svuint8_t a_u8x = svld1_u8(predicate_b8x, a_scalars + idx_scalars);
        svuint8_t b_u8x = svld1_u8(predicate_b8x, b_scalars + idx_scalars);
        sum_u32x = svdot_u32(sum_u32x, a_u8x, b_u8x);
        idx_scalars += svcntb();
    } while (idx_scalars < count_scalars);
    *result = (nk_u32_t)svaddv_u32(svptrue_b32(), sum_u32x);
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

#endif // NK_TARGET_SVESDOT
#endif // NK_TARGET_ARM64_
#endif // NK_DOT_SVESDOT_H
