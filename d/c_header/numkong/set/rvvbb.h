/**
 *  @brief SIMD-accelerated Set Similarity Measures for RISC-V with Zvbb.
 *  @file include/numkong/set/rvvbb.h
 *  @author Ash Vardanian
 *  @date February 9, 2026
 *
 *  @sa include/numkong/set.h
 *
 *  Zvbb (Vector Basic Bit-manipulation) provides native per-element popcount via `vcpop.v`,
 *  replacing the 11-instruction SWAR approach in set/rvv.h with a single instruction.
 *
 *  Only `nk_hamming_u1` and `nk_jaccard_u1` benefit from Zvbb (they need byte-level popcount).
 *  Integer set operations (hamming_u8, jaccard_u16, jaccard_u32) use mask-level vcpop.m
 *  which is already available in base RVV 1.0.
 *
 *  Requires: RVV 1.0 + Zvbb extension (GCC 14+ or Clang 18+)
 */
#ifndef NK_SET_RVVBB_H
#define NK_SET_RVVBB_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVVBB

#include "numkong/types.h"

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v,+zvbb"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v,+zvbb")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief  Native per-element popcount using Zvbb vcpop.v (1 instruction).
 *
 *  Replaces the 11-instruction SWAR approach in nk_popcount_u8m4_rvv_.
 */
NK_INTERNAL vuint8m4_t nk_popcount_u8m4_rvvbb_(vuint8m4_t v_u8m4) {
    vuint8m4_t result_u8m4;
    __asm__ volatile("vcpop.v %0, %1" : "=vr"(result_u8m4) : "vr"(v_u8m4));
    return result_u8m4;
}

NK_PUBLIC void nk_hamming_u1_rvvbb(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_size_t count_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);

    vuint32m1_t sum_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);

    nk_size_t i = 0;
    for (nk_size_t vector_length; i + 1 <= count_bytes; i += vector_length) {
        vector_length = __riscv_vsetvl_e8m4(count_bytes - i);

        vuint8m4_t a_u8m4 = __riscv_vle8_v_u8m4(a + i, vector_length);
        vuint8m4_t b_u8m4 = __riscv_vle8_v_u8m4(b + i, vector_length);
        vuint8m4_t xor_u8m4 = __riscv_vxor_vv_u8m4(a_u8m4, b_u8m4, vector_length);

        // Native per-element popcount via Zvbb (1 instruction vs 11 SWAR)
        vuint8m4_t popcount_u8m4 = nk_popcount_u8m4_rvvbb_(xor_u8m4);

        // Widen to u16 and accumulate via widening reduction sum
        vuint16m8_t popcount_u16m8 = __riscv_vwaddu_vx_u16m8(popcount_u8m4, 0, vector_length);
        sum_u32m1 = __riscv_vwredsumu_vs_u16m8_u32m1(popcount_u16m8, sum_u32m1, vector_length);
    }

    *result = __riscv_vmv_x_s_u32m1_u32(sum_u32m1);
}

NK_PUBLIC void nk_jaccard_u1_rvvbb(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t count_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);

    vuint32m1_t intersection_sum_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    vuint32m1_t union_sum_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);

    nk_size_t i = 0;
    for (nk_size_t vector_length; i + 1 <= count_bytes; i += vector_length) {
        vector_length = __riscv_vsetvl_e8m4(count_bytes - i);

        vuint8m4_t a_u8m4 = __riscv_vle8_v_u8m4(a + i, vector_length);
        vuint8m4_t b_u8m4 = __riscv_vle8_v_u8m4(b + i, vector_length);

        vuint8m4_t intersection_u8m4 = __riscv_vand_vv_u8m4(a_u8m4, b_u8m4, vector_length);
        vuint8m4_t union_u8m4 = __riscv_vor_vv_u8m4(a_u8m4, b_u8m4, vector_length);

        // Native per-element popcount via Zvbb
        vuint8m4_t intersection_popcount_u8m4 = nk_popcount_u8m4_rvvbb_(intersection_u8m4);
        vuint8m4_t union_popcount_u8m4 = nk_popcount_u8m4_rvvbb_(union_u8m4);

        // Widen and accumulate
        vuint16m8_t intersection_popcount_u16m8 = __riscv_vwaddu_vx_u16m8(intersection_popcount_u8m4, 0, vector_length);
        vuint16m8_t union_popcount_u16m8 = __riscv_vwaddu_vx_u16m8(union_popcount_u8m4, 0, vector_length);
        intersection_sum_u32m1 = __riscv_vwredsumu_vs_u16m8_u32m1(intersection_popcount_u16m8, intersection_sum_u32m1,
                                                                  vector_length);
        union_sum_u32m1 = __riscv_vwredsumu_vs_u16m8_u32m1(union_popcount_u16m8, union_sum_u32m1, vector_length);
    }

    nk_u32_t intersection_count_u32 = __riscv_vmv_x_s_u32m1_u32(intersection_sum_u32m1);
    nk_u32_t union_count_u32 = __riscv_vmv_x_s_u32m1_u32(union_sum_u32m1);
    *result = (union_count_u32 != 0) ? 1.0f - (nk_f32_t)intersection_count_u32 / (nk_f32_t)union_count_u32 : 0.0f;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_RVVBB
#endif // NK_TARGET_RISCV_
#endif // NK_SET_RVVBB_H
