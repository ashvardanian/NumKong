/**
 *  @brief SIMD-accelerated Dot Products for RISC-V with Zvbb.
 *  @file include/numkong/dot/rvvbb.h
 *  @author Ash Vardanian
 *  @date February 22, 2026
 *
 *  @sa include/numkong/dot.h
 *
 *  Zvbb (Vector Basic Bit-manipulation) provides native per-element popcount via `vcpop.v`,
 *  replacing the 11-instruction SWAR approach with a single instruction for u1 dot products.
 *
 *  Only `nk_dot_u1` benefits from Zvbb (it needs byte-level popcount of AND results).
 *  Requires: RVV 1.0 + Zvbb extension (GCC 14+ or Clang 18+)
 */
#ifndef NK_DOT_RVVBB_H
#define NK_DOT_RVVBB_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVVBB

#include "numkong/types.h"
#include "numkong/set/rvvbb.h" // `nk_popcount_u8m4_rvvbb_`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v,+zvbb"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v,+zvbb")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_u1_rvvbb(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result) {
    nk_size_t count_bytes = nk_size_divide_round_up_(n_bits, NK_BITS_PER_BYTE);

    vuint32m1_t sum_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);

    nk_size_t i = 0;
    for (nk_size_t vector_length; i + 1 <= count_bytes; i += vector_length) {
        vector_length = __riscv_vsetvl_e8m4(count_bytes - i);

        // Load and AND to find shared bits (dot product of binary vectors)
        vuint8m4_t a_u8m4 = __riscv_vle8_v_u8m4(a + i, vector_length);
        vuint8m4_t b_u8m4 = __riscv_vle8_v_u8m4(b + i, vector_length);
        vuint8m4_t and_u8m4 = __riscv_vand_vv_u8m4(a_u8m4, b_u8m4, vector_length);

        // Native per-element popcount via Zvbb (1 instruction vs 11 SWAR)
        vuint8m4_t popcount_u8m4 = nk_popcount_u8m4_rvvbb_(and_u8m4);

        // Widen to u16 and accumulate via widening reduction sum
        vuint16m8_t popcount_u16m8 = __riscv_vwaddu_vx_u16m8(popcount_u8m4, 0, vector_length);
        sum_u32m1 = __riscv_vwredsumu_vs_u16m8_u32m1(popcount_u16m8, sum_u32m1, vector_length);
    }

    *result = __riscv_vmv_x_s_u32m1_u32(sum_u32m1);
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
#endif // NK_DOT_RVVBB_H
