/**
 *  @brief SIMD-accelerated Set Similarity Measures for RISC-V.
 *  @file include/numkong/set/rvv.h
 *  @author Ash Vardanian
 *  @date January 13, 2026
 *
 *  @sa include/numkong/set.h
 *
 *  SpacemiT K1 and similar chips implement RVA22 profile with base RVV 1.0.
 *  This does NOT include the Zvbb extension, so we lack native element-wise popcount (`vcpop.v`).
 *
 *  @section rvv_popcount_lut Popcount via vrgather LUT
 *
 *  We implement popcount using a 16-entry nibble lookup table with `vrgather`:
 *  - Split each byte into high and low nibbles
 *  - Use vrgather to look up popcount of each nibble (0-4)
 *  - Sum the results (0-8 per byte)
 *
 *  This approach is efficient on SpacemiT X60 cores which have optimized vrgather
 *  for small indices (LMUL=1 with indices 0-15).
 *
 *  @section set_rvv_instructions Key RVV Set Instructions
 *
 *      Intrinsic                       Purpose
 *      vxor_vv_u8m1                    XOR for Hamming difference
 *      vand_vv_u8m1                    AND for Jaccard intersection
 *      vor_vv_u8m1                     OR for Jaccard union
 *      vrgather_vv_u8m1                LUT lookup (16-entry nibble table)
 *      vsrl_vx_u8m1                    Right shift to extract high nibble
 *      vwaddu_vx_u16m2                 Widen u8 → u16 for accumulation
 *      vwredsumu_vs_u16m2_u32m1        Widening reduction sum
 */
#ifndef NK_SET_RVV_H
#define NK_SET_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/set/serial.h" // `nk_u1x8_popcount_`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region - Binary Sets

/**
 *  @brief  Compute byte-level popcount using arithmetic SWAR.
 *
 *  Uses parallel bit counting (Hamming weight) — no vrgather, so scales
 *  linearly with LMUL unlike the nibble-LUT approach.
 *  Cost: 11 ALU instructions regardless of LMUL.
 *
 *  @param[in] v_u8m4 Input vector of bytes
 *  @param[in] vector_length Vector length
 *  @return Vector where each byte contains its popcount (0-8)
 */
NK_INTERNAL vuint8m4_t nk_popcount_u8m4_rvv_(vuint8m4_t v_u8m4, nk_size_t vector_length) {
    // Step 1: count pairs — v = (v & 0x55) + ((v >> 1) & 0x55)
    vuint8m4_t t_u8m4 = __riscv_vsrl_vx_u8m4(v_u8m4, 1, vector_length);
    t_u8m4 = __riscv_vand_vx_u8m4(t_u8m4, 0x55, vector_length);
    v_u8m4 = __riscv_vand_vx_u8m4(v_u8m4, 0x55, vector_length);
    v_u8m4 = __riscv_vadd_vv_u8m4(v_u8m4, t_u8m4, vector_length);
    // Step 2: count nibbles — v = (v & 0x33) + ((v >> 2) & 0x33)
    t_u8m4 = __riscv_vsrl_vx_u8m4(v_u8m4, 2, vector_length);
    t_u8m4 = __riscv_vand_vx_u8m4(t_u8m4, 0x33, vector_length);
    v_u8m4 = __riscv_vand_vx_u8m4(v_u8m4, 0x33, vector_length);
    v_u8m4 = __riscv_vadd_vv_u8m4(v_u8m4, t_u8m4, vector_length);
    // Step 3: count bytes — v = (v + (v >> 4)) & 0x0F
    t_u8m4 = __riscv_vsrl_vx_u8m4(v_u8m4, 4, vector_length);
    v_u8m4 = __riscv_vadd_vv_u8m4(v_u8m4, t_u8m4, vector_length);
    return __riscv_vand_vx_u8m4(v_u8m4, 0x0F, vector_length);
}

NK_PUBLIC void nk_hamming_u1_rvv(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_size_t count_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);

    // Accumulator for total differences
    vuint32m1_t sum_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);

    nk_size_t i = 0;
    for (nk_size_t vector_length; i + 1 <= count_bytes; i += vector_length) {
        vector_length = __riscv_vsetvl_e8m4(count_bytes - i);

        // Load and XOR to find differing bits
        vuint8m4_t a_u8m4 = __riscv_vle8_v_u8m4(a + i, vector_length);
        vuint8m4_t b_u8m4 = __riscv_vle8_v_u8m4(b + i, vector_length);
        vuint8m4_t xor_u8m4 = __riscv_vxor_vv_u8m4(a_u8m4, b_u8m4, vector_length);

        // Popcount each byte (0-8 per byte) using arithmetic SWAR
        vuint8m4_t popcount_u8m4 = nk_popcount_u8m4_rvv_(xor_u8m4, vector_length);

        // Widen to u16 and accumulate via widening reduction sum
        vuint16m8_t popcount_u16m8 = __riscv_vwaddu_vx_u16m8(popcount_u8m4, 0, vector_length);
        sum_u32m1 = __riscv_vwredsumu_vs_u16m8_u32m1(popcount_u16m8, sum_u32m1, vector_length);
    }

    *result = __riscv_vmv_x_s_u32m1_u32(sum_u32m1);
}

NK_PUBLIC void nk_jaccard_u1_rvv(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t count_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);

    // Accumulators for intersection and union counts
    vuint32m1_t intersection_sum_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    vuint32m1_t union_sum_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);

    nk_size_t i = 0;
    for (nk_size_t vector_length; i + 1 <= count_bytes; i += vector_length) {
        vector_length = __riscv_vsetvl_e8m4(count_bytes - i);

        // Load vectors
        vuint8m4_t a_u8m4 = __riscv_vle8_v_u8m4(a + i, vector_length);
        vuint8m4_t b_u8m4 = __riscv_vle8_v_u8m4(b + i, vector_length);

        // Compute intersection (AND) and union (OR)
        vuint8m4_t intersection_u8m4 = __riscv_vand_vv_u8m4(a_u8m4, b_u8m4, vector_length);
        vuint8m4_t union_u8m4 = __riscv_vor_vv_u8m4(a_u8m4, b_u8m4, vector_length);

        // Popcount each using arithmetic SWAR
        vuint8m4_t intersection_popcount_u8m4 = nk_popcount_u8m4_rvv_(intersection_u8m4, vector_length);
        vuint8m4_t union_popcount_u8m4 = nk_popcount_u8m4_rvv_(union_u8m4, vector_length);

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

#pragma endregion - Binary Sets

#pragma region - Integer Sets

NK_PUBLIC void nk_hamming_u8_rvv(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    vuint32m1_t difference_count_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);

    nk_size_t i = 0;
    for (nk_size_t vector_length; i + 1 <= n; i += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n - i);

        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1(a + i, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1(b + i, vector_length);

        // Compare: mask where a != b
        vbool8_t not_equal_mask_b8 = __riscv_vmsne_vv_u8m1_b8(a_u8m1, b_u8m1, vector_length);

        // Count set bits in mask via vcpop.m (this IS available in base RVV 1.0)
        nk_u32_t difference_count_u32 = __riscv_vcpop_m_b8(not_equal_mask_b8, vector_length);

        // Accumulate (scalar addition is fine here, vcpop already reduced)
        difference_count_u32m1 = __riscv_vadd_vx_u32m1(difference_count_u32m1, difference_count_u32, 1);
    }

    *result = __riscv_vmv_x_s_u32m1_u32(difference_count_u32m1);
}

NK_PUBLIC void nk_jaccard_u32_rvv(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t match_count_u32 = 0;

    nk_size_t i = 0;
    for (nk_size_t vector_length; i + 1 <= n; i += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(n - i);

        vuint32m1_t a_u32m1 = __riscv_vle32_v_u32m1(a + i, vector_length);
        vuint32m1_t b_u32m1 = __riscv_vle32_v_u32m1(b + i, vector_length);

        // Compare: mask where a == b
        vbool32_t equal_mask_b32 = __riscv_vmseq_vv_u32m1_b32(a_u32m1, b_u32m1, vector_length);

        // Count matches via vcpop.m
        match_count_u32 += __riscv_vcpop_m_b32(equal_mask_b32, vector_length);
    }

    *result = (n != 0) ? 1.0f - (nk_f32_t)match_count_u32 / (nk_f32_t)n : 0.0f;
}

NK_PUBLIC void nk_jaccard_u16_rvv(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t match_count_u32 = 0;

    nk_size_t i = 0;
    for (nk_size_t vector_length; i + 1 <= n; i += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n - i);

        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1(a + i, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1(b + i, vector_length);

        // Compare: mask where a == b
        vbool16_t equal_mask_b16 = __riscv_vmseq_vv_u16m1_b16(a_u16m1, b_u16m1, vector_length);

        // Count matches via vcpop.m
        match_count_u32 += __riscv_vcpop_m_b16(equal_mask_b16, vector_length);
    }

    *result = (n != 0) ? 1.0f - (nk_f32_t)match_count_u32 / (nk_f32_t)n : 0.0f;
}

#pragma endregion - Integer Sets

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_RVV
#endif // NK_TARGET_RISCV_
#endif // NK_SET_RVV_H
