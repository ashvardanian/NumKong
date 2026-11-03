/**
 *  @file include/numkong/set/rvvbb.h
 *  @author Ash Vardanian
 *  @date February 9, 2026
 *  @brief SIMD-accelerated set similarity measures for RISC-V with Zvbb.
 *
 *  @sa c/numkong/set.h
 *
 *  Zvbb, the Vector Basic Bit-manipulation extension, adds a per-element popcount, <tt>vcpop.v</tt>, which
 *  replaces the 11-instruction SWAR sequence of set/rvv.h with one instruction.
 *
 *  Only @c nk_hamming_u1_rvvbb and @c nk_jaccard_u1_rvvbb need that byte-level popcount; the u8,
 *  u16 and u32 set kernels count with the mask-level <tt>vcpop.m</tt> that base RVV 1.0 already has.
 *
 *  Requires RVV 1.0 with Zvbb, from GCC 14 or Clang 18.
 */
#ifndef NUMKONG_SET_RVVBB_H
#define NUMKONG_SET_RVVBB_H

#if NUMKONG_ARCH_RISCV64_
#if NUMKONG_TARGET_RVVBB

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
 *  @brief Native per-element popcount using Zvbb vcpop.v (1 instruction).
 *
 *  Replaces the 11-instruction SWAR approach in nk_popcount_u8m4_rvv_.
 */
NUMKONG_INLINE vuint8m4_t nk_popcount_u8m4_rvvbb_(vuint8m4_t v_u8m4) {
    vuint8m4_t result_u8m4;
    __asm__ volatile("vcpop.v %0, %1" : "=vr"(result_u8m4) : "vr"(v_u8m4));
    return result_u8m4;
}

NUMKONG_API nk_status_t nk_hamming_u1_rvvbb(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                            nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t count_bytes = n / NUMKONG_BITS_PER_BYTE;

    vuint32m1_t sum_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    nk_size_t const max_vector_length = __riscv_vsetvlmax_e8m4();

    nk_size_t i = 0;
    while (i < count_bytes) {
        // U8 lanes hold 31 strips of popcounts, up to 8 per byte, before they must drain
        vuint8m4_t lanes_u8m4 = __riscv_vmv_v_x_u8m4(0, max_vector_length);
        for (unsigned strip = 0; strip != 31 && i < count_bytes; ++strip) {
            nk_size_t const vector_length = __riscv_vsetvl_e8m4(count_bytes - i);

            vuint8m4_t a_u8m4 = __riscv_vle8_v_u8m4(a + i, vector_length);
            vuint8m4_t b_u8m4 = __riscv_vle8_v_u8m4(b + i, vector_length);
            vuint8m4_t xor_u8m4 = __riscv_vxor_vv_u8m4(a_u8m4, b_u8m4, vector_length);

            // Native per-element popcount via Zvbb (1 instruction vs 11 SWAR)
            vuint8m4_t popcount_u8m4 = nk_popcount_u8m4_rvvbb_(xor_u8m4);

            lanes_u8m4 = __riscv_vadd_vv_u8m4_tu(lanes_u8m4, lanes_u8m4, popcount_u8m4, vector_length);
            i += vector_length;
        }
        vuint16m8_t lanes_u16m8 = __riscv_vwaddu_vx_u16m8(lanes_u8m4, 0, max_vector_length);
        sum_u32m1 = __riscv_vwredsumu_vs_u16m8_u32m1(lanes_u16m8, sum_u32m1, max_vector_length);
    }

    *result = __riscv_vmv_x_s_u32m1_u32(sum_u32m1);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_jaccard_u1_rvvbb(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t count_bytes = n / NUMKONG_BITS_PER_BYTE;

    vuint32m1_t intersection_sum_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    vuint32m1_t union_sum_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);
    nk_size_t const max_vector_length = __riscv_vsetvlmax_e8m4();

    nk_size_t i = 0;
    while (i < count_bytes) {
        // U8 lanes hold 31 strips of popcounts, up to 8 per byte, before they must drain
        vuint8m4_t intersection_lanes_u8m4 = __riscv_vmv_v_x_u8m4(0, max_vector_length);
        vuint8m4_t union_lanes_u8m4 = __riscv_vmv_v_x_u8m4(0, max_vector_length);
        for (unsigned strip = 0; strip != 31 && i < count_bytes; ++strip) {
            nk_size_t const vector_length = __riscv_vsetvl_e8m4(count_bytes - i);

            vuint8m4_t a_u8m4 = __riscv_vle8_v_u8m4(a + i, vector_length);
            vuint8m4_t b_u8m4 = __riscv_vle8_v_u8m4(b + i, vector_length);

            vuint8m4_t intersection_u8m4 = __riscv_vand_vv_u8m4(a_u8m4, b_u8m4, vector_length);
            vuint8m4_t union_u8m4 = __riscv_vor_vv_u8m4(a_u8m4, b_u8m4, vector_length);

            // Native per-element popcount via Zvbb
            vuint8m4_t intersection_popcount_u8m4 = nk_popcount_u8m4_rvvbb_(intersection_u8m4);
            vuint8m4_t union_popcount_u8m4 = nk_popcount_u8m4_rvvbb_(union_u8m4);

            intersection_lanes_u8m4 = __riscv_vadd_vv_u8m4_tu(intersection_lanes_u8m4, intersection_lanes_u8m4,
                                                              intersection_popcount_u8m4, vector_length);
            union_lanes_u8m4 = __riscv_vadd_vv_u8m4_tu(union_lanes_u8m4, union_lanes_u8m4, union_popcount_u8m4,
                                                       vector_length);
            i += vector_length;
        }
        vuint16m8_t intersection_lanes_u16m8 = __riscv_vwaddu_vx_u16m8(intersection_lanes_u8m4, 0, max_vector_length);
        vuint16m8_t union_lanes_u16m8 = __riscv_vwaddu_vx_u16m8(union_lanes_u8m4, 0, max_vector_length);
        intersection_sum_u32m1 = __riscv_vwredsumu_vs_u16m8_u32m1(intersection_lanes_u16m8, intersection_sum_u32m1,
                                                                  max_vector_length);
        union_sum_u32m1 = __riscv_vwredsumu_vs_u16m8_u32m1(union_lanes_u16m8, union_sum_u32m1, max_vector_length);
    }

    nk_u32_t intersection_count_u32 = __riscv_vmv_x_s_u32m1_u32(intersection_sum_u32m1);
    nk_u32_t union_count_u32 = __riscv_vmv_x_s_u32m1_u32(union_sum_u32m1);
    *result = (union_count_u32 != 0) ? 1.0f - (nk_f32_t)intersection_count_u32 / (nk_f32_t)union_count_u32 : 0.0f;
    return nk_success_k;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NUMKONG_TARGET_RVVBB
#endif // NUMKONG_ARCH_RISCV64_
#endif // NUMKONG_SET_RVVBB_H
