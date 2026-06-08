/**
 *  @brief SIMD-accelerated Set Similarity Measures for LoongArch LASX (256-bit).
 *  @file include/numkong/set/loongsonasx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/set.h
 *
 *  @section set_loongsonasx_instructions Key LASX Set Instructions
 *
 *      Intrinsic                    Instruction
 *      __lasx_xvld                  XVLD (256-bit unaligned load)
 *      __lasx_xvxor_v               XVXOR.V (bitwise XOR)
 *      __lasx_xvor_v                XVOR.V (bitwise OR)
 *      __lasx_xvand_v               XVAND.V (bitwise AND)
 *      __lasx_xvpcnt_d              XVPCNT.D (popcount per u64 element)
 *      __lasx_xvseq_b               XVSEQ.B (byte-wise equality, 0xFF/0x00)
 *      __lasx_xvmin_bu              XVMIN.BU (unsigned byte minimum)
 *      __lasx_xvhaddw_hu_bu         XVHADDW.HU.BU (horizontal pairwise add u8->u16)
 *      __lasx_xvhaddw_wu_hu         XVHADDW.WU.HU (horizontal pairwise add u16->u32)
 *      __lasx_xvhaddw_du_wu         XVHADDW.DU.WU (horizontal pairwise add u32->u64)
 *      __lasx_xvadd_d               XVADD.D (i64 addition)
 *      __lasx_xvpermi_q             XVPERMI.Q (extract/permute 128-bit lanes)
 *
 *  LASX provides per-element popcount at multiple widths (`xvpcnt_b/h/w/d`).
 *  For binary set operations we use `xvpcnt_d` which gives 4 x u64 popcount values
 *  directly, eliminating the need for horizontal byte-sum reduction chains.
 *
 *  For sorted integer set operations (jaccard_u16, jaccard_u32), SIMD provides limited
 *  benefit due to the inherently serial merge-based algorithm, so we delegate to the
 *  serial implementations.
 */
#ifndef NK_SET_LOONGSONASX_H
#define NK_SET_LOONGSONASX_H

#if NK_TARGET_LOONGARCH64_
#if NK_TARGET_LOONGSONASX

#include "numkong/types.h"
#include "numkong/set/serial.h"      // `nk_u1x8_popcount_`, serial fallbacks
#include "numkong/dot/loongsonasx.h" // `nk_reduce_add_i32x8_loongsonasx_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Reduction Helpers

/** @brief Horizontal sum of 4 u64 lanes in a 256-bit LASX register. */
NK_INTERNAL nk_u64_t nk_reduce_add_u64x4_loongsonasx_(__m256i sum_u64x4) {
    __m256i high_u64x4 = __lasx_xvpermi_q(sum_u64x4, sum_u64x4, 0x11);
    __m256i sum_u64x2 = __lasx_xvadd_d(sum_u64x4, high_u64x4);
    __m256i swapped_u64x2 = __lasx_xvshuf4i_d(sum_u64x2, sum_u64x2, 0b0001);
    __m256i reduced_u64x2 = __lasx_xvadd_d(sum_u64x2, swapped_u64x2);
    return (nk_u64_t)__lasx_xvpickve2gr_du(reduced_u64x2, 0);
}

/** @brief Horizontally sum all bytes in a 256-bit register as unsigned values.
 *
 *  Chains pairwise widening additions: u8→u16→u32→u64, then reduces 4 u64 lanes.
 */
NK_INTERNAL nk_u64_t nk_reduce_add_u8x32_loongsonasx_(__m256i v_u8x32) {
    __m256i sum_u16x16 = __lasx_xvhaddw_hu_bu(v_u8x32, v_u8x32);
    __m256i sum_u32x8 = __lasx_xvhaddw_wu_hu(sum_u16x16, sum_u16x16);
    __m256i sum_u64x4 = __lasx_xvhaddw_du_wu(sum_u32x8, sum_u32x8);
    return nk_reduce_add_u64x4_loongsonasx_(sum_u64x4);
}

#pragma endregion Reduction Helpers

#pragma region Binary Sets

NK_PUBLIC void nk_hamming_u1_loongsonasx(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);
    __m256i count_u64x4 = __lasx_xvreplgr2vr_d(0);
    nk_size_t i = 0;

    for (; i + 32 <= n_bytes; i += 32) {
        __m256i a_u8x32 = __lasx_xvld(a + i, 0);
        __m256i b_u8x32 = __lasx_xvld(b + i, 0);
        __m256i xor_u8x32 = __lasx_xvxor_v(a_u8x32, b_u8x32);
        count_u64x4 = __lasx_xvadd_d(count_u64x4, __lasx_xvpcnt_d(xor_u8x32));
    }

    nk_u64_t count = nk_reduce_add_u64x4_loongsonasx_(count_u64x4);

    for (; i < n_bytes; ++i) count += nk_u1x8_popcount_(a[i] ^ b[i]);
    *result = (nk_u32_t)count;
}

NK_PUBLIC void nk_jaccard_u1_loongsonasx(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);
    __m256i xor_count_u64x4 = __lasx_xvreplgr2vr_d(0);
    __m256i or_count_u64x4 = __lasx_xvreplgr2vr_d(0);
    nk_size_t i = 0;

    for (; i + 32 <= n_bytes; i += 32) {
        __m256i a_u8x32 = __lasx_xvld(a + i, 0);
        __m256i b_u8x32 = __lasx_xvld(b + i, 0);
        __m256i xor_u8x32 = __lasx_xvxor_v(a_u8x32, b_u8x32);
        __m256i or_u8x32 = __lasx_xvor_v(a_u8x32, b_u8x32);
        xor_count_u64x4 = __lasx_xvadd_d(xor_count_u64x4, __lasx_xvpcnt_d(xor_u8x32));
        or_count_u64x4 = __lasx_xvadd_d(or_count_u64x4, __lasx_xvpcnt_d(or_u8x32));
    }

    nk_u64_t xor_count = nk_reduce_add_u64x4_loongsonasx_(xor_count_u64x4);
    nk_u64_t or_count = nk_reduce_add_u64x4_loongsonasx_(or_count_u64x4);

    for (; i < n_bytes; ++i) {
        xor_count += nk_u1x8_popcount_(a[i] ^ b[i]);
        or_count += nk_u1x8_popcount_(a[i] | b[i]);
    }
    *result = (or_count != 0) ? (nk_f32_t)xor_count / (nk_f32_t)or_count : 0.0f;
}

#pragma endregion Binary Sets

#pragma region Integer Sets

NK_PUBLIC void nk_hamming_u8_loongsonasx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    __m256i count_u64x4 = __lasx_xvreplgr2vr_d(0);
    __m256i ones_u8x32 = __lasx_xvreplgr2vr_b(1);
    nk_size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m256i a_u8x32 = __lasx_xvld(a + i, 0);
        __m256i b_u8x32 = __lasx_xvld(b + i, 0);
        __m256i xor_u8x32 = __lasx_xvxor_v(a_u8x32, b_u8x32);
        __m256i min_u8x32 = __lasx_xvmin_bu(xor_u8x32, ones_u8x32);
        __m256i sum_u16x16 = __lasx_xvhaddw_hu_bu(min_u8x32, min_u8x32);
        __m256i sum_u32x8 = __lasx_xvhaddw_wu_hu(sum_u16x16, sum_u16x16);
        __m256i sum_u64x4 = __lasx_xvhaddw_du_wu(sum_u32x8, sum_u32x8);
        count_u64x4 = __lasx_xvadd_d(count_u64x4, sum_u64x4);
    }

    nk_u64_t count = nk_reduce_add_u64x4_loongsonasx_(count_u64x4);

    for (; i < n; ++i) count += (a[i] != b[i]);
    *result = (nk_u32_t)count;
}

#pragma endregion Integer Sets

#pragma region Batched Finalizers

/** @brief Hamming from_dot: computes pop_a + pop_b − 2 × dot for 4 pairs (LSX). */
NK_INTERNAL void nk_hamming_u32x4_from_dot_loongsonasx_(nk_b128_vec_t const *dots_vec, nk_u32_t query_pop,
                                                        nk_b128_vec_t const *target_pops_vec,
                                                        nk_b128_vec_t *result_vec) {
    __m128i dots_u32x4 = dots_vec->xmm;
    __m128i query_u32x4 = __lsx_vreplgr2vr_w((int)query_pop);
    __m128i target_u32x4 = target_pops_vec->xmm;
    result_vec->xmm = __lsx_vsub_w(__lsx_vadd_w(query_u32x4, target_u32x4), __lsx_vslli_w(dots_u32x4, 1));
}

/** @brief Jaccard from_dot: computes 1 − dot / (pop_a + pop_b − dot) for 4 pairs (LSX). */
NK_INTERNAL void nk_jaccard_f32x4_from_dot_loongsonasx_(nk_b128_vec_t const *dots_vec, nk_u32_t query_pop,
                                                        nk_b128_vec_t const *target_pops_vec,
                                                        nk_b128_vec_t *result_vec) {
    __m128 dot_f32x4 = __lsx_vffint_s_wu(dots_vec->xmm);
    __m128 query_f32x4 = nk_xvreplgr2vr_s_128_((nk_f32_t)query_pop);
    __m128 target_f32x4 = __lsx_vffint_s_wu(target_pops_vec->xmm);
    __m128 union_f32x4 = __lsx_vfsub_s(__lsx_vfadd_s(query_f32x4, target_f32x4), dot_f32x4);

    __m128 zero_f32x4 = (__m128)__lsx_vreplgr2vr_w(0);
    __m128 one_f32x4 = nk_xvreplgr2vr_s_128_(1.0f);
    __m128i zero_union_mask_u32x4 = __lsx_vfcmp_ceq_s(union_f32x4, zero_f32x4);
    __m128 safe_union_f32x4 = (__m128)__lsx_vbitsel_v((__m128i)union_f32x4, (__m128i)one_f32x4, zero_union_mask_u32x4);

    __m128 ratio_f32x4 = __lsx_vfdiv_s(dot_f32x4, safe_union_f32x4);
    __m128 jaccard_f32x4 = __lsx_vfsub_s(one_f32x4, ratio_f32x4);
    result_vec->xmm_ps = (__m128)__lsx_vbitsel_v((__m128i)jaccard_f32x4, (__m128i)zero_f32x4, zero_union_mask_u32x4);
}

#pragma endregion Batched Finalizers

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_LOONGSONASX
#endif // NK_TARGET_LOONGARCH64_
#endif // NK_SET_LOONGSONASX_H
