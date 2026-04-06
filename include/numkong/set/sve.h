/**
 *  @brief SIMD-accelerated Set Similarity Measures for SVE.
 *  @file include/numkong/set/sve.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/set.h
 *
 *  @section set_sve_instructions ARM SVE Instructions
 *
 *      Intrinsic      Instruction                 V1
 *      svld1_u8       LD1B (Z.B, P/Z, [Xn])       4-6cy @ 2p
 *      svld1_u32      LD1W (Z.S, P/Z, [Xn])       4-6cy @ 2p
 *      sveor_u8_m     EOR (Z.B, P/M, Z.B, Z.B)    1cy @ 2p
 *      svand_u8_m     AND (Z.B, P/M, Z.B, Z.B)    1cy @ 2p
 *      svorr_u8_m     ORR (Z.B, P/M, Z.B, Z.B)    1cy @ 2p
 *      svcnt_u8_x     CNT (Z.B, P/M, Z.B)         2cy @ 2p
 *      svadd_u8_z     ADD (Z.B, P/M, Z.B, Z.B)    1cy @ 2p
 *      svaddv_u8      UADDV (D, P, Z.B)           6cy @ 1p
 *      svcmpeq_u32    CMPEQ (P.S, P/Z, Z.S, Z.S)  2cy @ 1p
 *      svcntp_b32     CNTP (Xd, P, P.S)           2cy @ 1p
 *      svdup_n_u8     DUP (Z.B, #imm)             1cy @ 2p
 *      svwhilelt_b8   WHILELT (P.B, Xn, Xm)       2cy @ 1p
 *      svwhilelt_b32  WHILELT (P.S, Xn, Xm)       2cy @ 1p
 *      svptrue_b8     PTRUE (P.B, pattern)        1cy @ 2p
 *      svcntb         CNTB (Xd)                   1cy @ 2p
 *      svcntw         CNTW (Xd)                   1cy @ 2p
 */
#ifndef NK_SET_SVE_H
#define NK_SET_SVE_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SVE

#include "numkong/types.h"    // `nk_u1x8_t`
#include "numkong/set/neon.h" // `nk_hamming_u1_neon`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

#pragma region Binary Sets

NK_PUBLIC void nk_hamming_u1_sve(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);

    // On very small register sizes, NEON is at least as fast as SVE.
    nk_size_t const words_per_register = svcntb();
    if (words_per_register <= 32) {
        nk_hamming_u1_neon(a, b, n, result);
        return;
    }

    // On larger register sizes, SVE is faster.
    nk_size_t i = 0, cycle = 0;
    nk_u32_t differences = 0;
    svuint8_t popcount_u8x = svdup_n_u8(0);
    svbool_t const all_predicate_b8x = svptrue_b8();
    while (i < n_bytes) {
        do {
            svbool_t active_predicate_b8x = svwhilelt_b8_u64(i, n_bytes);
            svuint8_t a_u8x = svld1_u8(active_predicate_b8x, a + i);
            svuint8_t b_u8x = svld1_u8(active_predicate_b8x, b + i);
            popcount_u8x = svadd_u8_z(all_predicate_b8x, popcount_u8x,
                                      svcnt_u8_x(all_predicate_b8x, sveor_u8_m(all_predicate_b8x, a_u8x, b_u8x)));
            i += words_per_register;
            ++cycle;
        } while (i < n_bytes && cycle < 31);
        differences += svaddv_u8(all_predicate_b8x, popcount_u8x);
        NK_UNPOISON(&differences, sizeof(differences));
        popcount_u8x = svdup_n_u8(0);
        cycle = 0; // Reset the cycle counter.
    }

    *result = differences;
}

NK_PUBLIC void nk_jaccard_u1_sve(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, NK_BITS_PER_BYTE);

    // On very small register sizes, NEON is at least as fast as SVE.
    nk_size_t const words_per_register = svcntb();
    if (words_per_register <= 32) {
        nk_jaccard_u1_neon(a, b, n, result);
        return;
    }

    // On larger register sizes, SVE is faster.
    nk_size_t i = 0, cycle = 0;
    nk_u32_t intersection_count = 0, union_count = 0;
    svuint8_t intersection_popcount_u8x = svdup_n_u8(0);
    svuint8_t union_popcount_u8x = svdup_n_u8(0);
    svbool_t const all_predicate_b8x = svptrue_b8();
    while (i < n_bytes) {
        do {
            svbool_t active_predicate_b8x = svwhilelt_b8_u64(i, n_bytes);
            svuint8_t a_u8x = svld1_u8(active_predicate_b8x, a + i);
            svuint8_t b_u8x = svld1_u8(active_predicate_b8x, b + i);
            intersection_popcount_u8x = svadd_u8_z(
                all_predicate_b8x, intersection_popcount_u8x,
                svcnt_u8_x(all_predicate_b8x, svand_u8_m(all_predicate_b8x, a_u8x, b_u8x)));
            union_popcount_u8x = svadd_u8_z(all_predicate_b8x, union_popcount_u8x,
                                            svcnt_u8_x(all_predicate_b8x, svorr_u8_m(all_predicate_b8x, a_u8x, b_u8x)));
            i += words_per_register;
            ++cycle;
        } while (i < n_bytes && cycle < 31);
        intersection_count += svaddv_u8(all_predicate_b8x, intersection_popcount_u8x);
        NK_UNPOISON(&intersection_count, sizeof(intersection_count));
        intersection_popcount_u8x = svdup_n_u8(0);
        union_count += svaddv_u8(all_predicate_b8x, union_popcount_u8x);
        NK_UNPOISON(&union_count, sizeof(union_count));
        union_popcount_u8x = svdup_n_u8(0);
        cycle = 0; // Reset the cycle counter.
    }

    *result = (union_count != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)union_count : 0.0f;
}

#pragma endregion Binary Sets

#pragma region Integer Sets

NK_PUBLIC void nk_jaccard_u32_sve(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t const words_per_register = svcntw();
    nk_size_t i = 0;
    nk_u32_t intersection_count = 0;
    while (i < n) {
        svbool_t active_predicate_b32x = svwhilelt_b32_u64(i, n);
        svuint32_t a_u32x = svld1_u32(active_predicate_b32x, a + i);
        svuint32_t b_u32x = svld1_u32(active_predicate_b32x, b + i);
        svbool_t equality_predicate_b32x = svcmpeq_u32(active_predicate_b32x, a_u32x, b_u32x);
        intersection_count += svcntp_b32(active_predicate_b32x, equality_predicate_b32x);
        i += words_per_register;
    }
    *result = (n != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)n : 0.0f;
}

NK_PUBLIC void nk_hamming_u8_sve(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_size_t const bytes_per_register = svcntb();
    nk_size_t i = 0;
    nk_u32_t differences = 0;
    while (i < n) {
        svbool_t active_predicate_b8x = svwhilelt_b8_u64(i, n);
        svuint8_t a_u8x = svld1_u8(active_predicate_b8x, a + i);
        svuint8_t b_u8x = svld1_u8(active_predicate_b8x, b + i);
        svbool_t neq_predicate_b8x = svcmpne_u8(active_predicate_b8x, a_u8x, b_u8x);
        differences += svcntp_b8(active_predicate_b8x, neq_predicate_b8x);
        i += bytes_per_register;
    }
    *result = differences;
}

NK_PUBLIC void nk_jaccard_u16_sve(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t const halfwords_per_register = svcnth();
    nk_size_t i = 0;
    nk_u32_t intersection_count = 0;
    while (i < n) {
        svbool_t active_predicate_b16x = svwhilelt_b16_u64(i, n);
        svuint16_t a_u16x = svld1_u16(active_predicate_b16x, a + i);
        svuint16_t b_u16x = svld1_u16(active_predicate_b16x, b + i);
        svbool_t equality_predicate_b16x = svcmpeq_u16(active_predicate_b16x, a_u16x, b_u16x);
        intersection_count += svcntp_b16(active_predicate_b16x, equality_predicate_b16x);
        i += halfwords_per_register;
    }
    *result = (n != 0) ? 1.0f - (nk_f32_t)intersection_count / (nk_f32_t)n : 0.0f;
}

#pragma endregion Integer Sets

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SVE
#endif // NK_TARGET_ARM64_
#endif // NK_SET_SVE_H
