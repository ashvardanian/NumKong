/**
 *  @brief SIMD-accelerated Dot Products for RISC-V BF16.
 *  @file include/numkong/dot/rvvbf16.h
 *  @author Ash Vardanian
 *  @date January 5, 2026
 *
 *  @sa include/numkong/dot.h
 *
 *  Alibaba XuanTie C930 and similar chips implement RVV 1.0 with Zvfbfwma extension.
 *  Zvfbfwma provides widening bf16 fused multiply-accumulate to f32:
 *    vfwmaccbf16: f32 ← bf16 ⨯ bf16
 *
 *  All mini-float types use 256-entry VLUXEI16 LUT gathers from cast/rvv.h (3 instructions each).
 *  All variants then use vfwmaccbf16_vv for fused bf16 ⨯ bf16 → f32 multiply-accumulate.
 *
 *  Requires: RVV 1.0 + Zvfbfwma extension (GCC 14+ or Clang 18+)
 */
#ifndef NK_DOT_RVVBF16_H
#define NK_DOT_RVVBF16_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVVBF16

#include "numkong/types.h"
#include "numkong/cast/rvv.h" // `nk_e4m3m1_to_bf16m2_rvv_`, `nk_e5m2m1_to_bf16m2_rvv_`, etc.

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v,+zvfbfwma"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v,+zvfbfwma")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_bf16_rvvbf16(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)b_scalars, vector_length);
        vbfloat16m1_t a_bf16m1 = __riscv_vreinterpret_v_u16m1_bf16m1(a_u16m1);
        vbfloat16m1_t b_bf16m1 = __riscv_vreinterpret_v_u16m1_bf16m1(b_u16m1);
        // Widening bf16 FMA: f32 ← bf16 ⨯ bf16, per-lane accumulation
        sum_f32m2 = __riscv_vfwmaccbf16_vv_f32m2_tu(sum_f32m2, a_bf16m1, b_bf16m1, vector_length);
    }
    // Single horizontal reduction at the end
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(sum_f32m2, zero_f32m1, vlmax));
}

/** @brief Convert e2m3 to bf16 via 256-entry LUT in cast/rvv.h + reinterpret. */
NK_INTERNAL vbfloat16m2_t nk_e2m3m1_to_bf16m2_rvvbf16_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    return __riscv_vreinterpret_v_u16m2_bf16m2(nk_e2m3m1_to_bf16m2_rvv_(raw_u8m1, vector_length));
}

/** @brief Convert e3m2 to bf16 via 256-entry LUT in cast/rvv.h + reinterpret. */
NK_INTERNAL vbfloat16m2_t nk_e3m2m1_to_bf16m2_rvvbf16_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    return __riscv_vreinterpret_v_u16m2_bf16m2(nk_e3m2m1_to_bf16m2_rvv_(raw_u8m1, vector_length));
}

/** @brief Convert e4m3 to bf16 via 256-entry LUT in cast/rvv.h + reinterpret. */
NK_INTERNAL vbfloat16m2_t nk_e4m3m1_to_bf16m2_rvvbf16_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    return __riscv_vreinterpret_v_u16m2_bf16m2(nk_e4m3m1_to_bf16m2_rvv_(raw_u8m1, vector_length));
}

/** @brief Convert e5m2 to bf16 via 256-entry LUT in cast/rvv.h + reinterpret. */
NK_INTERNAL vbfloat16m2_t nk_e5m2m1_to_bf16m2_rvvbf16_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    return __riscv_vreinterpret_v_u16m2_bf16m2(nk_e5m2m1_to_bf16m2_rvv_(raw_u8m1, vector_length));
}

NK_PUBLIC void nk_dot_e4m3_rvvbf16(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vbfloat16m2_t a_bf16m2 = nk_e4m3m1_to_bf16m2_rvvbf16_(a_u8m1, vector_length);
        vbfloat16m2_t b_bf16m2 = nk_e4m3m1_to_bf16m2_rvvbf16_(b_u8m1, vector_length);
        sum_f32m4 = __riscv_vfwmaccbf16_vv_f32m4_tu(sum_f32m4, a_bf16m2, b_bf16m2, vector_length);
    }
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_dot_e5m2_rvvbf16(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vbfloat16m2_t a_bf16m2 = nk_e5m2m1_to_bf16m2_rvvbf16_(a_u8m1, vector_length);
        vbfloat16m2_t b_bf16m2 = nk_e5m2m1_to_bf16m2_rvvbf16_(b_u8m1, vector_length);
        sum_f32m4 = __riscv_vfwmaccbf16_vv_f32m4_tu(sum_f32m4, a_bf16m2, b_bf16m2, vector_length);
    }
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax));
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_RVVBF16
#endif // NK_TARGET_RISCV_
#endif // NK_DOT_RVVBF16_H
