/**
 *  @brief SIMD-accelerated Dot Products for RISC-V FP16.
 *  @file include/numkong/dot/rvvhalf.h
 *  @author Ash Vardanian
 *  @date January 5, 2026
 *
 *  @sa include/numkong/dot.h
 *
 *  SiFive P670/X280 and similar chips implement RVV 1.0 with Zvfh extension.
 *  Zvfh provides native half-precision (f16) vector operations.
 *  Uses widening multiply (f16 ⨯ f16 → f32) for precision, then reduces to f32.
 *
 *  For e2m3, e3m2, e4m3: conversion uses 256-entry VLUXEI16 LUT gathers from cast/rvv.h (3 instructions each).
 *  For e5m2: conversion uses pure shift (vzext + vsll) since e5m2 and f16 share the same exponent bias.
 *  All variants then use vfwmacc_vv for widening fused f16 ⨯ f16 → f32 multiply-accumulate.
 *
 *  Requires: RVV 1.0 + Zvfh extension (GCC 14+ or Clang 18+)
 */
#ifndef NK_DOT_RVVHALF_H
#define NK_DOT_RVVHALF_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVVHALF

#include "numkong/types.h"
#include "numkong/cast/rvv.h" // `nk_e4m3m1_to_f16m2_rvv_`, `nk_e2m3m1_to_f16m2_rvv_`, etc.

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v,+zvfh"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v,+zvfh")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_f16_rvvhalf(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((unsigned short const *)b_scalars, vector_length);
        vfloat16m1_t a_f16m1 = __riscv_vreinterpret_v_u16m1_f16m1(a_u16m1);
        vfloat16m1_t b_f16m1 = __riscv_vreinterpret_v_u16m1_f16m1(b_u16m1);
        // Widening FMA: f32 += f16 ⨯ f16, per-lane accumulation
        sum_f32m2 = __riscv_vfwmacc_vv_f32m2_tu(sum_f32m2, a_f16m1, b_f16m1, vector_length);
    }
    // Single horizontal reduction at the end
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(sum_f32m2, zero_f32m1, vlmax));
}

/** @brief Convert e2m3 to f16 via 256-entry LUT in cast/rvv.h + reinterpret. */
NK_INTERNAL vfloat16m2_t nk_e2m3m1_to_f16m2_rvvhalf_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    return __riscv_vreinterpret_v_u16m2_f16m2(nk_e2m3m1_to_f16m2_rvv_(raw_u8m1, vector_length));
}

/** @brief Convert e3m2 to f16 via 256-entry LUT in cast/rvv.h + reinterpret. */
NK_INTERNAL vfloat16m2_t nk_e3m2m1_to_f16m2_rvvhalf_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    return __riscv_vreinterpret_v_u16m2_f16m2(nk_e3m2m1_to_f16m2_rvv_(raw_u8m1, vector_length));
}

/** @brief Convert e4m3 to f16 via 256-entry LUT in cast/rvv.h + reinterpret. */
NK_INTERNAL vfloat16m2_t nk_e4m3m1_to_f16m2_rvvhalf_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    return __riscv_vreinterpret_v_u16m2_f16m2(nk_e4m3m1_to_f16m2_rvv_(raw_u8m1, vector_length));
}

/**
 *  @brief Convert e5m2 (1-5-2 sign-exp-mantissa, 8-bit) to f16 via pure shift (no LUT).
 *  Same exponent bias (15) means f16 = (lower7 << 8) | (sign << 15). Handles all cases.
 */
NK_INTERNAL vfloat16m2_t nk_e5m2m1_to_f16m2_rvvhalf_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    vuint16m2_t wide_u16m2 = __riscv_vzext_vf2_u16m2(raw_u8m1, vector_length);
    vuint16m2_t result_u16m2 = __riscv_vsll_vx_u16m2(wide_u16m2, 8, vector_length);
    return __riscv_vreinterpret_v_u16m2_f16m2(result_u16m2);
}

NK_PUBLIC void nk_dot_e4m3_rvvhalf(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat16m2_t a_f16m2 = nk_e4m3m1_to_f16m2_rvvhalf_(a_u8m1, vector_length);
        vfloat16m2_t b_f16m2 = nk_e4m3m1_to_f16m2_rvvhalf_(b_u8m1, vector_length);
        sum_f32m4 = __riscv_vfwmacc_vv_f32m4_tu(sum_f32m4, a_f16m2, b_f16m2, vector_length);
    }
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_dot_e5m2_rvvhalf(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat16m2_t a_f16m2 = nk_e5m2m1_to_f16m2_rvvhalf_(a_u8m1, vector_length);
        vfloat16m2_t b_f16m2 = nk_e5m2m1_to_f16m2_rvvhalf_(b_u8m1, vector_length);
        sum_f32m4 = __riscv_vfwmacc_vv_f32m4_tu(sum_f32m4, a_f16m2, b_f16m2, vector_length);
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

#endif // NK_TARGET_RVVHALF
#endif // NK_TARGET_RISCV_
#endif // NK_DOT_RVVHALF_H
