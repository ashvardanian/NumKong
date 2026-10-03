/**
 *  @brief SIMD-accelerated Dot Products for RISC-V.
 *  @file include/numkong/dot/rvv.h
 *  @author Ash Vardanian
 *  @date January 5, 2026
 *
 *  @sa include/numkong/dot.h
 *
 *  SpacemiT K1 and similar chips implement base RVV 1.0 without half-precision extensions.
 *  RVV uses vector length agnostic programming where:
 *  - `vsetvl_e*m*(n)` sets VL = min(n, VLMAX) and returns actual VL
 *  - Loads/stores with VL automatically handle partial vectors (tail elements)
 *  - No explicit masking needed for simple reductions
 *
 *  This file contains base RVV 1.0 operations (i8, u8, f32, f64).
 *  For f16 (Zvfh) see rvvhalf.h, for bf16 (Zvfbfwma) see rvvbf16.h.
 *
 *  Widening operations:
 *  - i8 ⨯ i8 → i16 via vwmul, then i16 reduction → i32 via vwredsum
 *  - f32 ⨯ f32 → f64 via vfwmul (for precision, like Skylake)
 */
#ifndef NK_DOT_RVV_H
#define NK_DOT_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/cast/rvv.h" // `nk_e4m3m1_to_f32m4_rvv_`
#include "numkong/set/rvv.h"  // `nk_popcount_u8m4_rvv_`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Compensated horizontal sum of RVV f64m1 lanes via TwoSum tree reduction.
 *
 *  Uses vslidedown to extract the upper half at each tree level (same pattern as
 *  nk_reduce_vsaddu_u64m1_rvv_ in reduce/rvv.h). Tail lanes beyond vlmax are zero
 *  from the initial vfmv_v_f, so they are harmless in the reduction.
 */
NK_INTERNAL nk_f64_t nk_dot_stable_sum_f64m1_rvv_(vfloat64m1_t sum_f64m1, vfloat64m1_t compensation_f64m1) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    // Stage 0: TwoSum merge of sum + compensation
    vfloat64m1_t tentative_sum_f64m1 = __riscv_vfadd_vv_f64m1(sum_f64m1, compensation_f64m1, vlmax);
    vfloat64m1_t virtual_addend_f64m1 = __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, sum_f64m1, vlmax);
    vfloat64m1_t accumulated_error_f64m1 = __riscv_vfadd_vv_f64m1(
        __riscv_vfsub_vv_f64m1(sum_f64m1, __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, virtual_addend_f64m1, vlmax),
                               vlmax),
        __riscv_vfsub_vv_f64m1(compensation_f64m1, virtual_addend_f64m1, vlmax), vlmax);
    // Tree reduction: TwoSum halving at each level
    for (nk_size_t half = vlmax / 2; half > 0; half >>= 1) {
        vfloat64m1_t upper_sum_f64m1 = __riscv_vslidedown_vx_f64m1(tentative_sum_f64m1, half, vlmax);
        vfloat64m1_t upper_error_f64m1 = __riscv_vslidedown_vx_f64m1(accumulated_error_f64m1, half, vlmax);
        vfloat64m1_t halved_tentative_sum_f64m1 = __riscv_vfadd_vv_f64m1(tentative_sum_f64m1, upper_sum_f64m1, vlmax);
        vfloat64m1_t halved_virtual_addend_f64m1 = __riscv_vfsub_vv_f64m1(halved_tentative_sum_f64m1,
                                                                          tentative_sum_f64m1, vlmax);
        vfloat64m1_t rounding_error_f64m1 = __riscv_vfadd_vv_f64m1(
            __riscv_vfsub_vv_f64m1(
                tentative_sum_f64m1,
                __riscv_vfsub_vv_f64m1(halved_tentative_sum_f64m1, halved_virtual_addend_f64m1, vlmax), vlmax),
            __riscv_vfsub_vv_f64m1(upper_sum_f64m1, halved_virtual_addend_f64m1, vlmax), vlmax);
        tentative_sum_f64m1 = halved_tentative_sum_f64m1;
        accumulated_error_f64m1 = __riscv_vfadd_vv_f64m1(
            __riscv_vfadd_vv_f64m1(accumulated_error_f64m1, upper_error_f64m1, vlmax), rounding_error_f64m1, vlmax);
    }
    return __riscv_vfmv_f_s_f64m1_f64(tentative_sum_f64m1) + __riscv_vfmv_f_s_f64m1_f64(accumulated_error_f64m1);
}

NK_PUBLIC void nk_dot_i8_rvv(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                             nk_i32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vint32m4_t sum_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vint8m1_t a_i8m1 = __riscv_vle8_v_i8m1(a_scalars, vector_length);
        vint8m1_t b_i8m1 = __riscv_vle8_v_i8m1(b_scalars, vector_length);
        // Widening multiply: i8 ⨯ i8 → i16
        vint16m2_t ab_i16m2 = __riscv_vwmul_vv_i16m2(a_i8m1, b_i8m1, vector_length);
        // Per-lane widening accumulate: i32 += i16
        sum_i32m4 = __riscv_vwadd_wv_i32m4_tu(sum_i32m4, sum_i32m4, ab_i16m2, vector_length);
    }
    // Single horizontal reduction at the end
    vint32m1_t zero_i32m1 = __riscv_vmv_v_x_i32m1(0, vlmax);
    *result = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(sum_i32m4, zero_i32m1, vlmax));
}

NK_PUBLIC void nk_dot_u8_rvv(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                             nk_u32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vuint32m4_t sum_u32m4 = __riscv_vmv_v_x_u32m4(0, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1(a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1(b_scalars, vector_length);
        // Widening multiply: u8 ⨯ u8 → u16
        vuint16m2_t ab_u16m2 = __riscv_vwmulu_vv_u16m2(a_u8m1, b_u8m1, vector_length);
        // Per-lane widening accumulate: u32 += u16
        sum_u32m4 = __riscv_vwaddu_wv_u32m4_tu(sum_u32m4, sum_u32m4, ab_u16m2, vector_length);
    }
    // Single horizontal reduction at the end
    vuint32m1_t zero_u32m1 = __riscv_vmv_v_x_u32m1(0, vlmax);
    *result = __riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(sum_u32m4, zero_u32m1, vlmax));
}

NK_PUBLIC void nk_dot_f32_rvv(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                              nk_f64_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count_scalars);
        vfloat32m1_t a_f32m1 = __riscv_vle32_v_f32m1(a_scalars, vector_length);
        vfloat32m1_t b_f32m1 = __riscv_vle32_v_f32m1(b_scalars, vector_length);
        // Widening FMA: f64 += f32 ⨯ f32, per-lane accumulation
        sum_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(sum_f64m2, a_f32m1, b_f32m1, vector_length);
    }
    // Single horizontal reduction at the end
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    *result = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_f64m2, zero_f64m1, vlmax));
}

NK_PUBLIC void nk_dot_f64_rvv(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                              nk_f64_t *result) {
    // Dot2 (Ogita-Rump-Oishi) compensated accumulation via TwoProd + TwoSum
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(count_scalars);
        vfloat64m1_t a_f64m1 = __riscv_vle64_v_f64m1(a_scalars, vector_length);
        vfloat64m1_t b_f64m1 = __riscv_vle64_v_f64m1(b_scalars, vector_length);
        // TwoProd: product = a*b, product_error = fma(a,b,-product)
        vfloat64m1_t product_f64m1 = __riscv_vfmul_vv_f64m1(a_f64m1, b_f64m1, vector_length);
        vfloat64m1_t product_error_f64m1 = __riscv_vfmsac_vv_f64m1(product_f64m1, a_f64m1, b_f64m1, vector_length);
        // TwoSum: tentative_sum = sum + product
        vfloat64m1_t tentative_sum_f64m1 = __riscv_vfadd_vv_f64m1(sum_f64m1, product_f64m1, vector_length);
        vfloat64m1_t virtual_addend_f64m1 = __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, sum_f64m1, vector_length);
        vfloat64m1_t sum_error_f64m1 = __riscv_vfadd_vv_f64m1(
            __riscv_vfsub_vv_f64m1(sum_f64m1,
                                   __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, virtual_addend_f64m1, vector_length),
                                   vector_length),
            __riscv_vfsub_vv_f64m1(product_f64m1, virtual_addend_f64m1, vector_length), vector_length);
        // Tail-undisturbed updates: preserve zero tails across partial iterations
        sum_f64m1 = __riscv_vslideup_vx_f64m1_tu(sum_f64m1, tentative_sum_f64m1, 0, vector_length);
        vfloat64m1_t total_error_f64m1 = __riscv_vfadd_vv_f64m1(sum_error_f64m1, product_error_f64m1, vector_length);
        compensation_f64m1 = __riscv_vfadd_vv_f64m1_tu(compensation_f64m1, compensation_f64m1, total_error_f64m1,
                                                       vector_length);
    }
    // Compensated horizontal reduction
    *result = nk_dot_stable_sum_f64m1_rvv_(sum_f64m1, compensation_f64m1);
}

NK_PUBLIC void nk_dot_f16_rvv(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                              nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);

        // Load f16 as u16 bits and convert to f32 via helper
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b_scalars, vector_length);
        vfloat32m2_t a_f32m2 = nk_f16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_f16m1_to_f32m2_rvv_(b_u16m1, vector_length);

        // Per-lane FMA accumulation
        sum_f32m2 = __riscv_vfmacc_vv_f32m2_tu(sum_f32m2, a_f32m2, b_f32m2, vector_length);
    }
    // Single horizontal reduction at the end
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(sum_f32m2, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_dot_bf16_rvv(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                               nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count_scalars);

        // Load bf16 as u16 and convert to f32 via helper
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a_scalars, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b_scalars, vector_length);
        vfloat32m2_t a_f32m2 = nk_bf16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_bf16m1_to_f32m2_rvv_(b_u16m1, vector_length);

        // Per-lane FMA accumulation
        sum_f32m2 = __riscv_vfmacc_vv_f32m2_tu(sum_f32m2, a_f32m2, b_f32m2, vector_length);
    }
    // Single horizontal reduction at the end
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(sum_f32m2, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_dot_e4m3_rvv(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                               nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);

        // Load e4m3 as u8 and convert to f32 via helper
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat32m4_t a_f32m4 = nk_e4m3m1_to_f32m4_rvv_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e4m3m1_to_f32m4_rvv_(b_u8m1, vector_length);

        // Per-lane FMA accumulation
        sum_f32m4 = __riscv_vfmacc_vv_f32m4_tu(sum_f32m4, a_f32m4, b_f32m4, vector_length);
    }
    // Single horizontal reduction at the end
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_dot_e5m2_rvv(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                               nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);

        // Load e5m2 as u8 and convert to f32 via helper
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);
        vfloat32m4_t a_f32m4 = nk_e5m2m1_to_f32m4_rvv_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e5m2m1_to_f32m4_rvv_(b_u8m1, vector_length);

        // Per-lane FMA accumulation
        sum_f32m4 = __riscv_vfmacc_vv_f32m4_tu(sum_f32m4, a_f32m4, b_f32m4, vector_length);
    }
    // Single horizontal reduction at the end
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_dot_e2m3_rvv(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars, nk_size_t count_scalars,
                               nk_f32_t *result) {
    // Integer dot product for e2m3 using byte gather LUT + widening multiply.
    // Every e2m3 value × 16 is an exact integer in [-120, +120].
    // Result = i32_dot / 256.0f (exact, no rounding error).
    static nk_u8_t const lut_magnitude[32] = {0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26,  28,  30,
                                              32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120};

    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vint32m4_t sum_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_e2m3_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_e2m3_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);

        // Magnitude extraction + byte gather LUT
        vuint8m1_t a_magnitude_u8m1 = __riscv_vand_vx_u8m1(a_e2m3_u8m1, 0x1F, vector_length);
        vuint8m1_t b_magnitude_u8m1 = __riscv_vand_vx_u8m1(b_e2m3_u8m1, 0x1F, vector_length);
        vuint8m1_t a_unsigned_u8m1 = __riscv_vluxei8_v_u8m1(lut_magnitude, a_magnitude_u8m1, vector_length);
        vuint8m1_t b_unsigned_u8m1 = __riscv_vluxei8_v_u8m1(lut_magnitude, b_magnitude_u8m1, vector_length);

        // Combined sign + conditional negate
        vuint8m1_t sign_combined_u8m1 = __riscv_vand_vx_u8m1(
            __riscv_vxor_vv_u8m1(a_e2m3_u8m1, b_e2m3_u8m1, vector_length), 0x20, vector_length);
        vbool8_t negate_mask_b8 = __riscv_vmsne_vx_u8m1_b8(sign_combined_u8m1, 0, vector_length);
        vint8m1_t b_positive_i8m1 = __riscv_vreinterpret_v_u8m1_i8m1(b_unsigned_u8m1);
        vint8m1_t b_negated_i8m1 = __riscv_vneg_v_i8m1(b_positive_i8m1, vector_length);
        vint8m1_t b_signed_i8m1 = __riscv_vmerge_vvm_i8m1(b_positive_i8m1, b_negated_i8m1, negate_mask_b8,
                                                          vector_length);

        // Widening multiply: i8×i8 → i16, then accumulate: i32 += i16
        vint8m1_t a_signed_i8m1 = __riscv_vreinterpret_v_u8m1_i8m1(a_unsigned_u8m1);
        vint16m2_t products_i16m2 = __riscv_vwmul_vv_i16m2(a_signed_i8m1, b_signed_i8m1, vector_length);
        sum_i32m4 = __riscv_vwadd_wv_i32m4_tu(sum_i32m4, sum_i32m4, products_i16m2, vector_length);
    }
    vint32m1_t zero_i32m1 = __riscv_vmv_v_x_i32m1(0, vlmax);
    nk_i32_t sum = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(sum_i32m4, zero_i32m1, vlmax));
    *result = (nk_f32_t)sum / 256.0f;
}

NK_PUBLIC void nk_dot_e3m2_rvv(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars, nk_size_t count_scalars,
                               nk_f32_t *result) {
    // Integer dot product for e3m2 using i16 gather LUT + widening multiply.
    // Every e3m2 value × 16 is an exact integer, but magnitudes reach 448, requiring i16.
    // Result = i32_dot / 256.0f (exact, no rounding error).
    static nk_u16_t const lut_magnitude[32] = {0,  1,  2,  3,  4,  5,  6,  7,   8,   10,  12,  14,  16,  20,  24,  28,
                                               32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448};

    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vint32m4_t sum_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
    for (nk_size_t vector_length; count_scalars > 0;
         count_scalars -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count_scalars);
        vuint8m1_t a_e3m2_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_e3m2_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);

        // Magnitude extraction: lower 5 bits as u16 byte offsets for gather
        vuint8m1_t a_mag_u8m1 = __riscv_vand_vx_u8m1(a_e3m2_u8m1, 0x1F, vector_length);
        vuint8m1_t b_mag_u8m1 = __riscv_vand_vx_u8m1(b_e3m2_u8m1, 0x1F, vector_length);
        vuint16m2_t a_idx_u16m2 = __riscv_vzext_vf2_u16m2(a_mag_u8m1, vector_length);
        vuint16m2_t b_idx_u16m2 = __riscv_vzext_vf2_u16m2(b_mag_u8m1, vector_length);

        // Gather from i16 LUT: byte offsets = index × 2
        vuint16m2_t a_byte_offsets_u16m2 = __riscv_vsll_vx_u16m2(a_idx_u16m2, 1, vector_length);
        vuint16m2_t b_byte_offsets_u16m2 = __riscv_vsll_vx_u16m2(b_idx_u16m2, 1, vector_length);
        vuint16m2_t a_unsigned_u16m2 = __riscv_vluxei16_v_u16m2(lut_magnitude, a_byte_offsets_u16m2, vector_length);
        vuint16m2_t b_unsigned_u16m2 = __riscv_vluxei16_v_u16m2(lut_magnitude, b_byte_offsets_u16m2, vector_length);

        // Extract sign bits and apply conditional negate
        vuint8m1_t a_sign_u8m1 = __riscv_vand_vx_u8m1(a_e3m2_u8m1, 0x20, vector_length);
        vuint8m1_t b_sign_u8m1 = __riscv_vand_vx_u8m1(b_e3m2_u8m1, 0x20, vector_length);
        vbool8_t a_negate_b8 = __riscv_vmsne_vx_u8m1_b8(a_sign_u8m1, 0, vector_length);
        vbool8_t b_negate_b8 = __riscv_vmsne_vx_u8m1_b8(b_sign_u8m1, 0, vector_length);

        vint16m2_t a_signed_i16m2 = __riscv_vreinterpret_v_u16m2_i16m2(a_unsigned_u16m2);
        a_signed_i16m2 = __riscv_vneg_v_i16m2_mu(a_negate_b8, a_signed_i16m2, a_signed_i16m2, vector_length);

        vint16m2_t b_signed_i16m2 = __riscv_vreinterpret_v_u16m2_i16m2(b_unsigned_u16m2);
        b_signed_i16m2 = __riscv_vneg_v_i16m2_mu(b_negate_b8, b_signed_i16m2, b_signed_i16m2, vector_length);

        // Widening multiply-accumulate: i16×i16 → i32
        sum_i32m4 = __riscv_vwmacc_vv_i32m4_tu(sum_i32m4, a_signed_i16m2, b_signed_i16m2, vector_length);
    }
    vint32m1_t zero_i32m1 = __riscv_vmv_v_x_i32m1(0, vlmax);
    nk_i32_t sum = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(sum_i32m4, zero_i32m1, vlmax));
    *result = (nk_f32_t)sum / 256.0f;
}

NK_PUBLIC void nk_dot_i4_rvv(nk_i4x2_t const *a_scalars, nk_i4x2_t const *b_scalars, nk_size_t count_dimensions,
                             nk_i32_t *result) {
    // count_dimensions = number of 4-bit values, not bytes
    count_dimensions = nk_size_round_up_to_multiple_(count_dimensions, 2);
    nk_size_t n_full_bytes = count_dimensions / 2;

    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vint32m4_t sum_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax);
    for (nk_size_t vector_length; n_full_bytes > 0;
         n_full_bytes -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n_full_bytes);

        vuint8m1_t a_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);

        vuint8m1_t a_high_u8m1 = __riscv_vsrl_vx_u8m1(a_packed_u8m1, 4, vector_length);
        vuint8m1_t b_high_u8m1 = __riscv_vsrl_vx_u8m1(b_packed_u8m1, 4, vector_length);
        vuint8m1_t a_low_u8m1 = __riscv_vand_vx_u8m1(a_packed_u8m1, 0x0F, vector_length);
        vuint8m1_t b_low_u8m1 = __riscv_vand_vx_u8m1(b_packed_u8m1, 0x0F, vector_length);

        // Sign extend 4-bit to 8-bit: (x ^ 8) - 8
        vint8m1_t a_high_i8m1 = __riscv_vsub_vx_i8m1(
            __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(a_high_u8m1), 8, vector_length), 8, vector_length);
        vint8m1_t b_high_i8m1 = __riscv_vsub_vx_i8m1(
            __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(b_high_u8m1), 8, vector_length), 8, vector_length);
        vint8m1_t a_low_i8m1 = __riscv_vsub_vx_i8m1(
            __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(a_low_u8m1), 8, vector_length), 8, vector_length);
        vint8m1_t b_low_i8m1 = __riscv_vsub_vx_i8m1(
            __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(b_low_u8m1), 8, vector_length), 8, vector_length);

        // Widening multiply: i8 ⨯ i8 → i16
        vint16m2_t ab_high_i16m2 = __riscv_vwmul_vv_i16m2(a_high_i8m1, b_high_i8m1, vector_length);
        vint16m2_t ab_low_i16m2 = __riscv_vwmul_vv_i16m2(a_low_i8m1, b_low_i8m1, vector_length);

        // Per-lane widening accumulate: i32 += i16
        sum_i32m4 = __riscv_vwadd_wv_i32m4_tu(sum_i32m4, sum_i32m4, ab_high_i16m2, vector_length);
        sum_i32m4 = __riscv_vwadd_wv_i32m4_tu(sum_i32m4, sum_i32m4, ab_low_i16m2, vector_length);
    }
    // Single horizontal reduction at the end
    vint32m1_t zero_i32m1 = __riscv_vmv_v_x_i32m1(0, vlmax);
    *result = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(sum_i32m4, zero_i32m1, vlmax));
}

NK_PUBLIC void nk_dot_u4_rvv(nk_u4x2_t const *a_scalars, nk_u4x2_t const *b_scalars, nk_size_t count_dimensions,
                             nk_u32_t *result) {
    // count_dimensions = number of 4-bit values, not bytes
    count_dimensions = nk_size_round_up_to_multiple_(count_dimensions, 2);
    nk_size_t n_full_bytes = count_dimensions / 2;

    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vuint32m4_t sum_u32m4 = __riscv_vmv_v_x_u32m4(0, vlmax);
    for (nk_size_t vector_length; n_full_bytes > 0;
         n_full_bytes -= vector_length, a_scalars += vector_length, b_scalars += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n_full_bytes);

        vuint8m1_t a_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a_scalars, vector_length);
        vuint8m1_t b_packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b_scalars, vector_length);

        vuint8m1_t a_high_u8m1 = __riscv_vsrl_vx_u8m1(a_packed_u8m1, 4, vector_length);
        vuint8m1_t b_high_u8m1 = __riscv_vsrl_vx_u8m1(b_packed_u8m1, 4, vector_length);
        vuint8m1_t a_low_u8m1 = __riscv_vand_vx_u8m1(a_packed_u8m1, 0x0F, vector_length);
        vuint8m1_t b_low_u8m1 = __riscv_vand_vx_u8m1(b_packed_u8m1, 0x0F, vector_length);

        // Widening multiply: u8 ⨯ u8 → u16
        vuint16m2_t ab_high_u16m2 = __riscv_vwmulu_vv_u16m2(a_high_u8m1, b_high_u8m1, vector_length);
        vuint16m2_t ab_low_u16m2 = __riscv_vwmulu_vv_u16m2(a_low_u8m1, b_low_u8m1, vector_length);

        // Per-lane widening accumulate: u32 += u16
        sum_u32m4 = __riscv_vwaddu_wv_u32m4_tu(sum_u32m4, sum_u32m4, ab_high_u16m2, vector_length);
        sum_u32m4 = __riscv_vwaddu_wv_u32m4_tu(sum_u32m4, sum_u32m4, ab_low_u16m2, vector_length);
    }
    // Single horizontal reduction at the end
    vuint32m1_t zero_u32m1 = __riscv_vmv_v_x_u32m1(0, vlmax);
    *result = __riscv_vmv_x_s_u32m1_u32(__riscv_vredsum_vs_u32m4_u32m1(sum_u32m4, zero_u32m1, vlmax));
}

NK_PUBLIC void nk_dot_u1_rvv(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result) {
    nk_size_t count_bytes = nk_size_divide_round_up_(n_bits, NK_BITS_PER_BYTE);

    vuint32m1_t sum_u32m1 = __riscv_vmv_v_x_u32m1(0, 1);

    nk_size_t i = 0;
    for (nk_size_t vector_length; i + 1 <= count_bytes; i += vector_length) {
        vector_length = __riscv_vsetvl_e8m4(count_bytes - i);

        // Load and AND to find shared bits (dot product of binary vectors)
        vuint8m4_t a_u8m4 = __riscv_vle8_v_u8m4(a + i, vector_length);
        vuint8m4_t b_u8m4 = __riscv_vle8_v_u8m4(b + i, vector_length);
        vuint8m4_t and_u8m4 = __riscv_vand_vv_u8m4(a_u8m4, b_u8m4, vector_length);

        // Popcount each byte using arithmetic SWAR
        vuint8m4_t popcount_u8m4 = nk_popcount_u8m4_rvv_(and_u8m4, vector_length);

        // Widen to u16 and accumulate via widening reduction sum
        vuint16m8_t popcount_u16m8 = __riscv_vwaddu_vx_u16m8(popcount_u8m4, 0, vector_length);
        sum_u32m1 = __riscv_vwredsumu_vs_u16m8_u32m1(popcount_u16m8, sum_u32m1, vector_length);
    }

    *result = __riscv_vmv_x_s_u32m1_u32(sum_u32m1);
}

NK_PUBLIC void nk_dot_f32c_rvv(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                               nk_f64c_t *results) {
    nk_f32_t const *a_f32 = (nk_f32_t const *)a_pairs;
    nk_f32_t const *b_f32 = (nk_f32_t const *)b_pairs;
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_real_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_imag_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    for (nk_size_t vector_length; count_pairs > 0;
         count_pairs -= vector_length, a_f32 += vector_length * 2, b_f32 += vector_length * 2) {
        vector_length = __riscv_vsetvl_e32m1(count_pairs);
        vfloat32m1x2_t a_f32m1x2 = __riscv_vlseg2e32_v_f32m1x2(a_f32, vector_length);
        vfloat32m1x2_t b_f32m1x2 = __riscv_vlseg2e32_v_f32m1x2(b_f32, vector_length);
        vfloat32m1_t a_real_f32m1 = __riscv_vget_v_f32m1x2_f32m1(a_f32m1x2, 0);
        vfloat32m1_t a_imag_f32m1 = __riscv_vget_v_f32m1x2_f32m1(a_f32m1x2, 1);
        vfloat32m1_t b_real_f32m1 = __riscv_vget_v_f32m1x2_f32m1(b_f32m1x2, 0);
        vfloat32m1_t b_imag_f32m1 = __riscv_vget_v_f32m1x2_f32m1(b_f32m1x2, 1);
        // real += a_real * b_real - a_imag * b_imag
        sum_real_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(sum_real_f64m2, a_real_f32m1, b_real_f32m1, vector_length);
        sum_real_f64m2 = __riscv_vfwnmsac_vv_f64m2_tu(sum_real_f64m2, a_imag_f32m1, b_imag_f32m1, vector_length);
        // imag += a_real * b_imag + a_imag * b_real
        sum_imag_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(sum_imag_f64m2, a_real_f32m1, b_imag_f32m1, vector_length);
        sum_imag_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(sum_imag_f64m2, a_imag_f32m1, b_real_f32m1, vector_length);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    results->real = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_real_f64m2, zero_f64m1, vlmax));
    results->imag = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_imag_f64m2, zero_f64m1, vlmax));
}

NK_PUBLIC void nk_vdot_f32c_rvv(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                nk_f64c_t *results) {
    nk_f32_t const *a_f32 = (nk_f32_t const *)a_pairs;
    nk_f32_t const *b_f32 = (nk_f32_t const *)b_pairs;
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_real_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_imag_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    for (nk_size_t vector_length; count_pairs > 0;
         count_pairs -= vector_length, a_f32 += vector_length * 2, b_f32 += vector_length * 2) {
        vector_length = __riscv_vsetvl_e32m1(count_pairs);
        vfloat32m1x2_t a_f32m1x2 = __riscv_vlseg2e32_v_f32m1x2(a_f32, vector_length);
        vfloat32m1x2_t b_f32m1x2 = __riscv_vlseg2e32_v_f32m1x2(b_f32, vector_length);
        vfloat32m1_t a_real_f32m1 = __riscv_vget_v_f32m1x2_f32m1(a_f32m1x2, 0);
        vfloat32m1_t a_imag_f32m1 = __riscv_vget_v_f32m1x2_f32m1(a_f32m1x2, 1);
        vfloat32m1_t b_real_f32m1 = __riscv_vget_v_f32m1x2_f32m1(b_f32m1x2, 0);
        vfloat32m1_t b_imag_f32m1 = __riscv_vget_v_f32m1x2_f32m1(b_f32m1x2, 1);
        // Conjugate dot: real += a_real * b_real + a_imag * b_imag
        sum_real_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(sum_real_f64m2, a_real_f32m1, b_real_f32m1, vector_length);
        sum_real_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(sum_real_f64m2, a_imag_f32m1, b_imag_f32m1, vector_length);
        // Conjugate dot: imag += a_real * b_imag - a_imag * b_real
        sum_imag_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(sum_imag_f64m2, a_real_f32m1, b_imag_f32m1, vector_length);
        sum_imag_f64m2 = __riscv_vfwnmsac_vv_f64m2_tu(sum_imag_f64m2, a_imag_f32m1, b_real_f32m1, vector_length);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    results->real = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_real_f64m2, zero_f64m1, vlmax));
    results->imag = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_imag_f64m2, zero_f64m1, vlmax));
}

NK_PUBLIC void nk_dot_f64c_rvv(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                               nk_f64c_t *results) {
    // Dot2 (Ogita-Rump-Oishi) compensated complex dot product
    nk_f64_t const *a_f64 = (nk_f64_t const *)a_pairs;
    nk_f64_t const *b_f64 = (nk_f64_t const *)b_pairs;
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_real_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t comp_real_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_imag_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t comp_imag_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    for (nk_size_t vector_length; count_pairs > 0;
         count_pairs -= vector_length, a_f64 += vector_length * 2, b_f64 += vector_length * 2) {
        vector_length = __riscv_vsetvl_e64m1(count_pairs);
        vfloat64m1x2_t a_f64m1x2 = __riscv_vlseg2e64_v_f64m1x2(a_f64, vector_length);
        vfloat64m1x2_t b_f64m1x2 = __riscv_vlseg2e64_v_f64m1x2(b_f64, vector_length);
        vfloat64m1_t a_real_f64m1 = __riscv_vget_v_f64m1x2_f64m1(a_f64m1x2, 0);
        vfloat64m1_t a_imag_f64m1 = __riscv_vget_v_f64m1x2_f64m1(a_f64m1x2, 1);
        vfloat64m1_t b_real_f64m1 = __riscv_vget_v_f64m1x2_f64m1(b_f64m1x2, 0);
        vfloat64m1_t b_imag_f64m1 = __riscv_vget_v_f64m1x2_f64m1(b_f64m1x2, 1);
        // TwoProd+TwoSum: sum_real += a_real * b_real
        {
            vfloat64m1_t product_f64m1 = __riscv_vfmul_vv_f64m1(a_real_f64m1, b_real_f64m1, vector_length);
            vfloat64m1_t product_error_f64m1 = __riscv_vfmsac_vv_f64m1(product_f64m1, a_real_f64m1, b_real_f64m1,
                                                                       vector_length);
            vfloat64m1_t tentative_sum_f64m1 = __riscv_vfadd_vv_f64m1(sum_real_f64m1, product_f64m1, vector_length);
            vfloat64m1_t virtual_addend_f64m1 = __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, sum_real_f64m1,
                                                                       vector_length);
            vfloat64m1_t sum_error_f64m1 = __riscv_vfadd_vv_f64m1(
                __riscv_vfsub_vv_f64m1(sum_real_f64m1,
                                       __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, virtual_addend_f64m1, vector_length),
                                       vector_length),
                __riscv_vfsub_vv_f64m1(product_f64m1, virtual_addend_f64m1, vector_length), vector_length);
            sum_real_f64m1 = __riscv_vslideup_vx_f64m1_tu(sum_real_f64m1, tentative_sum_f64m1, 0, vector_length);
            vfloat64m1_t total_error_f64m1 = __riscv_vfadd_vv_f64m1(sum_error_f64m1, product_error_f64m1,
                                                                    vector_length);
            comp_real_f64m1 = __riscv_vfadd_vv_f64m1_tu(comp_real_f64m1, comp_real_f64m1, total_error_f64m1,
                                                        vector_length);
        }
        // TwoProd+TwoSum: sum_real -= a_imag * b_imag
        {
            vfloat64m1_t product_f64m1 = __riscv_vfmul_vv_f64m1(a_imag_f64m1, b_imag_f64m1, vector_length);
            vfloat64m1_t product_error_f64m1 = __riscv_vfmsac_vv_f64m1(product_f64m1, a_imag_f64m1, b_imag_f64m1,
                                                                       vector_length);
            vfloat64m1_t neg_product_f64m1 = __riscv_vfneg_v_f64m1(product_f64m1, vector_length);
            vfloat64m1_t neg_product_error_f64m1 = __riscv_vfneg_v_f64m1(product_error_f64m1, vector_length);
            vfloat64m1_t tentative_sum_f64m1 = __riscv_vfadd_vv_f64m1(sum_real_f64m1, neg_product_f64m1, vector_length);
            vfloat64m1_t virtual_addend_f64m1 = __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, sum_real_f64m1,
                                                                       vector_length);
            vfloat64m1_t sum_error_f64m1 = __riscv_vfadd_vv_f64m1(
                __riscv_vfsub_vv_f64m1(sum_real_f64m1,
                                       __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, virtual_addend_f64m1, vector_length),
                                       vector_length),
                __riscv_vfsub_vv_f64m1(neg_product_f64m1, virtual_addend_f64m1, vector_length), vector_length);
            sum_real_f64m1 = __riscv_vslideup_vx_f64m1_tu(sum_real_f64m1, tentative_sum_f64m1, 0, vector_length);
            vfloat64m1_t total_error_f64m1 = __riscv_vfadd_vv_f64m1(sum_error_f64m1, neg_product_error_f64m1,
                                                                    vector_length);
            comp_real_f64m1 = __riscv_vfadd_vv_f64m1_tu(comp_real_f64m1, comp_real_f64m1, total_error_f64m1,
                                                        vector_length);
        }
        // TwoProd+TwoSum: sum_imag += a_real * b_imag
        {
            vfloat64m1_t product_f64m1 = __riscv_vfmul_vv_f64m1(a_real_f64m1, b_imag_f64m1, vector_length);
            vfloat64m1_t product_error_f64m1 = __riscv_vfmsac_vv_f64m1(product_f64m1, a_real_f64m1, b_imag_f64m1,
                                                                       vector_length);
            vfloat64m1_t tentative_sum_f64m1 = __riscv_vfadd_vv_f64m1(sum_imag_f64m1, product_f64m1, vector_length);
            vfloat64m1_t virtual_addend_f64m1 = __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, sum_imag_f64m1,
                                                                       vector_length);
            vfloat64m1_t sum_error_f64m1 = __riscv_vfadd_vv_f64m1(
                __riscv_vfsub_vv_f64m1(sum_imag_f64m1,
                                       __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, virtual_addend_f64m1, vector_length),
                                       vector_length),
                __riscv_vfsub_vv_f64m1(product_f64m1, virtual_addend_f64m1, vector_length), vector_length);
            sum_imag_f64m1 = __riscv_vslideup_vx_f64m1_tu(sum_imag_f64m1, tentative_sum_f64m1, 0, vector_length);
            vfloat64m1_t total_error_f64m1 = __riscv_vfadd_vv_f64m1(sum_error_f64m1, product_error_f64m1,
                                                                    vector_length);
            comp_imag_f64m1 = __riscv_vfadd_vv_f64m1_tu(comp_imag_f64m1, comp_imag_f64m1, total_error_f64m1,
                                                        vector_length);
        }
        // TwoProd+TwoSum: sum_imag += a_imag * b_real
        {
            vfloat64m1_t product_f64m1 = __riscv_vfmul_vv_f64m1(a_imag_f64m1, b_real_f64m1, vector_length);
            vfloat64m1_t product_error_f64m1 = __riscv_vfmsac_vv_f64m1(product_f64m1, a_imag_f64m1, b_real_f64m1,
                                                                       vector_length);
            vfloat64m1_t tentative_sum_f64m1 = __riscv_vfadd_vv_f64m1(sum_imag_f64m1, product_f64m1, vector_length);
            vfloat64m1_t virtual_addend_f64m1 = __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, sum_imag_f64m1,
                                                                       vector_length);
            vfloat64m1_t sum_error_f64m1 = __riscv_vfadd_vv_f64m1(
                __riscv_vfsub_vv_f64m1(sum_imag_f64m1,
                                       __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, virtual_addend_f64m1, vector_length),
                                       vector_length),
                __riscv_vfsub_vv_f64m1(product_f64m1, virtual_addend_f64m1, vector_length), vector_length);
            sum_imag_f64m1 = __riscv_vslideup_vx_f64m1_tu(sum_imag_f64m1, tentative_sum_f64m1, 0, vector_length);
            vfloat64m1_t total_error_f64m1 = __riscv_vfadd_vv_f64m1(sum_error_f64m1, product_error_f64m1,
                                                                    vector_length);
            comp_imag_f64m1 = __riscv_vfadd_vv_f64m1_tu(comp_imag_f64m1, comp_imag_f64m1, total_error_f64m1,
                                                        vector_length);
        }
    }
    results->real = nk_dot_stable_sum_f64m1_rvv_(sum_real_f64m1, comp_real_f64m1);
    results->imag = nk_dot_stable_sum_f64m1_rvv_(sum_imag_f64m1, comp_imag_f64m1);
}

NK_PUBLIC void nk_vdot_f64c_rvv(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                nk_f64c_t *results) {
    // Dot2 (Ogita-Rump-Oishi) compensated conjugate complex dot product
    nk_f64_t const *a_f64 = (nk_f64_t const *)a_pairs;
    nk_f64_t const *b_f64 = (nk_f64_t const *)b_pairs;
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_real_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t comp_real_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_imag_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t comp_imag_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    for (nk_size_t vector_length; count_pairs > 0;
         count_pairs -= vector_length, a_f64 += vector_length * 2, b_f64 += vector_length * 2) {
        vector_length = __riscv_vsetvl_e64m1(count_pairs);
        vfloat64m1x2_t a_f64m1x2 = __riscv_vlseg2e64_v_f64m1x2(a_f64, vector_length);
        vfloat64m1x2_t b_f64m1x2 = __riscv_vlseg2e64_v_f64m1x2(b_f64, vector_length);
        vfloat64m1_t a_real_f64m1 = __riscv_vget_v_f64m1x2_f64m1(a_f64m1x2, 0);
        vfloat64m1_t a_imag_f64m1 = __riscv_vget_v_f64m1x2_f64m1(a_f64m1x2, 1);
        vfloat64m1_t b_real_f64m1 = __riscv_vget_v_f64m1x2_f64m1(b_f64m1x2, 0);
        vfloat64m1_t b_imag_f64m1 = __riscv_vget_v_f64m1x2_f64m1(b_f64m1x2, 1);
        // TwoProd+TwoSum: sum_real += a_real * b_real
        {
            vfloat64m1_t product_f64m1 = __riscv_vfmul_vv_f64m1(a_real_f64m1, b_real_f64m1, vector_length);
            vfloat64m1_t product_error_f64m1 = __riscv_vfmsac_vv_f64m1(product_f64m1, a_real_f64m1, b_real_f64m1,
                                                                       vector_length);
            vfloat64m1_t tentative_sum_f64m1 = __riscv_vfadd_vv_f64m1(sum_real_f64m1, product_f64m1, vector_length);
            vfloat64m1_t virtual_addend_f64m1 = __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, sum_real_f64m1,
                                                                       vector_length);
            vfloat64m1_t sum_error_f64m1 = __riscv_vfadd_vv_f64m1(
                __riscv_vfsub_vv_f64m1(sum_real_f64m1,
                                       __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, virtual_addend_f64m1, vector_length),
                                       vector_length),
                __riscv_vfsub_vv_f64m1(product_f64m1, virtual_addend_f64m1, vector_length), vector_length);
            sum_real_f64m1 = __riscv_vslideup_vx_f64m1_tu(sum_real_f64m1, tentative_sum_f64m1, 0, vector_length);
            vfloat64m1_t total_error_f64m1 = __riscv_vfadd_vv_f64m1(sum_error_f64m1, product_error_f64m1,
                                                                    vector_length);
            comp_real_f64m1 = __riscv_vfadd_vv_f64m1_tu(comp_real_f64m1, comp_real_f64m1, total_error_f64m1,
                                                        vector_length);
        }
        // TwoProd+TwoSum: sum_real += a_imag * b_imag (conjugate: + instead of -)
        {
            vfloat64m1_t product_f64m1 = __riscv_vfmul_vv_f64m1(a_imag_f64m1, b_imag_f64m1, vector_length);
            vfloat64m1_t product_error_f64m1 = __riscv_vfmsac_vv_f64m1(product_f64m1, a_imag_f64m1, b_imag_f64m1,
                                                                       vector_length);
            vfloat64m1_t tentative_sum_f64m1 = __riscv_vfadd_vv_f64m1(sum_real_f64m1, product_f64m1, vector_length);
            vfloat64m1_t virtual_addend_f64m1 = __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, sum_real_f64m1,
                                                                       vector_length);
            vfloat64m1_t sum_error_f64m1 = __riscv_vfadd_vv_f64m1(
                __riscv_vfsub_vv_f64m1(sum_real_f64m1,
                                       __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, virtual_addend_f64m1, vector_length),
                                       vector_length),
                __riscv_vfsub_vv_f64m1(product_f64m1, virtual_addend_f64m1, vector_length), vector_length);
            sum_real_f64m1 = __riscv_vslideup_vx_f64m1_tu(sum_real_f64m1, tentative_sum_f64m1, 0, vector_length);
            vfloat64m1_t total_error_f64m1 = __riscv_vfadd_vv_f64m1(sum_error_f64m1, product_error_f64m1,
                                                                    vector_length);
            comp_real_f64m1 = __riscv_vfadd_vv_f64m1_tu(comp_real_f64m1, comp_real_f64m1, total_error_f64m1,
                                                        vector_length);
        }
        // TwoProd+TwoSum: sum_imag += a_real * b_imag
        {
            vfloat64m1_t product_f64m1 = __riscv_vfmul_vv_f64m1(a_real_f64m1, b_imag_f64m1, vector_length);
            vfloat64m1_t product_error_f64m1 = __riscv_vfmsac_vv_f64m1(product_f64m1, a_real_f64m1, b_imag_f64m1,
                                                                       vector_length);
            vfloat64m1_t tentative_sum_f64m1 = __riscv_vfadd_vv_f64m1(sum_imag_f64m1, product_f64m1, vector_length);
            vfloat64m1_t virtual_addend_f64m1 = __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, sum_imag_f64m1,
                                                                       vector_length);
            vfloat64m1_t sum_error_f64m1 = __riscv_vfadd_vv_f64m1(
                __riscv_vfsub_vv_f64m1(sum_imag_f64m1,
                                       __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, virtual_addend_f64m1, vector_length),
                                       vector_length),
                __riscv_vfsub_vv_f64m1(product_f64m1, virtual_addend_f64m1, vector_length), vector_length);
            sum_imag_f64m1 = __riscv_vslideup_vx_f64m1_tu(sum_imag_f64m1, tentative_sum_f64m1, 0, vector_length);
            vfloat64m1_t total_error_f64m1 = __riscv_vfadd_vv_f64m1(sum_error_f64m1, product_error_f64m1,
                                                                    vector_length);
            comp_imag_f64m1 = __riscv_vfadd_vv_f64m1_tu(comp_imag_f64m1, comp_imag_f64m1, total_error_f64m1,
                                                        vector_length);
        }
        // TwoProd+TwoSum: sum_imag -= a_imag * b_real (conjugate: - instead of +)
        {
            vfloat64m1_t product_f64m1 = __riscv_vfmul_vv_f64m1(a_imag_f64m1, b_real_f64m1, vector_length);
            vfloat64m1_t product_error_f64m1 = __riscv_vfmsac_vv_f64m1(product_f64m1, a_imag_f64m1, b_real_f64m1,
                                                                       vector_length);
            vfloat64m1_t neg_product_f64m1 = __riscv_vfneg_v_f64m1(product_f64m1, vector_length);
            vfloat64m1_t neg_product_error_f64m1 = __riscv_vfneg_v_f64m1(product_error_f64m1, vector_length);
            vfloat64m1_t tentative_sum_f64m1 = __riscv_vfadd_vv_f64m1(sum_imag_f64m1, neg_product_f64m1, vector_length);
            vfloat64m1_t virtual_addend_f64m1 = __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, sum_imag_f64m1,
                                                                       vector_length);
            vfloat64m1_t sum_error_f64m1 = __riscv_vfadd_vv_f64m1(
                __riscv_vfsub_vv_f64m1(sum_imag_f64m1,
                                       __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, virtual_addend_f64m1, vector_length),
                                       vector_length),
                __riscv_vfsub_vv_f64m1(neg_product_f64m1, virtual_addend_f64m1, vector_length), vector_length);
            sum_imag_f64m1 = __riscv_vslideup_vx_f64m1_tu(sum_imag_f64m1, tentative_sum_f64m1, 0, vector_length);
            vfloat64m1_t total_error_f64m1 = __riscv_vfadd_vv_f64m1(sum_error_f64m1, neg_product_error_f64m1,
                                                                    vector_length);
            comp_imag_f64m1 = __riscv_vfadd_vv_f64m1_tu(comp_imag_f64m1, comp_imag_f64m1, total_error_f64m1,
                                                        vector_length);
        }
    }
    results->real = nk_dot_stable_sum_f64m1_rvv_(sum_real_f64m1, comp_real_f64m1);
    results->imag = nk_dot_stable_sum_f64m1_rvv_(sum_imag_f64m1, comp_imag_f64m1);
}

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
#endif // NK_DOT_RVV_H
