/**
 *  @brief SIMD-accelerated Curved Space Distances for RISC-V.
 *  @file include/numkong/curved/rvv.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/curved.h
 *
 *  Implements bilinear forms and Mahalanobis distance using RVV 1.0:
 *  - f32 inputs use f32 SIMD accumulation with vfredusum ordered reduction
 *  - f64 inputs use f64 SIMD accumulation with vfredusum ordered reduction
 *  - f16/bf16 inputs are converted to f32 via cast helpers, then accumulated in f32
 *  - Complex bilinear forms delegate to serial implementations
 */
#ifndef NK_CURVED_RVV_H
#define NK_CURVED_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/curved/serial.h"
#include "numkong/cast/rvv.h"
#include "numkong/spatial/rvv.h" // `nk_f64_sqrt_rvv`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_bilinear_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                   nk_f64_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    nk_f64_t outer_sum = 0;
    for (nk_size_t i = 0; i < n; ++i) {
        vfloat64m4_t inner_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
        nk_f32_t const *c_row = c + i * n;
        nk_size_t remaining = n;
        for (nk_size_t vector_length; remaining > 0; remaining -= vector_length, c_row += vector_length) {
            vector_length = __riscv_vsetvl_e32m2(remaining);
            vfloat32m2_t c_f32m2 = __riscv_vle32_v_f32m2(c_row, vector_length);
            vfloat32m2_t b_f32m2 = __riscv_vle32_v_f32m2(b + (n - remaining), vector_length);
            inner_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(inner_f64m4, c_f32m2, b_f32m2, vector_length);
        }
        vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
        nk_f64_t inner_val = __riscv_vfmv_f_s_f64m1_f64(
            __riscv_vfredusum_vs_f64m4_f64m1(inner_f64m4, zero_f64m1, vlmax));
        outer_sum += (nk_f64_t)a[i] * inner_val;
    }
    *result = outer_sum;
}

NK_PUBLIC void nk_bilinear_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                   nk_f64_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m1_t sum_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f64_t outer_compensation = 0;
    for (nk_size_t i = 0; i < n; ++i) {
        vfloat64m4_t inner_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
        vfloat64m4_t compensation_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
        nk_f64_t const *c_row = c + i * n;
        nk_size_t remaining = n;
        for (nk_size_t vector_length; remaining > 0; remaining -= vector_length, c_row += vector_length) {
            vector_length = __riscv_vsetvl_e64m4(remaining);
            vfloat64m4_t vc_f64m4 = __riscv_vle64_v_f64m4(c_row, vector_length);
            vfloat64m4_t vb_f64m4 = __riscv_vle64_v_f64m4(b + (n - remaining), vector_length);
            vfloat64m4_t product_f64m4 = __riscv_vfmul_vv_f64m4(vc_f64m4, vb_f64m4, vector_length);
            vfloat64m4_t corrected_term_f64m4 = __riscv_vfsub_vv_f64m4(product_f64m4, compensation_f64m4,
                                                                       vector_length);
            vfloat64m4_t running_sum_f64m4 = __riscv_vfadd_vv_f64m4_tu(inner_f64m4, inner_f64m4, corrected_term_f64m4,
                                                                       vector_length);
            compensation_f64m4 = __riscv_vfsub_vv_f64m4_tu(
                compensation_f64m4, __riscv_vfsub_vv_f64m4(running_sum_f64m4, inner_f64m4, vector_length),
                corrected_term_f64m4, vector_length);
            inner_f64m4 = running_sum_f64m4;
        }
        vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
        nk_f64_t inner_val = __riscv_vfmv_f_s_f64m1_f64(
            __riscv_vfredusum_vs_f64m4_f64m1(inner_f64m4, zero_f64m1, vlmax));
        nk_f64_t product_outer = a[i] * inner_val;
        nk_f64_t old_sum = __riscv_vfmv_f_s_f64m1_f64(sum_f64m1);
        nk_f64_t new_sum = old_sum + product_outer;
        if (nk_f64_abs_(old_sum) >= nk_f64_abs_(product_outer))
            outer_compensation += (old_sum - new_sum) + product_outer;
        else outer_compensation += (product_outer - new_sum) + old_sum;
        sum_f64m1 = __riscv_vfmv_v_f_f64m1(new_sum, 1);
    }
    *result = __riscv_vfmv_f_s_f64m1_f64(sum_f64m1) + outer_compensation;
}

NK_PUBLIC void nk_bilinear_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                   nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t i = 0; i < n; ++i) {
        // Convert a[i] from f16 to f32
        nk_f32_t a_i;
        nk_f16_to_f32_serial(a + i, &a_i);

        vfloat32m2_t inner_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        nk_f16_t const *c_row = c + i * n;
        nk_size_t remaining = n;
        for (nk_size_t vector_length; remaining > 0; remaining -= vector_length, c_row += vector_length) {
            vector_length = __riscv_vsetvl_e16m1(remaining);
            // Load f16 as u16 bits and convert to f32
            vuint16m1_t vc_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)c_row, vector_length);
            vuint16m1_t vb_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)(b + (n - remaining)), vector_length);
            vfloat32m2_t vc_f32m2 = nk_f16m1_to_f32m2_rvv_(vc_u16m1, vector_length);
            vfloat32m2_t vb_f32m2 = nk_f16m1_to_f32m2_rvv_(vb_u16m1, vector_length);
            inner_f32m2 = __riscv_vfmacc_vv_f32m2_tu(inner_f32m2, vc_f32m2, vb_f32m2, vector_length);
        }
        vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        nk_f32_t inner_val = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m2_f32m1(inner_f32m2, zero_f32m1, vlmax));
        sum_f32m1 = __riscv_vfmv_v_f_f32m1(__riscv_vfmv_f_s_f32m1_f32(sum_f32m1) + a_i * inner_val, 1);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_bilinear_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                    nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t i = 0; i < n; ++i) {
        // Convert a[i] from bf16 to f32
        nk_f32_t a_i;
        nk_bf16_to_f32_serial(a + i, &a_i);

        vfloat32m2_t inner_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        nk_bf16_t const *c_row = c + i * n;
        nk_size_t remaining = n;
        for (nk_size_t vector_length; remaining > 0; remaining -= vector_length, c_row += vector_length) {
            vector_length = __riscv_vsetvl_e16m1(remaining);
            // Load bf16 as u16 bits and convert to f32
            vuint16m1_t vc_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)c_row, vector_length);
            vuint16m1_t vb_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)(b + (n - remaining)), vector_length);
            vfloat32m2_t vc_f32m2 = nk_bf16m1_to_f32m2_rvv_(vc_u16m1, vector_length);
            vfloat32m2_t vb_f32m2 = nk_bf16m1_to_f32m2_rvv_(vb_u16m1, vector_length);
            inner_f32m2 = __riscv_vfmacc_vv_f32m2_tu(inner_f32m2, vc_f32m2, vb_f32m2, vector_length);
        }
        vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        nk_f32_t inner_val = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m2_f32m1(inner_f32m2, zero_f32m1, vlmax));
        sum_f32m1 = __riscv_vfmv_v_f_f32m1(__riscv_vfmv_f_s_f32m1_f32(sum_f32m1) + a_i * inner_val, 1);
    }
    *result = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

NK_PUBLIC void nk_mahalanobis_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                      nk_f64_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    nk_f64_t outer_sum = 0;
    for (nk_size_t i = 0; i < n; ++i) {
        nk_f64_t diff_i = (nk_f64_t)a[i] - (nk_f64_t)b[i];
        vfloat64m4_t inner_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
        nk_f32_t const *c_row = c + i * n;
        nk_size_t remaining = n;
        for (nk_size_t vector_length; remaining > 0; remaining -= vector_length, c_row += vector_length) {
            vector_length = __riscv_vsetvl_e32m2(remaining);
            nk_size_t j = n - remaining;
            vfloat32m2_t c_f32m2 = __riscv_vle32_v_f32m2(c_row, vector_length);
            vfloat32m2_t a_f32m2 = __riscv_vle32_v_f32m2(a + j, vector_length);
            vfloat32m2_t b_f32m2 = __riscv_vle32_v_f32m2(b + j, vector_length);
            vfloat64m4_t diff_f64m4 = __riscv_vfwsub_vv_f64m4(a_f32m2, b_f32m2, vector_length);
            vfloat64m4_t c_f64m4 = __riscv_vfwcvt_f_f_v_f64m4(c_f32m2, vector_length);
            inner_f64m4 = __riscv_vfmacc_vv_f64m4_tu(inner_f64m4, c_f64m4, diff_f64m4, vector_length);
        }
        vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
        nk_f64_t inner_val = __riscv_vfmv_f_s_f64m1_f64(
            __riscv_vfredusum_vs_f64m4_f64m1(inner_f64m4, zero_f64m1, vlmax));
        outer_sum += diff_i * inner_val;
    }
    *result = nk_f64_sqrt_rvv(outer_sum > 0 ? outer_sum : 0);
}

NK_PUBLIC void nk_mahalanobis_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                      nk_f64_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m1_t sum_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f64_t outer_compensation = 0;
    for (nk_size_t i = 0; i < n; ++i) {
        nk_f64_t diff_i = a[i] - b[i];
        vfloat64m4_t inner_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
        vfloat64m4_t compensation_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
        nk_f64_t const *c_row = c + i * n;
        nk_size_t remaining = n;
        for (nk_size_t vector_length; remaining > 0; remaining -= vector_length, c_row += vector_length) {
            vector_length = __riscv_vsetvl_e64m4(remaining);
            nk_size_t j = n - remaining;
            vfloat64m4_t vc_f64m4 = __riscv_vle64_v_f64m4(c_row, vector_length);
            vfloat64m4_t va_f64m4 = __riscv_vle64_v_f64m4(a + j, vector_length);
            vfloat64m4_t vb_f64m4 = __riscv_vle64_v_f64m4(b + j, vector_length);
            vfloat64m4_t diff_j_f64m4 = __riscv_vfsub_vv_f64m4(va_f64m4, vb_f64m4, vector_length);
            vfloat64m4_t product_f64m4 = __riscv_vfmul_vv_f64m4(vc_f64m4, diff_j_f64m4, vector_length);
            vfloat64m4_t corrected_term_f64m4 = __riscv_vfsub_vv_f64m4(product_f64m4, compensation_f64m4,
                                                                       vector_length);
            vfloat64m4_t running_sum_f64m4 = __riscv_vfadd_vv_f64m4_tu(inner_f64m4, inner_f64m4, corrected_term_f64m4,
                                                                       vector_length);
            compensation_f64m4 = __riscv_vfsub_vv_f64m4_tu(
                compensation_f64m4, __riscv_vfsub_vv_f64m4(running_sum_f64m4, inner_f64m4, vector_length),
                corrected_term_f64m4, vector_length);
            inner_f64m4 = running_sum_f64m4;
        }
        vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
        nk_f64_t inner_val = __riscv_vfmv_f_s_f64m1_f64(
            __riscv_vfredusum_vs_f64m4_f64m1(inner_f64m4, zero_f64m1, vlmax));
        nk_f64_t product_outer = diff_i * inner_val;
        nk_f64_t old_sum = __riscv_vfmv_f_s_f64m1_f64(sum_f64m1);
        nk_f64_t new_sum = old_sum + product_outer;
        if (nk_f64_abs_(old_sum) >= nk_f64_abs_(product_outer))
            outer_compensation += (old_sum - new_sum) + product_outer;
        else outer_compensation += (product_outer - new_sum) + old_sum;
        sum_f64m1 = __riscv_vfmv_v_f_f64m1(new_sum, 1);
    }
    nk_f64_t quadratic = __riscv_vfmv_f_s_f64m1_f64(sum_f64m1) + outer_compensation;
    *result = nk_f64_sqrt_rvv(quadratic > 0 ? quadratic : 0);
}

NK_PUBLIC void nk_mahalanobis_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                      nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t i = 0; i < n; ++i) {
        nk_f32_t a_i, b_i;
        nk_f16_to_f32_serial(a + i, &a_i);
        nk_f16_to_f32_serial(b + i, &b_i);
        nk_f32_t diff_i = a_i - b_i;

        vfloat32m2_t inner_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        nk_f16_t const *c_row = c + i * n;
        nk_size_t remaining = n;
        for (nk_size_t vector_length; remaining > 0; remaining -= vector_length, c_row += vector_length) {
            vector_length = __riscv_vsetvl_e16m1(remaining);
            nk_size_t j = n - remaining;
            vuint16m1_t vc_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)c_row, vector_length);
            vuint16m1_t va_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)(a + j), vector_length);
            vuint16m1_t vb_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)(b + j), vector_length);
            vfloat32m2_t vc_f32m2 = nk_f16m1_to_f32m2_rvv_(vc_u16m1, vector_length);
            vfloat32m2_t va_f32m2 = nk_f16m1_to_f32m2_rvv_(va_u16m1, vector_length);
            vfloat32m2_t vb_f32m2 = nk_f16m1_to_f32m2_rvv_(vb_u16m1, vector_length);
            vfloat32m2_t diff_j_f32m2 = __riscv_vfsub_vv_f32m2(va_f32m2, vb_f32m2, vector_length);
            inner_f32m2 = __riscv_vfmacc_vv_f32m2_tu(inner_f32m2, vc_f32m2, diff_j_f32m2, vector_length);
        }
        vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        nk_f32_t inner_val = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m2_f32m1(inner_f32m2, zero_f32m1, vlmax));
        sum_f32m1 = __riscv_vfmv_v_f_f32m1(__riscv_vfmv_f_s_f32m1_f32(sum_f32m1) + diff_i * inner_val, 1);
    }
    nk_f32_t quadratic_f16 = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
    *result = nk_f32_sqrt_rvv(quadratic_f16 > 0 ? quadratic_f16 : 0);
}

NK_PUBLIC void nk_mahalanobis_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                       nk_f32_t *result) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (nk_size_t i = 0; i < n; ++i) {
        nk_f32_t a_i, b_i;
        nk_bf16_to_f32_serial(a + i, &a_i);
        nk_bf16_to_f32_serial(b + i, &b_i);
        nk_f32_t diff_i = a_i - b_i;

        vfloat32m2_t inner_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        nk_bf16_t const *c_row = c + i * n;
        nk_size_t remaining = n;
        for (nk_size_t vector_length; remaining > 0; remaining -= vector_length, c_row += vector_length) {
            vector_length = __riscv_vsetvl_e16m1(remaining);
            nk_size_t j = n - remaining;
            vuint16m1_t vc_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)c_row, vector_length);
            vuint16m1_t va_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)(a + j), vector_length);
            vuint16m1_t vb_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)(b + j), vector_length);
            vfloat32m2_t vc_f32m2 = nk_bf16m1_to_f32m2_rvv_(vc_u16m1, vector_length);
            vfloat32m2_t va_f32m2 = nk_bf16m1_to_f32m2_rvv_(va_u16m1, vector_length);
            vfloat32m2_t vb_f32m2 = nk_bf16m1_to_f32m2_rvv_(vb_u16m1, vector_length);
            vfloat32m2_t diff_j_f32m2 = __riscv_vfsub_vv_f32m2(va_f32m2, vb_f32m2, vector_length);
            inner_f32m2 = __riscv_vfmacc_vv_f32m2_tu(inner_f32m2, vc_f32m2, diff_j_f32m2, vector_length);
        }
        vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        nk_f32_t inner_val = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m2_f32m1(inner_f32m2, zero_f32m1, vlmax));
        sum_f32m1 = __riscv_vfmv_v_f_f32m1(__riscv_vfmv_f_s_f32m1_f32(sum_f32m1) + diff_i * inner_val, 1);
    }
    nk_f32_t quadratic_bf16 = __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
    *result = nk_f32_sqrt_rvv(quadratic_bf16 > 0 ? quadratic_bf16 : 0);
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
#endif // NK_CURVED_RVV_H
