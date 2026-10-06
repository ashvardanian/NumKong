/**
 *  @brief SIMD-accelerated Elementwise Arithmetic for RISC-V.
 *  @file include/numkong/each/rvv.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/each.h
 */
#ifndef NK_EACH_RVV_H
#define NK_EACH_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/cast/rvv.h"

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_each_sum_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        vfloat64m4_t a_f64m4 = __riscv_vle64_v_f64m4(a, vector_length);
        vfloat64m4_t b_f64m4 = __riscv_vle64_v_f64m4(b, vector_length);
        __riscv_vse64_v_f64m4(result, __riscv_vfadd_vv_f64m4(a_f64m4, b_f64m4, vector_length), vector_length);
    }
}

NK_PUBLIC void nk_each_sum_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t a_f32m4 = __riscv_vle32_v_f32m4(a, vector_length);
        vfloat32m4_t b_f32m4 = __riscv_vle32_v_f32m4(b, vector_length);
        __riscv_vse32_v_f32m4(result, __riscv_vfadd_vv_f32m4(a_f32m4, b_f32m4, vector_length), vector_length);
    }
}

NK_PUBLIC void nk_each_sum_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result) {
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b, vector_length);
        vfloat32m2_t a_f32m2 = nk_f16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_f16m1_to_f32m2_rvv_(b_u16m1, vector_length);
        vfloat32m2_t result_f32m2 = __riscv_vfadd_vv_f32m2(a_f32m2, b_f32m2, vector_length);
        vuint16m1_t result_u16m1 = nk_f32m2_to_f16m1_rvv_(result_f32m2, vector_length);
        __riscv_vse16_v_u16m1((nk_u16_t *)result, result_u16m1, vector_length);
    }
}

NK_PUBLIC void nk_each_sum_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result) {
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b, vector_length);
        vfloat32m2_t a_f32m2 = nk_bf16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_bf16m1_to_f32m2_rvv_(b_u16m1, vector_length);
        vfloat32m2_t result_f32m2 = __riscv_vfadd_vv_f32m2(a_f32m2, b_f32m2, vector_length);
        vuint16m1_t result_u16m1 = nk_f32m2_to_bf16m1_rvv_(result_f32m2, vector_length);
        __riscv_vse16_v_u16m1((nk_u16_t *)result, result_u16m1, vector_length);
    }
}

NK_PUBLIC void nk_each_sum_i8_rvv(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_i8_t *result) {
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e8m4(n);
        vint8m4_t a_i8m4 = __riscv_vle8_v_i8m4(a, vector_length);
        vint8m4_t b_i8m4 = __riscv_vle8_v_i8m4(b, vector_length);
        __riscv_vse8_v_i8m4(result, __riscv_vsadd_vv_i8m4(a_i8m4, b_i8m4, vector_length), vector_length);
    }
}

NK_PUBLIC void nk_each_sum_u8_rvv(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u8_t *result) {
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e8m4(n);
        vuint8m4_t a_u8m4 = __riscv_vle8_v_u8m4(a, vector_length);
        vuint8m4_t b_u8m4 = __riscv_vle8_v_u8m4(b, vector_length);
        __riscv_vse8_v_u8m4(result, __riscv_vsaddu_vv_u8m4(a_u8m4, b_u8m4, vector_length), vector_length);
    }
}

NK_PUBLIC void nk_each_sum_i16_rvv(nk_i16_t const *a, nk_i16_t const *b, nk_size_t n, nk_i16_t *result) {
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e16m4(n);
        vint16m4_t a_i16m4 = __riscv_vle16_v_i16m4(a, vector_length);
        vint16m4_t b_i16m4 = __riscv_vle16_v_i16m4(b, vector_length);
        __riscv_vse16_v_i16m4(result, __riscv_vsadd_vv_i16m4(a_i16m4, b_i16m4, vector_length), vector_length);
    }
}

NK_PUBLIC void nk_each_sum_u16_rvv(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_u16_t *result) {
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e16m4(n);
        vuint16m4_t a_u16m4 = __riscv_vle16_v_u16m4(a, vector_length);
        vuint16m4_t b_u16m4 = __riscv_vle16_v_u16m4(b, vector_length);
        __riscv_vse16_v_u16m4(result, __riscv_vsaddu_vv_u16m4(a_u16m4, b_u16m4, vector_length), vector_length);
    }
}

NK_PUBLIC void nk_each_sum_i32_rvv(nk_i32_t const *a, nk_i32_t const *b, nk_size_t n, nk_i32_t *result) {
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e32m4(n);
        vint32m4_t a_i32m4 = __riscv_vle32_v_i32m4(a, vector_length);
        vint32m4_t b_i32m4 = __riscv_vle32_v_i32m4(b, vector_length);
        __riscv_vse32_v_i32m4(result, __riscv_vsadd_vv_i32m4(a_i32m4, b_i32m4, vector_length), vector_length);
    }
}

NK_PUBLIC void nk_each_sum_u32_rvv(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_u32_t *result) {
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e32m4(n);
        vuint32m4_t a_u32m4 = __riscv_vle32_v_u32m4(a, vector_length);
        vuint32m4_t b_u32m4 = __riscv_vle32_v_u32m4(b, vector_length);
        __riscv_vse32_v_u32m4(result, __riscv_vsaddu_vv_u32m4(a_u32m4, b_u32m4, vector_length), vector_length);
    }
}

NK_PUBLIC void nk_each_sum_i64_rvv(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_i64_t *result) {
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        vint64m4_t a_i64m4 = __riscv_vle64_v_i64m4(a, vector_length);
        vint64m4_t b_i64m4 = __riscv_vle64_v_i64m4(b, vector_length);
        __riscv_vse64_v_i64m4(result, __riscv_vsadd_vv_i64m4(a_i64m4, b_i64m4, vector_length), vector_length);
    }
}

NK_PUBLIC void nk_each_sum_u64_rvv(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_u64_t *result) {
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        vuint64m4_t a_u64m4 = __riscv_vle64_v_u64m4(a, vector_length);
        vuint64m4_t b_u64m4 = __riscv_vle64_v_u64m4(b, vector_length);
        __riscv_vse64_v_u64m4(result, __riscv_vsaddu_vv_u64m4(a_u64m4, b_u64m4, vector_length), vector_length);
    }
}

NK_PUBLIC void nk_each_sum_e4m3_rvv(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_e4m3_t *result) {
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b, vector_length);
        vfloat32m4_t a_f32m4 = nk_e4m3m1_to_f32m4_rvv_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e4m3m1_to_f32m4_rvv_(b_u8m1, vector_length);
        vfloat32m4_t result_f32m4 = __riscv_vfadd_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        vuint8m1_t result_u8m1 = nk_f32m4_to_e4m3m1_rvv_(result_f32m4, vector_length);
        __riscv_vse8_v_u8m1((nk_u8_t *)result, result_u8m1, vector_length);
    }
}

NK_PUBLIC void nk_each_sum_e5m2_rvv(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_e5m2_t *result) {
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b, vector_length);
        vfloat32m4_t a_f32m4 = nk_e5m2m1_to_f32m4_rvv_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e5m2m1_to_f32m4_rvv_(b_u8m1, vector_length);
        vfloat32m4_t result_f32m4 = __riscv_vfadd_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        vuint8m1_t result_u8m1 = nk_f32m4_to_e5m2m1_rvv_(result_f32m4, vector_length);
        __riscv_vse8_v_u8m1((nk_u8_t *)result, result_u8m1, vector_length);
    }
}

NK_PUBLIC void nk_each_scale_f64_rvv(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                     nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha, beta_val = *beta;
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t beta_f64m4 = __riscv_vfmv_v_f_f64m4(beta_val, vlmax);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        vfloat64m4_t a_f64m4 = __riscv_vle64_v_f64m4(a, vector_length);
        a_f64m4 = __riscv_vfmadd_vf_f64m4(a_f64m4, alpha_val, beta_f64m4, vector_length);
        __riscv_vse64_v_f64m4(result, a_f64m4, vector_length);
    }
}

NK_PUBLIC void nk_each_scale_f32_rvv(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                     nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t beta_f32m4 = __riscv_vfmv_v_f_f32m4(beta_val, vlmax);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t a_f32m4 = __riscv_vle32_v_f32m4(a, vector_length);
        a_f32m4 = __riscv_vfmadd_vf_f32m4(a_f32m4, alpha_val, beta_f32m4, vector_length);
        __riscv_vse32_v_f32m4(result, a_f32m4, vector_length);
    }
}

NK_PUBLIC void nk_each_scale_f16_rvv(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                     nk_f16_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t beta_f32m2 = __riscv_vfmv_v_f_f32m2(beta_val, vlmax);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a, vector_length);
        vfloat32m2_t a_f32m2 = nk_f16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        a_f32m2 = __riscv_vfmadd_vf_f32m2(a_f32m2, alpha_val, beta_f32m2, vector_length);
        vuint16m1_t result_u16m1 = nk_f32m2_to_f16m1_rvv_(a_f32m2, vector_length);
        __riscv_vse16_v_u16m1((nk_u16_t *)result, result_u16m1, vector_length);
    }
}

NK_PUBLIC void nk_each_scale_bf16_rvv(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                      nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t beta_f32m2 = __riscv_vfmv_v_f_f32m2(beta_val, vlmax);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a, vector_length);
        vfloat32m2_t a_f32m2 = nk_bf16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        a_f32m2 = __riscv_vfmadd_vf_f32m2(a_f32m2, alpha_val, beta_f32m2, vector_length);
        vuint16m1_t result_u16m1 = nk_f32m2_to_bf16m1_rvv_(a_f32m2, vector_length);
        __riscv_vse16_v_u16m1((nk_u16_t *)result, result_u16m1, vector_length);
    }
}

NK_PUBLIC void nk_each_scale_i8_rvv(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t beta_f32m4 = __riscv_vfmv_v_f_f32m4(beta_val, vlmax);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n);
        vint8m1_t a_i8m1 = __riscv_vle8_v_i8m1(a, vector_length);
        vint16m2_t a_i16m2 = __riscv_vwadd_vx_i16m2(a_i8m1, 0, vector_length);
        vint32m4_t a_i32m4 = __riscv_vwadd_vx_i32m4(a_i16m2, 0, vector_length);
        vfloat32m4_t a_f32m4 = __riscv_vfcvt_f_x_v_f32m4(a_i32m4, vector_length);
        a_f32m4 = __riscv_vfmadd_vf_f32m4(a_f32m4, alpha_val, beta_f32m4, vector_length);
        vint32m4_t result_i32m4 = __riscv_vfcvt_x_f_v_i32m4(a_f32m4, vector_length);
        result_i32m4 = __riscv_vmax_vx_i32m4(result_i32m4, -128, vector_length);
        result_i32m4 = __riscv_vmin_vx_i32m4(result_i32m4, 127, vector_length);
        vint16m2_t result_i16m2 = __riscv_vncvt_x_x_w_i16m2(result_i32m4, vector_length);
        vint8m1_t result_i8m1 = __riscv_vncvt_x_x_w_i8m1(result_i16m2, vector_length);
        __riscv_vse8_v_i8m1(result, result_i8m1, vector_length);
    }
}

NK_PUBLIC void nk_each_scale_u8_rvv(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t beta_f32m4 = __riscv_vfmv_v_f_f32m4(beta_val, vlmax);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1(a, vector_length);
        vuint16m2_t a_u16m2 = __riscv_vwaddu_vx_u16m2(a_u8m1, 0, vector_length);
        vuint32m4_t a_u32m4 = __riscv_vwaddu_vx_u32m4(a_u16m2, 0, vector_length);
        vfloat32m4_t a_f32m4 = __riscv_vfcvt_f_xu_v_f32m4(a_u32m4, vector_length);
        a_f32m4 = __riscv_vfmadd_vf_f32m4(a_f32m4, alpha_val, beta_f32m4, vector_length);
        vuint32m4_t result_u32m4 = __riscv_vfcvt_xu_f_v_u32m4(a_f32m4, vector_length);
        result_u32m4 = __riscv_vmaxu_vx_u32m4(result_u32m4, 0, vector_length);
        result_u32m4 = __riscv_vminu_vx_u32m4(result_u32m4, 255, vector_length);
        vuint16m2_t result_u16m2 = __riscv_vncvt_x_x_w_u16m2(result_u32m4, vector_length);
        vuint8m1_t result_u8m1 = __riscv_vncvt_x_x_w_u8m1(result_u16m2, vector_length);
        __riscv_vse8_v_u8m1(result, result_u8m1, vector_length);
    }
}

NK_PUBLIC void nk_each_scale_i16_rvv(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                     nk_i16_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t beta_f32m2 = __riscv_vfmv_v_f_f32m2(beta_val, vlmax);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        vint16m1_t a_i16m1 = __riscv_vle16_v_i16m1(a, vector_length);
        vint32m2_t a_i32m2 = __riscv_vwadd_vx_i32m2(a_i16m1, 0, vector_length);
        vfloat32m2_t a_f32m2 = __riscv_vfcvt_f_x_v_f32m2(a_i32m2, vector_length);
        a_f32m2 = __riscv_vfmadd_vf_f32m2(a_f32m2, alpha_val, beta_f32m2, vector_length);
        vint32m2_t result_i32m2 = __riscv_vfcvt_x_f_v_i32m2(a_f32m2, vector_length);
        result_i32m2 = __riscv_vmax_vx_i32m2(result_i32m2, -32768, vector_length);
        result_i32m2 = __riscv_vmin_vx_i32m2(result_i32m2, 32767, vector_length);
        vint16m1_t result_i16m1 = __riscv_vncvt_x_x_w_i16m1(result_i32m2, vector_length);
        __riscv_vse16_v_i16m1(result, result_i16m1, vector_length);
    }
}

NK_PUBLIC void nk_each_scale_u16_rvv(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                     nk_u16_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    nk_size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t beta_f32m2 = __riscv_vfmv_v_f_f32m2(beta_val, vlmax);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1(a, vector_length);
        vuint32m2_t a_u32m2 = __riscv_vwaddu_vx_u32m2(a_u16m1, 0, vector_length);
        vfloat32m2_t a_f32m2 = __riscv_vfcvt_f_xu_v_f32m2(a_u32m2, vector_length);
        a_f32m2 = __riscv_vfmadd_vf_f32m2(a_f32m2, alpha_val, beta_f32m2, vector_length);
        vuint32m2_t result_u32m2 = __riscv_vfcvt_xu_f_v_u32m2(a_f32m2, vector_length);
        result_u32m2 = __riscv_vmaxu_vx_u32m2(result_u32m2, 0, vector_length);
        result_u32m2 = __riscv_vminu_vx_u32m2(result_u32m2, 65535, vector_length);
        vuint16m1_t result_u16m1 = __riscv_vncvt_x_x_w_u16m1(result_u32m2, vector_length);
        __riscv_vse16_v_u16m1(result, result_u16m1, vector_length);
    }
}

NK_PUBLIC void nk_each_scale_i32_rvv(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                     nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha, beta_val = *beta;
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t beta_f64m2 = __riscv_vfmv_v_f_f64m2(beta_val, vlmax);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(n);
        vint32m1_t a_i32m1 = __riscv_vle32_v_i32m1(a, vector_length);
        vfloat64m2_t a_f64m2 = __riscv_vfwcvt_f_x_v_f64m2(a_i32m1, vector_length);
        a_f64m2 = __riscv_vfmadd_vf_f64m2(a_f64m2, alpha_val, beta_f64m2, vector_length);
        a_f64m2 = __riscv_vfmax_vf_f64m2(a_f64m2, -2147483648.0, vector_length);
        a_f64m2 = __riscv_vfmin_vf_f64m2(a_f64m2, 2147483647.0, vector_length);
        vint32m1_t result_i32m1 = __riscv_vfncvt_x_f_w_i32m1(a_f64m2, vector_length);
        __riscv_vse32_v_i32m1(result, result_i32m1, vector_length);
    }
}

NK_PUBLIC void nk_each_scale_u32_rvv(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                     nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha, beta_val = *beta;
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t beta_f64m2 = __riscv_vfmv_v_f_f64m2(beta_val, vlmax);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(n);
        vuint32m1_t a_u32m1 = __riscv_vle32_v_u32m1(a, vector_length);
        vfloat64m2_t a_f64m2 = __riscv_vfwcvt_f_xu_v_f64m2(a_u32m1, vector_length);
        a_f64m2 = __riscv_vfmadd_vf_f64m2(a_f64m2, alpha_val, beta_f64m2, vector_length);
        a_f64m2 = __riscv_vfmax_vf_f64m2(a_f64m2, 0.0, vector_length);
        a_f64m2 = __riscv_vfmin_vf_f64m2(a_f64m2, 4294967295.0, vector_length);
        vuint32m1_t result_u32m1 = __riscv_vfncvt_xu_f_w_u32m1(a_f64m2, vector_length);
        __riscv_vse32_v_u32m1(result, result_u32m1, vector_length);
    }
}

NK_PUBLIC void nk_each_scale_i64_rvv(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                     nk_i64_t *result) {
    nk_f64_t alpha_val = *alpha, beta_val = *beta;
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t beta_f64m4 = __riscv_vfmv_v_f_f64m4(beta_val, vlmax);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        vint64m4_t a_i64m4 = __riscv_vle64_v_i64m4(a, vector_length);
        vfloat64m4_t a_f64m4 = __riscv_vfcvt_f_x_v_f64m4(a_i64m4, vector_length);
        a_f64m4 = __riscv_vfmadd_vf_f64m4(a_f64m4, alpha_val, beta_f64m4, vector_length);
        vint64m4_t result_i64m4 = __riscv_vfcvt_x_f_v_i64m4(a_f64m4, vector_length);
        __riscv_vse64_v_i64m4(result, result_i64m4, vector_length);
    }
}

NK_PUBLIC void nk_each_scale_u64_rvv(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                     nk_u64_t *result) {
    nk_f64_t alpha_val = *alpha, beta_val = *beta;
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t beta_f64m4 = __riscv_vfmv_v_f_f64m4(beta_val, vlmax);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        vuint64m4_t a_u64m4 = __riscv_vle64_v_u64m4(a, vector_length);
        vfloat64m4_t a_f64m4 = __riscv_vfcvt_f_xu_v_f64m4(a_u64m4, vector_length);
        a_f64m4 = __riscv_vfmadd_vf_f64m4(a_f64m4, alpha_val, beta_f64m4, vector_length);
        a_f64m4 = __riscv_vfmax_vf_f64m4(a_f64m4, 0.0, vector_length);
        vuint64m4_t result_u64m4 = __riscv_vfcvt_xu_f_v_u64m4(a_f64m4, vector_length);
        __riscv_vse64_v_u64m4(result, result_u64m4, vector_length);
    }
}

NK_PUBLIC void nk_each_scale_e4m3_rvv(nk_e4m3_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                      nk_e4m3_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t beta_f32m4 = __riscv_vfmv_v_f_f32m4(beta_val, vlmax);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a, vector_length);
        vfloat32m4_t a_f32m4 = nk_e4m3m1_to_f32m4_rvv_(a_u8m1, vector_length);
        a_f32m4 = __riscv_vfmadd_vf_f32m4(a_f32m4, alpha_val, beta_f32m4, vector_length);
        vuint8m1_t result_u8m1 = nk_f32m4_to_e4m3m1_rvv_(a_f32m4, vector_length);
        __riscv_vse8_v_u8m1((nk_u8_t *)result, result_u8m1, vector_length);
    }
}

NK_PUBLIC void nk_each_scale_e5m2_rvv(nk_e5m2_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                      nk_e5m2_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t beta_f32m4 = __riscv_vfmv_v_f_f32m4(beta_val, vlmax);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a, vector_length);
        vfloat32m4_t a_f32m4 = nk_e5m2m1_to_f32m4_rvv_(a_u8m1, vector_length);
        a_f32m4 = __riscv_vfmadd_vf_f32m4(a_f32m4, alpha_val, beta_f32m4, vector_length);
        vuint8m1_t result_u8m1 = nk_f32m4_to_e5m2m1_rvv_(a_f32m4, vector_length);
        __riscv_vse8_v_u8m1((nk_u8_t *)result, result_u8m1, vector_length);
    }
}

NK_PUBLIC void nk_each_blend_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t const *alpha,
                                     nk_f64_t const *beta, nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha, beta_val = *beta;
    if (alpha_val == 1 && beta_val == 1) {
        nk_each_sum_f64_rvv(a, b, n, result);
        return;
    }
    else if (alpha_val == 0 || beta_val == 0) {
        nk_f64_t zero = 0;
        if (beta_val == 0) { nk_each_scale_f64_rvv(a, n, alpha, &zero, result); }
        else { nk_each_scale_f64_rvv(b, n, beta, &zero, result); }
        return;
    }
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        vfloat64m4_t a_f64m4 = __riscv_vle64_v_f64m4(a, vector_length);
        vfloat64m4_t b_f64m4 = __riscv_vle64_v_f64m4(b, vector_length);
        vfloat64m4_t result_f64m4 = __riscv_vfmul_vf_f64m4(a_f64m4, alpha_val, vector_length);
        result_f64m4 = __riscv_vfmacc_vf_f64m4(result_f64m4, beta_val, b_f64m4, vector_length);
        __riscv_vse64_v_f64m4(result, result_f64m4, vector_length);
    }
}

NK_PUBLIC void nk_each_blend_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                     nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    if (alpha_val == 1 && beta_val == 1) {
        nk_each_sum_f32_rvv(a, b, n, result);
        return;
    }
    else if (alpha_val == 0 || beta_val == 0) {
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_f32_rvv(a, n, alpha, &zero, result); }
        else { nk_each_scale_f32_rvv(b, n, beta, &zero, result); }
        return;
    }
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t a_f32m4 = __riscv_vle32_v_f32m4(a, vector_length);
        vfloat32m4_t b_f32m4 = __riscv_vle32_v_f32m4(b, vector_length);
        vfloat32m4_t a_scaled_f32m4 = __riscv_vfmul_vf_f32m4(a_f32m4, alpha_val, vector_length);
        vfloat32m4_t result_f32m4 = __riscv_vfmacc_vf_f32m4(a_scaled_f32m4, beta_val, b_f32m4, vector_length);
        __riscv_vse32_v_f32m4(result, result_f32m4, vector_length);
    }
}

NK_PUBLIC void nk_each_blend_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                     nk_f32_t const *beta, nk_f16_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    if (alpha_val == 1 && beta_val == 1) {
        nk_each_sum_f16_rvv(a, b, n, result);
        return;
    }
    else if (alpha_val == 0 || beta_val == 0) {
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_f16_rvv(a, n, alpha, &zero, result); }
        else { nk_each_scale_f16_rvv(b, n, beta, &zero, result); }
        return;
    }
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b, vector_length);
        vfloat32m2_t a_f32m2 = nk_f16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_f16m1_to_f32m2_rvv_(b_u16m1, vector_length);
        vfloat32m2_t result_f32m2 = __riscv_vfmul_vf_f32m2(a_f32m2, alpha_val, vector_length);
        result_f32m2 = __riscv_vfmacc_vf_f32m2(result_f32m2, beta_val, b_f32m2, vector_length);
        vuint16m1_t result_u16m1 = nk_f32m2_to_f16m1_rvv_(result_f32m2, vector_length);
        __riscv_vse16_v_u16m1((nk_u16_t *)result, result_u16m1, vector_length);
    }
}

NK_PUBLIC void nk_each_blend_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                      nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    if (alpha_val == 1 && beta_val == 1) {
        nk_each_sum_bf16_rvv(a, b, n, result);
        return;
    }
    else if (alpha_val == 0 || beta_val == 0) {
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_bf16_rvv(a, n, alpha, &zero, result); }
        else { nk_each_scale_bf16_rvv(b, n, beta, &zero, result); }
        return;
    }
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b, vector_length);
        vfloat32m2_t a_f32m2 = nk_bf16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_bf16m1_to_f32m2_rvv_(b_u16m1, vector_length);
        vfloat32m2_t result_f32m2 = __riscv_vfmul_vf_f32m2(a_f32m2, alpha_val, vector_length);
        result_f32m2 = __riscv_vfmacc_vf_f32m2(result_f32m2, beta_val, b_f32m2, vector_length);
        vuint16m1_t result_u16m1 = nk_f32m2_to_bf16m1_rvv_(result_f32m2, vector_length);
        __riscv_vse16_v_u16m1((nk_u16_t *)result, result_u16m1, vector_length);
    }
}

NK_PUBLIC void nk_each_blend_i8_rvv(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                    nk_f32_t const *beta, nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    if (alpha_val == 1 && beta_val == 1) {
        nk_each_sum_i8_rvv(a, b, n, result);
        return;
    }
    else if (alpha_val == 0 || beta_val == 0) {
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_i8_rvv(a, n, alpha, &zero, result); }
        else { nk_each_scale_i8_rvv(b, n, beta, &zero, result); }
        return;
    }
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n);
        vint8m1_t a_i8m1 = __riscv_vle8_v_i8m1(a, vector_length);
        vint8m1_t b_i8m1 = __riscv_vle8_v_i8m1(b, vector_length);
        vint16m2_t a_i16m2 = __riscv_vwadd_vx_i16m2(a_i8m1, 0, vector_length);
        vint32m4_t a_i32m4 = __riscv_vwadd_vx_i32m4(a_i16m2, 0, vector_length);
        vfloat32m4_t a_f32m4 = __riscv_vfcvt_f_x_v_f32m4(a_i32m4, vector_length);
        vint16m2_t b_i16m2 = __riscv_vwadd_vx_i16m2(b_i8m1, 0, vector_length);
        vint32m4_t b_i32m4 = __riscv_vwadd_vx_i32m4(b_i16m2, 0, vector_length);
        vfloat32m4_t b_f32m4 = __riscv_vfcvt_f_x_v_f32m4(b_i32m4, vector_length);
        vfloat32m4_t result_f32m4 = __riscv_vfmul_vf_f32m4(a_f32m4, alpha_val, vector_length);
        result_f32m4 = __riscv_vfmacc_vf_f32m4(result_f32m4, beta_val, b_f32m4, vector_length);
        vint32m4_t result_i32m4 = __riscv_vfcvt_x_f_v_i32m4(result_f32m4, vector_length);
        result_i32m4 = __riscv_vmax_vx_i32m4(result_i32m4, -128, vector_length);
        result_i32m4 = __riscv_vmin_vx_i32m4(result_i32m4, 127, vector_length);
        vint16m2_t result_i16m2 = __riscv_vncvt_x_x_w_i16m2(result_i32m4, vector_length);
        vint8m1_t result_i8m1 = __riscv_vncvt_x_x_w_i8m1(result_i16m2, vector_length);
        __riscv_vse8_v_i8m1(result, result_i8m1, vector_length);
    }
}

NK_PUBLIC void nk_each_blend_u8_rvv(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                    nk_f32_t const *beta, nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    if (alpha_val == 1 && beta_val == 1) {
        nk_each_sum_u8_rvv(a, b, n, result);
        return;
    }
    else if (alpha_val == 0 || beta_val == 0) {
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_each_scale_u8_rvv(a, n, alpha, &zero, result); }
        else { nk_each_scale_u8_rvv(b, n, beta, &zero, result); }
        return;
    }
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1(a, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1(b, vector_length);
        vuint16m2_t a_u16m2 = __riscv_vwaddu_vx_u16m2(a_u8m1, 0, vector_length);
        vuint32m4_t a_u32m4 = __riscv_vwaddu_vx_u32m4(a_u16m2, 0, vector_length);
        vfloat32m4_t a_f32m4 = __riscv_vfcvt_f_xu_v_f32m4(a_u32m4, vector_length);
        vuint16m2_t b_u16m2 = __riscv_vwaddu_vx_u16m2(b_u8m1, 0, vector_length);
        vuint32m4_t b_u32m4 = __riscv_vwaddu_vx_u32m4(b_u16m2, 0, vector_length);
        vfloat32m4_t b_f32m4 = __riscv_vfcvt_f_xu_v_f32m4(b_u32m4, vector_length);
        vfloat32m4_t result_f32m4 = __riscv_vfmul_vf_f32m4(a_f32m4, alpha_val, vector_length);
        result_f32m4 = __riscv_vfmacc_vf_f32m4(result_f32m4, beta_val, b_f32m4, vector_length);
        vuint32m4_t result_u32m4 = __riscv_vfcvt_xu_f_v_u32m4(result_f32m4, vector_length);
        result_u32m4 = __riscv_vmaxu_vx_u32m4(result_u32m4, 0, vector_length);
        result_u32m4 = __riscv_vminu_vx_u32m4(result_u32m4, 255, vector_length);
        vuint16m2_t result_u16m2 = __riscv_vncvt_x_x_w_u16m2(result_u32m4, vector_length);
        vuint8m1_t result_u8m1 = __riscv_vncvt_x_x_w_u8m1(result_u16m2, vector_length);
        __riscv_vse8_v_u8m1(result, result_u8m1, vector_length);
    }
}

NK_PUBLIC void nk_each_blend_e4m3_rvv(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                      nk_f32_t const *beta, nk_e4m3_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b, vector_length);
        vfloat32m4_t a_f32m4 = nk_e4m3m1_to_f32m4_rvv_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e4m3m1_to_f32m4_rvv_(b_u8m1, vector_length);
        vfloat32m4_t result_f32m4 = __riscv_vfmul_vf_f32m4(a_f32m4, alpha_val, vector_length);
        result_f32m4 = __riscv_vfmacc_vf_f32m4(result_f32m4, beta_val, b_f32m4, vector_length);
        vuint8m1_t result_u8m1 = nk_f32m4_to_e4m3m1_rvv_(result_f32m4, vector_length);
        __riscv_vse8_v_u8m1((nk_u8_t *)result, result_u8m1, vector_length);
    }
}

NK_PUBLIC void nk_each_blend_e5m2_rvv(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t const *alpha,
                                      nk_f32_t const *beta, nk_e5m2_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b, vector_length);
        vfloat32m4_t a_f32m4 = nk_e5m2m1_to_f32m4_rvv_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e5m2m1_to_f32m4_rvv_(b_u8m1, vector_length);
        vfloat32m4_t result_f32m4 = __riscv_vfmul_vf_f32m4(a_f32m4, alpha_val, vector_length);
        result_f32m4 = __riscv_vfmacc_vf_f32m4(result_f32m4, beta_val, b_f32m4, vector_length);
        vuint8m1_t result_u8m1 = nk_f32m4_to_e5m2m1_rvv_(result_f32m4, vector_length);
        __riscv_vse8_v_u8m1((nk_u8_t *)result, result_u8m1, vector_length);
    }
}

NK_PUBLIC void nk_each_fma_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                   nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha, beta_val = *beta;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, c += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        vfloat64m4_t a_f64m4 = __riscv_vle64_v_f64m4(a, vector_length);
        vfloat64m4_t b_f64m4 = __riscv_vle64_v_f64m4(b, vector_length);
        vfloat64m4_t c_f64m4 = __riscv_vle64_v_f64m4(c, vector_length);
        vfloat64m4_t product_f64m4 = __riscv_vfmul_vv_f64m4(a_f64m4, b_f64m4, vector_length);
        vfloat64m4_t scaled_product_f64m4 = __riscv_vfmul_vf_f64m4(product_f64m4, alpha_val, vector_length);
        vfloat64m4_t result_f64m4 = __riscv_vfmacc_vf_f64m4(scaled_product_f64m4, beta_val, c_f64m4, vector_length);
        __riscv_vse64_v_f64m4(result, result_f64m4, vector_length);
    }
}

NK_PUBLIC void nk_each_fma_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                   nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, c += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t a_f32m4 = __riscv_vle32_v_f32m4(a, vector_length);
        vfloat32m4_t b_f32m4 = __riscv_vle32_v_f32m4(b, vector_length);
        vfloat32m4_t c_f32m4 = __riscv_vle32_v_f32m4(c, vector_length);
        vfloat32m4_t product_f32m4 = __riscv_vfmul_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        vfloat32m4_t scaled_product_f32m4 = __riscv_vfmul_vf_f32m4(product_f32m4, alpha_val, vector_length);
        vfloat32m4_t result_f32m4 = __riscv_vfmacc_vf_f32m4(scaled_product_f32m4, beta_val, c_f32m4, vector_length);
        __riscv_vse32_v_f32m4(result, result_f32m4, vector_length);
    }
}

NK_PUBLIC void nk_each_fma_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                   nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, c += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b, vector_length);
        vuint16m1_t c_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)c, vector_length);
        vfloat32m2_t a_f32m2 = nk_f16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_f16m1_to_f32m2_rvv_(b_u16m1, vector_length);
        vfloat32m2_t c_f32m2 = nk_f16m1_to_f32m2_rvv_(c_u16m1, vector_length);
        vfloat32m2_t product_f32m2 = __riscv_vfmul_vv_f32m2(a_f32m2, b_f32m2, vector_length);
        vfloat32m2_t scaled_product_f32m2 = __riscv_vfmul_vf_f32m2(product_f32m2, alpha_val, vector_length);
        vfloat32m2_t result_f32m2 = __riscv_vfmacc_vf_f32m2(scaled_product_f32m2, beta_val, c_f32m2, vector_length);
        vuint16m1_t result_u16m1 = nk_f32m2_to_f16m1_rvv_(result_f32m2, vector_length);
        __riscv_vse16_v_u16m1((nk_u16_t *)result, result_u16m1, vector_length);
    }
}

NK_PUBLIC void nk_each_fma_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                    nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, c += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b, vector_length);
        vuint16m1_t c_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)c, vector_length);
        vfloat32m2_t a_f32m2 = nk_bf16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_bf16m1_to_f32m2_rvv_(b_u16m1, vector_length);
        vfloat32m2_t c_f32m2 = nk_bf16m1_to_f32m2_rvv_(c_u16m1, vector_length);
        vfloat32m2_t product_f32m2 = __riscv_vfmul_vv_f32m2(a_f32m2, b_f32m2, vector_length);
        vfloat32m2_t scaled_product_f32m2 = __riscv_vfmul_vf_f32m2(product_f32m2, alpha_val, vector_length);
        vfloat32m2_t result_f32m2 = __riscv_vfmacc_vf_f32m2(scaled_product_f32m2, beta_val, c_f32m2, vector_length);
        vuint16m1_t result_u16m1 = nk_f32m2_to_bf16m1_rvv_(result_f32m2, vector_length);
        __riscv_vse16_v_u16m1((nk_u16_t *)result, result_u16m1, vector_length);
    }
}

NK_PUBLIC void nk_each_fma_i8_rvv(nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n,
                                  nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, c += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n);
        vint8m1_t a_i8m1 = __riscv_vle8_v_i8m1(a, vector_length);
        vint8m1_t b_i8m1 = __riscv_vle8_v_i8m1(b, vector_length);
        vint8m1_t c_i8m1 = __riscv_vle8_v_i8m1(c, vector_length);
        vint16m2_t a_i16m2 = __riscv_vwadd_vx_i16m2(a_i8m1, 0, vector_length);
        vint32m4_t a_i32m4 = __riscv_vwadd_vx_i32m4(a_i16m2, 0, vector_length);
        vfloat32m4_t a_f32m4 = __riscv_vfcvt_f_x_v_f32m4(a_i32m4, vector_length);
        vint16m2_t b_i16m2 = __riscv_vwadd_vx_i16m2(b_i8m1, 0, vector_length);
        vint32m4_t b_i32m4 = __riscv_vwadd_vx_i32m4(b_i16m2, 0, vector_length);
        vfloat32m4_t b_f32m4 = __riscv_vfcvt_f_x_v_f32m4(b_i32m4, vector_length);
        vint16m2_t c_i16m2 = __riscv_vwadd_vx_i16m2(c_i8m1, 0, vector_length);
        vint32m4_t c_i32m4 = __riscv_vwadd_vx_i32m4(c_i16m2, 0, vector_length);
        vfloat32m4_t c_f32m4 = __riscv_vfcvt_f_x_v_f32m4(c_i32m4, vector_length);
        vfloat32m4_t product_f32m4 = __riscv_vfmul_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        vfloat32m4_t scaled_product_f32m4 = __riscv_vfmul_vf_f32m4(product_f32m4, alpha_val, vector_length);
        vfloat32m4_t result_f32m4 = __riscv_vfmacc_vf_f32m4(scaled_product_f32m4, beta_val, c_f32m4, vector_length);
        vint32m4_t result_i32m4 = __riscv_vfcvt_x_f_v_i32m4(result_f32m4, vector_length);
        result_i32m4 = __riscv_vmax_vx_i32m4(result_i32m4, -128, vector_length);
        result_i32m4 = __riscv_vmin_vx_i32m4(result_i32m4, 127, vector_length);
        vint16m2_t result_i16m2 = __riscv_vncvt_x_x_w_i16m2(result_i32m4, vector_length);
        vint8m1_t result_i8m1 = __riscv_vncvt_x_x_w_i8m1(result_i16m2, vector_length);
        __riscv_vse8_v_i8m1(result, result_i8m1, vector_length);
    }
}

NK_PUBLIC void nk_each_fma_u8_rvv(nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n,
                                  nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, c += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1(a, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1(b, vector_length);
        vuint8m1_t c_u8m1 = __riscv_vle8_v_u8m1(c, vector_length);
        vuint16m2_t a_u16m2 = __riscv_vwaddu_vx_u16m2(a_u8m1, 0, vector_length);
        vuint32m4_t a_u32m4 = __riscv_vwaddu_vx_u32m4(a_u16m2, 0, vector_length);
        vfloat32m4_t a_f32m4 = __riscv_vfcvt_f_xu_v_f32m4(a_u32m4, vector_length);
        vuint16m2_t b_u16m2 = __riscv_vwaddu_vx_u16m2(b_u8m1, 0, vector_length);
        vuint32m4_t b_u32m4 = __riscv_vwaddu_vx_u32m4(b_u16m2, 0, vector_length);
        vfloat32m4_t b_f32m4 = __riscv_vfcvt_f_xu_v_f32m4(b_u32m4, vector_length);
        vuint16m2_t c_u16m2 = __riscv_vwaddu_vx_u16m2(c_u8m1, 0, vector_length);
        vuint32m4_t c_u32m4 = __riscv_vwaddu_vx_u32m4(c_u16m2, 0, vector_length);
        vfloat32m4_t c_f32m4 = __riscv_vfcvt_f_xu_v_f32m4(c_u32m4, vector_length);
        vfloat32m4_t product_f32m4 = __riscv_vfmul_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        vfloat32m4_t scaled_product_f32m4 = __riscv_vfmul_vf_f32m4(product_f32m4, alpha_val, vector_length);
        vfloat32m4_t result_f32m4 = __riscv_vfmacc_vf_f32m4(scaled_product_f32m4, beta_val, c_f32m4, vector_length);
        vuint32m4_t result_u32m4 = __riscv_vfcvt_xu_f_v_u32m4(result_f32m4, vector_length);
        result_u32m4 = __riscv_vmaxu_vx_u32m4(result_u32m4, 0, vector_length);
        result_u32m4 = __riscv_vminu_vx_u32m4(result_u32m4, 255, vector_length);
        vuint16m2_t result_u16m2 = __riscv_vncvt_x_x_w_u16m2(result_u32m4, vector_length);
        vuint8m1_t result_u8m1 = __riscv_vncvt_x_x_w_u8m1(result_u16m2, vector_length);
        __riscv_vse8_v_u8m1(result, result_u8m1, vector_length);
    }
}

NK_PUBLIC void nk_each_fma_i16_rvv(nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c, nk_size_t n,
                                   nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, c += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        vint16m1_t a_i16m1 = __riscv_vle16_v_i16m1(a, vector_length);
        vint16m1_t b_i16m1 = __riscv_vle16_v_i16m1(b, vector_length);
        vint16m1_t c_i16m1 = __riscv_vle16_v_i16m1(c, vector_length);
        vint32m2_t a_i32m2 = __riscv_vwadd_vx_i32m2(a_i16m1, 0, vector_length);
        vfloat32m2_t a_f32m2 = __riscv_vfcvt_f_x_v_f32m2(a_i32m2, vector_length);
        vint32m2_t b_i32m2 = __riscv_vwadd_vx_i32m2(b_i16m1, 0, vector_length);
        vfloat32m2_t b_f32m2 = __riscv_vfcvt_f_x_v_f32m2(b_i32m2, vector_length);
        vint32m2_t c_i32m2 = __riscv_vwadd_vx_i32m2(c_i16m1, 0, vector_length);
        vfloat32m2_t c_f32m2 = __riscv_vfcvt_f_x_v_f32m2(c_i32m2, vector_length);
        vfloat32m2_t product_f32m2 = __riscv_vfmul_vv_f32m2(a_f32m2, b_f32m2, vector_length);
        vfloat32m2_t scaled_product_f32m2 = __riscv_vfmul_vf_f32m2(product_f32m2, alpha_val, vector_length);
        vfloat32m2_t result_f32m2 = __riscv_vfmacc_vf_f32m2(scaled_product_f32m2, beta_val, c_f32m2, vector_length);
        vint32m2_t result_i32m2 = __riscv_vfcvt_x_f_v_i32m2(result_f32m2, vector_length);
        result_i32m2 = __riscv_vmax_vx_i32m2(result_i32m2, -32768, vector_length);
        result_i32m2 = __riscv_vmin_vx_i32m2(result_i32m2, 32767, vector_length);
        vint16m1_t result_i16m1 = __riscv_vncvt_x_x_w_i16m1(result_i32m2, vector_length);
        __riscv_vse16_v_i16m1(result, result_i16m1, vector_length);
    }
}

NK_PUBLIC void nk_each_fma_u16_rvv(nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c, nk_size_t n,
                                   nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, c += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1(a, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1(b, vector_length);
        vuint16m1_t c_u16m1 = __riscv_vle16_v_u16m1(c, vector_length);
        vuint32m2_t a_u32m2 = __riscv_vwaddu_vx_u32m2(a_u16m1, 0, vector_length);
        vfloat32m2_t a_f32m2 = __riscv_vfcvt_f_xu_v_f32m2(a_u32m2, vector_length);
        vuint32m2_t b_u32m2 = __riscv_vwaddu_vx_u32m2(b_u16m1, 0, vector_length);
        vfloat32m2_t b_f32m2 = __riscv_vfcvt_f_xu_v_f32m2(b_u32m2, vector_length);
        vuint32m2_t c_u32m2 = __riscv_vwaddu_vx_u32m2(c_u16m1, 0, vector_length);
        vfloat32m2_t c_f32m2 = __riscv_vfcvt_f_xu_v_f32m2(c_u32m2, vector_length);
        vfloat32m2_t product_f32m2 = __riscv_vfmul_vv_f32m2(a_f32m2, b_f32m2, vector_length);
        vfloat32m2_t scaled_product_f32m2 = __riscv_vfmul_vf_f32m2(product_f32m2, alpha_val, vector_length);
        vfloat32m2_t result_f32m2 = __riscv_vfmacc_vf_f32m2(scaled_product_f32m2, beta_val, c_f32m2, vector_length);
        vuint32m2_t result_u32m2 = __riscv_vfcvt_xu_f_v_u32m2(result_f32m2, vector_length);
        result_u32m2 = __riscv_vmaxu_vx_u32m2(result_u32m2, 0, vector_length);
        result_u32m2 = __riscv_vminu_vx_u32m2(result_u32m2, 65535, vector_length);
        vuint16m1_t result_u16m1 = __riscv_vncvt_x_x_w_u16m1(result_u32m2, vector_length);
        __riscv_vse16_v_u16m1(result, result_u16m1, vector_length);
    }
}

NK_PUBLIC void nk_each_fma_i32_rvv(nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c, nk_size_t n,
                                   nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha, beta_val = *beta;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, c += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(n);
        vint32m1_t a_i32m1 = __riscv_vle32_v_i32m1(a, vector_length);
        vint32m1_t b_i32m1 = __riscv_vle32_v_i32m1(b, vector_length);
        vint32m1_t c_i32m1 = __riscv_vle32_v_i32m1(c, vector_length);
        vfloat64m2_t a_f64m2 = __riscv_vfwcvt_f_x_v_f64m2(a_i32m1, vector_length);
        vfloat64m2_t b_f64m2 = __riscv_vfwcvt_f_x_v_f64m2(b_i32m1, vector_length);
        vfloat64m2_t c_f64m2 = __riscv_vfwcvt_f_x_v_f64m2(c_i32m1, vector_length);
        vfloat64m2_t product_f64m2 = __riscv_vfmul_vv_f64m2(a_f64m2, b_f64m2, vector_length);
        vfloat64m2_t scaled_product_f64m2 = __riscv_vfmul_vf_f64m2(product_f64m2, alpha_val, vector_length);
        vfloat64m2_t result_f64m2 = __riscv_vfmacc_vf_f64m2(scaled_product_f64m2, beta_val, c_f64m2, vector_length);
        result_f64m2 = __riscv_vfmax_vf_f64m2(result_f64m2, -2147483648.0, vector_length);
        result_f64m2 = __riscv_vfmin_vf_f64m2(result_f64m2, 2147483647.0, vector_length);
        vint32m1_t result_i32m1 = __riscv_vfncvt_x_f_w_i32m1(result_f64m2, vector_length);
        __riscv_vse32_v_i32m1(result, result_i32m1, vector_length);
    }
}

NK_PUBLIC void nk_each_fma_u32_rvv(nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c, nk_size_t n,
                                   nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha, beta_val = *beta;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, c += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(n);
        vuint32m1_t a_u32m1 = __riscv_vle32_v_u32m1(a, vector_length);
        vuint32m1_t b_u32m1 = __riscv_vle32_v_u32m1(b, vector_length);
        vuint32m1_t c_u32m1 = __riscv_vle32_v_u32m1(c, vector_length);
        vfloat64m2_t a_f64m2 = __riscv_vfwcvt_f_xu_v_f64m2(a_u32m1, vector_length);
        vfloat64m2_t b_f64m2 = __riscv_vfwcvt_f_xu_v_f64m2(b_u32m1, vector_length);
        vfloat64m2_t c_f64m2 = __riscv_vfwcvt_f_xu_v_f64m2(c_u32m1, vector_length);
        vfloat64m2_t product_f64m2 = __riscv_vfmul_vv_f64m2(a_f64m2, b_f64m2, vector_length);
        vfloat64m2_t scaled_product_f64m2 = __riscv_vfmul_vf_f64m2(product_f64m2, alpha_val, vector_length);
        vfloat64m2_t result_f64m2 = __riscv_vfmacc_vf_f64m2(scaled_product_f64m2, beta_val, c_f64m2, vector_length);
        result_f64m2 = __riscv_vfmax_vf_f64m2(result_f64m2, 0.0, vector_length);
        result_f64m2 = __riscv_vfmin_vf_f64m2(result_f64m2, 4294967295.0, vector_length);
        vuint32m1_t result_u32m1 = __riscv_vfncvt_xu_f_w_u32m1(result_f64m2, vector_length);
        __riscv_vse32_v_u32m1(result, result_u32m1, vector_length);
    }
}

NK_PUBLIC void nk_each_fma_i64_rvv(nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c, nk_size_t n,
                                   nk_f64_t const *alpha, nk_f64_t const *beta, nk_i64_t *result) {
    nk_f64_t alpha_val = *alpha, beta_val = *beta;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, c += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        vint64m4_t a_i64m4 = __riscv_vle64_v_i64m4(a, vector_length);
        vint64m4_t b_i64m4 = __riscv_vle64_v_i64m4(b, vector_length);
        vint64m4_t c_i64m4 = __riscv_vle64_v_i64m4(c, vector_length);
        vfloat64m4_t a_f64m4 = __riscv_vfcvt_f_x_v_f64m4(a_i64m4, vector_length);
        vfloat64m4_t b_f64m4 = __riscv_vfcvt_f_x_v_f64m4(b_i64m4, vector_length);
        vfloat64m4_t c_f64m4 = __riscv_vfcvt_f_x_v_f64m4(c_i64m4, vector_length);
        vfloat64m4_t product_f64m4 = __riscv_vfmul_vv_f64m4(a_f64m4, b_f64m4, vector_length);
        vfloat64m4_t scaled_product_f64m4 = __riscv_vfmul_vf_f64m4(product_f64m4, alpha_val, vector_length);
        vfloat64m4_t result_f64m4 = __riscv_vfmacc_vf_f64m4(scaled_product_f64m4, beta_val, c_f64m4, vector_length);
        vint64m4_t result_i64m4 = __riscv_vfcvt_x_f_v_i64m4(result_f64m4, vector_length);
        __riscv_vse64_v_i64m4(result, result_i64m4, vector_length);
    }
}

NK_PUBLIC void nk_each_fma_u64_rvv(nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c, nk_size_t n,
                                   nk_f64_t const *alpha, nk_f64_t const *beta, nk_u64_t *result) {
    nk_f64_t alpha_val = *alpha, beta_val = *beta;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, c += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        vuint64m4_t a_u64m4 = __riscv_vle64_v_u64m4(a, vector_length);
        vuint64m4_t b_u64m4 = __riscv_vle64_v_u64m4(b, vector_length);
        vuint64m4_t c_u64m4 = __riscv_vle64_v_u64m4(c, vector_length);
        vfloat64m4_t a_f64m4 = __riscv_vfcvt_f_xu_v_f64m4(a_u64m4, vector_length);
        vfloat64m4_t b_f64m4 = __riscv_vfcvt_f_xu_v_f64m4(b_u64m4, vector_length);
        vfloat64m4_t c_f64m4 = __riscv_vfcvt_f_xu_v_f64m4(c_u64m4, vector_length);
        vfloat64m4_t product_f64m4 = __riscv_vfmul_vv_f64m4(a_f64m4, b_f64m4, vector_length);
        vfloat64m4_t scaled_product_f64m4 = __riscv_vfmul_vf_f64m4(product_f64m4, alpha_val, vector_length);
        vfloat64m4_t result_f64m4 = __riscv_vfmacc_vf_f64m4(scaled_product_f64m4, beta_val, c_f64m4, vector_length);
        result_f64m4 = __riscv_vfmax_vf_f64m4(result_f64m4, 0.0, vector_length);
        vuint64m4_t result_u64m4 = __riscv_vfcvt_xu_f_v_u64m4(result_f64m4, vector_length);
        __riscv_vse64_v_u64m4(result, result_u64m4, vector_length);
    }
}

NK_PUBLIC void nk_each_fma_e4m3_rvv(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_e4m3_t const *c, nk_size_t n,
                                    nk_f32_t const *alpha, nk_f32_t const *beta, nk_e4m3_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, c += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b, vector_length);
        vuint8m1_t c_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)c, vector_length);
        vfloat32m4_t a_f32m4 = nk_e4m3m1_to_f32m4_rvv_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e4m3m1_to_f32m4_rvv_(b_u8m1, vector_length);
        vfloat32m4_t c_f32m4 = nk_e4m3m1_to_f32m4_rvv_(c_u8m1, vector_length);
        vfloat32m4_t product_f32m4 = __riscv_vfmul_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        vfloat32m4_t scaled_product_f32m4 = __riscv_vfmul_vf_f32m4(product_f32m4, alpha_val, vector_length);
        vfloat32m4_t result_f32m4 = __riscv_vfmacc_vf_f32m4(scaled_product_f32m4, beta_val, c_f32m4, vector_length);
        vuint8m1_t result_u8m1 = nk_f32m4_to_e4m3m1_rvv_(result_f32m4, vector_length);
        __riscv_vse8_v_u8m1((nk_u8_t *)result, result_u8m1, vector_length);
    }
}

NK_PUBLIC void nk_each_fma_e5m2_rvv(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_e5m2_t const *c, nk_size_t n,
                                    nk_f32_t const *alpha, nk_f32_t const *beta, nk_e5m2_t *result) {
    nk_f32_t alpha_val = *alpha, beta_val = *beta;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a += vector_length, b += vector_length, c += vector_length, result += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(n);
        vuint8m1_t a_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)a, vector_length);
        vuint8m1_t b_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)b, vector_length);
        vuint8m1_t c_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)c, vector_length);
        vfloat32m4_t a_f32m4 = nk_e5m2m1_to_f32m4_rvv_(a_u8m1, vector_length);
        vfloat32m4_t b_f32m4 = nk_e5m2m1_to_f32m4_rvv_(b_u8m1, vector_length);
        vfloat32m4_t c_f32m4 = nk_e5m2m1_to_f32m4_rvv_(c_u8m1, vector_length);
        vfloat32m4_t product_f32m4 = __riscv_vfmul_vv_f32m4(a_f32m4, b_f32m4, vector_length);
        vfloat32m4_t scaled_product_f32m4 = __riscv_vfmul_vf_f32m4(product_f32m4, alpha_val, vector_length);
        vfloat32m4_t result_f32m4 = __riscv_vfmacc_vf_f32m4(scaled_product_f32m4, beta_val, c_f32m4, vector_length);
        vuint8m1_t result_u8m1 = nk_f32m4_to_e5m2m1_rvv_(result_f32m4, vector_length);
        __riscv_vse8_v_u8m1((nk_u8_t *)result, result_u8m1, vector_length);
    }
}

NK_PUBLIC void nk_each_scale_f32c_rvv(nk_f32c_t const *a, nk_size_t n, nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                      nk_f32c_t *result) {
    nk_f32_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f32_t beta_real = beta->real, beta_imag = beta->imag;
    nk_f32_t const *a_f32 = (nk_f32_t const *)a;
    nk_f32_t *result_f32 = (nk_f32_t *)result;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a_f32 += vector_length * 2, result_f32 += vector_length * 2) {
        vector_length = __riscv_vsetvl_e32m2(n);
        vfloat32m2x2_t a_f32m2x2 = __riscv_vlseg2e32_v_f32m2x2(a_f32, vector_length);
        vfloat32m2_t a_real_f32m2 = __riscv_vget_v_f32m2x2_f32m2(a_f32m2x2, 0);
        vfloat32m2_t a_imag_f32m2 = __riscv_vget_v_f32m2x2_f32m2(a_f32m2x2, 1);
        vfloat32m2_t y_real_f32m2 = __riscv_vfmv_v_f_f32m2(beta_real, vector_length);
        y_real_f32m2 = __riscv_vfmacc_vf_f32m2(y_real_f32m2, alpha_real, a_real_f32m2, vector_length);
        y_real_f32m2 = __riscv_vfnmsac_vf_f32m2(y_real_f32m2, alpha_imag, a_imag_f32m2, vector_length);
        vfloat32m2_t y_imag_f32m2 = __riscv_vfmv_v_f_f32m2(beta_imag, vector_length);
        y_imag_f32m2 = __riscv_vfmacc_vf_f32m2(y_imag_f32m2, alpha_real, a_imag_f32m2, vector_length);
        y_imag_f32m2 = __riscv_vfmacc_vf_f32m2(y_imag_f32m2, alpha_imag, a_real_f32m2, vector_length);
        vfloat32m2x2_t out_f32m2x2 = __riscv_vcreate_v_f32m2x2(y_real_f32m2, y_imag_f32m2);
        __riscv_vsseg2e32_v_f32m2x2(result_f32, out_f32m2x2, vector_length);
    }
}

NK_PUBLIC void nk_each_scale_f64c_rvv(nk_f64c_t const *a, nk_size_t n, nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                      nk_f64c_t *result) {
    nk_f64_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f64_t beta_real = beta->real, beta_imag = beta->imag;
    nk_f64_t const *a_f64 = (nk_f64_t const *)a;
    nk_f64_t *result_f64 = (nk_f64_t *)result;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a_f64 += vector_length * 2, result_f64 += vector_length * 2) {
        vector_length = __riscv_vsetvl_e64m2(n);
        vfloat64m2x2_t a_f64m2x2 = __riscv_vlseg2e64_v_f64m2x2(a_f64, vector_length);
        vfloat64m2_t a_real_f64m2 = __riscv_vget_v_f64m2x2_f64m2(a_f64m2x2, 0);
        vfloat64m2_t a_imag_f64m2 = __riscv_vget_v_f64m2x2_f64m2(a_f64m2x2, 1);
        vfloat64m2_t y_real_f64m2 = __riscv_vfmv_v_f_f64m2(beta_real, vector_length);
        y_real_f64m2 = __riscv_vfmacc_vf_f64m2(y_real_f64m2, alpha_real, a_real_f64m2, vector_length);
        y_real_f64m2 = __riscv_vfnmsac_vf_f64m2(y_real_f64m2, alpha_imag, a_imag_f64m2, vector_length);
        vfloat64m2_t y_imag_f64m2 = __riscv_vfmv_v_f_f64m2(beta_imag, vector_length);
        y_imag_f64m2 = __riscv_vfmacc_vf_f64m2(y_imag_f64m2, alpha_real, a_imag_f64m2, vector_length);
        y_imag_f64m2 = __riscv_vfmacc_vf_f64m2(y_imag_f64m2, alpha_imag, a_real_f64m2, vector_length);
        vfloat64m2x2_t out_f64m2x2 = __riscv_vcreate_v_f64m2x2(y_real_f64m2, y_imag_f64m2);
        __riscv_vsseg2e64_v_f64m2x2(result_f64, out_f64m2x2, vector_length);
    }
}

NK_PUBLIC void nk_each_blend_f32c_rvv(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t const *alpha,
                                      nk_f32c_t const *beta, nk_f32c_t *result) {
    nk_f32_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f32_t beta_real = beta->real, beta_imag = beta->imag;
    nk_f32_t const *a_f32 = (nk_f32_t const *)a;
    nk_f32_t const *b_f32 = (nk_f32_t const *)b;
    nk_f32_t *result_f32 = (nk_f32_t *)result;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a_f32 += vector_length * 2, b_f32 += vector_length * 2, result_f32 += vector_length * 2) {
        vector_length = __riscv_vsetvl_e32m2(n);
        vfloat32m2x2_t a_f32m2x2 = __riscv_vlseg2e32_v_f32m2x2(a_f32, vector_length);
        vfloat32m2x2_t b_f32m2x2 = __riscv_vlseg2e32_v_f32m2x2(b_f32, vector_length);
        vfloat32m2_t a_real_f32m2 = __riscv_vget_v_f32m2x2_f32m2(a_f32m2x2, 0);
        vfloat32m2_t a_imag_f32m2 = __riscv_vget_v_f32m2x2_f32m2(a_f32m2x2, 1);
        vfloat32m2_t b_real_f32m2 = __riscv_vget_v_f32m2x2_f32m2(b_f32m2x2, 0);
        vfloat32m2_t b_imag_f32m2 = __riscv_vget_v_f32m2x2_f32m2(b_f32m2x2, 1);
        vfloat32m2_t ya_real_f32m2 = __riscv_vfmul_vf_f32m2(a_real_f32m2, alpha_real, vector_length);
        ya_real_f32m2 = __riscv_vfnmsac_vf_f32m2(ya_real_f32m2, alpha_imag, a_imag_f32m2, vector_length);
        vfloat32m2_t ya_imag_f32m2 = __riscv_vfmul_vf_f32m2(a_imag_f32m2, alpha_real, vector_length);
        ya_imag_f32m2 = __riscv_vfmacc_vf_f32m2(ya_imag_f32m2, alpha_imag, a_real_f32m2, vector_length);
        vfloat32m2_t y_real_f32m2 = __riscv_vfmacc_vf_f32m2(ya_real_f32m2, beta_real, b_real_f32m2, vector_length);
        y_real_f32m2 = __riscv_vfnmsac_vf_f32m2(y_real_f32m2, beta_imag, b_imag_f32m2, vector_length);
        vfloat32m2_t y_imag_f32m2 = __riscv_vfmacc_vf_f32m2(ya_imag_f32m2, beta_real, b_imag_f32m2, vector_length);
        y_imag_f32m2 = __riscv_vfmacc_vf_f32m2(y_imag_f32m2, beta_imag, b_real_f32m2, vector_length);
        vfloat32m2x2_t out_f32m2x2 = __riscv_vcreate_v_f32m2x2(y_real_f32m2, y_imag_f32m2);
        __riscv_vsseg2e32_v_f32m2x2(result_f32, out_f32m2x2, vector_length);
    }
}

NK_PUBLIC void nk_each_blend_f64c_rvv(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t const *alpha,
                                      nk_f64c_t const *beta, nk_f64c_t *result) {
    nk_f64_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f64_t beta_real = beta->real, beta_imag = beta->imag;
    nk_f64_t const *a_f64 = (nk_f64_t const *)a;
    nk_f64_t const *b_f64 = (nk_f64_t const *)b;
    nk_f64_t *result_f64 = (nk_f64_t *)result;
    for (nk_size_t vector_length; n > 0;
         n -= vector_length, a_f64 += vector_length * 2, b_f64 += vector_length * 2, result_f64 += vector_length * 2) {
        vector_length = __riscv_vsetvl_e64m2(n);
        vfloat64m2x2_t a_f64m2x2 = __riscv_vlseg2e64_v_f64m2x2(a_f64, vector_length);
        vfloat64m2x2_t b_f64m2x2 = __riscv_vlseg2e64_v_f64m2x2(b_f64, vector_length);
        vfloat64m2_t a_real_f64m2 = __riscv_vget_v_f64m2x2_f64m2(a_f64m2x2, 0);
        vfloat64m2_t a_imag_f64m2 = __riscv_vget_v_f64m2x2_f64m2(a_f64m2x2, 1);
        vfloat64m2_t b_real_f64m2 = __riscv_vget_v_f64m2x2_f64m2(b_f64m2x2, 0);
        vfloat64m2_t b_imag_f64m2 = __riscv_vget_v_f64m2x2_f64m2(b_f64m2x2, 1);
        vfloat64m2_t ya_real_f64m2 = __riscv_vfmul_vf_f64m2(a_real_f64m2, alpha_real, vector_length);
        ya_real_f64m2 = __riscv_vfnmsac_vf_f64m2(ya_real_f64m2, alpha_imag, a_imag_f64m2, vector_length);
        vfloat64m2_t ya_imag_f64m2 = __riscv_vfmul_vf_f64m2(a_imag_f64m2, alpha_real, vector_length);
        ya_imag_f64m2 = __riscv_vfmacc_vf_f64m2(ya_imag_f64m2, alpha_imag, a_real_f64m2, vector_length);
        vfloat64m2_t y_real_f64m2 = __riscv_vfmacc_vf_f64m2(ya_real_f64m2, beta_real, b_real_f64m2, vector_length);
        y_real_f64m2 = __riscv_vfnmsac_vf_f64m2(y_real_f64m2, beta_imag, b_imag_f64m2, vector_length);
        vfloat64m2_t y_imag_f64m2 = __riscv_vfmacc_vf_f64m2(ya_imag_f64m2, beta_real, b_imag_f64m2, vector_length);
        y_imag_f64m2 = __riscv_vfmacc_vf_f64m2(y_imag_f64m2, beta_imag, b_real_f64m2, vector_length);
        vfloat64m2x2_t out_f64m2x2 = __riscv_vcreate_v_f64m2x2(y_real_f64m2, y_imag_f64m2);
        __riscv_vsseg2e64_v_f64m2x2(result_f64, out_f64m2x2, vector_length);
    }
}

NK_PUBLIC void nk_each_fma_f32c_rvv(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                    nk_f32c_t const *alpha, nk_f32c_t const *beta, nk_f32c_t *result) {
    nk_f32_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f32_t beta_real = beta->real, beta_imag = beta->imag;
    nk_f32_t const *a_f32 = (nk_f32_t const *)a;
    nk_f32_t const *b_f32 = (nk_f32_t const *)b;
    nk_f32_t const *c_f32 = (nk_f32_t const *)c;
    nk_f32_t *result_f32 = (nk_f32_t *)result;
    for (nk_size_t vector_length; n > 0; n -= vector_length, a_f32 += vector_length * 2, b_f32 += vector_length * 2,
                                         c_f32 += vector_length * 2, result_f32 += vector_length * 2) {
        vector_length = __riscv_vsetvl_e32m2(n);
        vfloat32m2x2_t a_f32m2x2 = __riscv_vlseg2e32_v_f32m2x2(a_f32, vector_length);
        vfloat32m2x2_t b_f32m2x2 = __riscv_vlseg2e32_v_f32m2x2(b_f32, vector_length);
        vfloat32m2x2_t c_f32m2x2 = __riscv_vlseg2e32_v_f32m2x2(c_f32, vector_length);
        vfloat32m2_t a_real_f32m2 = __riscv_vget_v_f32m2x2_f32m2(a_f32m2x2, 0);
        vfloat32m2_t a_imag_f32m2 = __riscv_vget_v_f32m2x2_f32m2(a_f32m2x2, 1);
        vfloat32m2_t b_real_f32m2 = __riscv_vget_v_f32m2x2_f32m2(b_f32m2x2, 0);
        vfloat32m2_t b_imag_f32m2 = __riscv_vget_v_f32m2x2_f32m2(b_f32m2x2, 1);
        vfloat32m2_t c_real_f32m2 = __riscv_vget_v_f32m2x2_f32m2(c_f32m2x2, 0);
        vfloat32m2_t c_imag_f32m2 = __riscv_vget_v_f32m2x2_f32m2(c_f32m2x2, 1);
        vfloat32m2_t ab_real_f32m2 = __riscv_vfmul_vv_f32m2(a_real_f32m2, b_real_f32m2, vector_length);
        ab_real_f32m2 = __riscv_vfnmsac_vv_f32m2(ab_real_f32m2, a_imag_f32m2, b_imag_f32m2, vector_length);
        vfloat32m2_t ab_imag_f32m2 = __riscv_vfmul_vv_f32m2(a_real_f32m2, b_imag_f32m2, vector_length);
        ab_imag_f32m2 = __riscv_vfmacc_vv_f32m2(ab_imag_f32m2, a_imag_f32m2, b_real_f32m2, vector_length);
        vfloat32m2_t y_real_f32m2 = __riscv_vfmul_vf_f32m2(ab_real_f32m2, alpha_real, vector_length);
        y_real_f32m2 = __riscv_vfnmsac_vf_f32m2(y_real_f32m2, alpha_imag, ab_imag_f32m2, vector_length);
        vfloat32m2_t y_imag_f32m2 = __riscv_vfmul_vf_f32m2(ab_imag_f32m2, alpha_real, vector_length);
        y_imag_f32m2 = __riscv_vfmacc_vf_f32m2(y_imag_f32m2, alpha_imag, ab_real_f32m2, vector_length);
        y_real_f32m2 = __riscv_vfmacc_vf_f32m2(y_real_f32m2, beta_real, c_real_f32m2, vector_length);
        y_real_f32m2 = __riscv_vfnmsac_vf_f32m2(y_real_f32m2, beta_imag, c_imag_f32m2, vector_length);
        y_imag_f32m2 = __riscv_vfmacc_vf_f32m2(y_imag_f32m2, beta_real, c_imag_f32m2, vector_length);
        y_imag_f32m2 = __riscv_vfmacc_vf_f32m2(y_imag_f32m2, beta_imag, c_real_f32m2, vector_length);
        vfloat32m2x2_t out_f32m2x2 = __riscv_vcreate_v_f32m2x2(y_real_f32m2, y_imag_f32m2);
        __riscv_vsseg2e32_v_f32m2x2(result_f32, out_f32m2x2, vector_length);
    }
}

NK_PUBLIC void nk_each_fma_f64c_rvv(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                    nk_f64c_t const *alpha, nk_f64c_t const *beta, nk_f64c_t *result) {
    nk_f64_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f64_t beta_real = beta->real, beta_imag = beta->imag;
    nk_f64_t const *a_f64 = (nk_f64_t const *)a;
    nk_f64_t const *b_f64 = (nk_f64_t const *)b;
    nk_f64_t const *c_f64 = (nk_f64_t const *)c;
    nk_f64_t *result_f64 = (nk_f64_t *)result;
    for (nk_size_t vector_length; n > 0; n -= vector_length, a_f64 += vector_length * 2, b_f64 += vector_length * 2,
                                         c_f64 += vector_length * 2, result_f64 += vector_length * 2) {
        vector_length = __riscv_vsetvl_e64m2(n);
        vfloat64m2x2_t a_f64m2x2 = __riscv_vlseg2e64_v_f64m2x2(a_f64, vector_length);
        vfloat64m2x2_t b_f64m2x2 = __riscv_vlseg2e64_v_f64m2x2(b_f64, vector_length);
        vfloat64m2x2_t c_f64m2x2 = __riscv_vlseg2e64_v_f64m2x2(c_f64, vector_length);
        vfloat64m2_t a_real_f64m2 = __riscv_vget_v_f64m2x2_f64m2(a_f64m2x2, 0);
        vfloat64m2_t a_imag_f64m2 = __riscv_vget_v_f64m2x2_f64m2(a_f64m2x2, 1);
        vfloat64m2_t b_real_f64m2 = __riscv_vget_v_f64m2x2_f64m2(b_f64m2x2, 0);
        vfloat64m2_t b_imag_f64m2 = __riscv_vget_v_f64m2x2_f64m2(b_f64m2x2, 1);
        vfloat64m2_t c_real_f64m2 = __riscv_vget_v_f64m2x2_f64m2(c_f64m2x2, 0);
        vfloat64m2_t c_imag_f64m2 = __riscv_vget_v_f64m2x2_f64m2(c_f64m2x2, 1);
        vfloat64m2_t ab_real_f64m2 = __riscv_vfmul_vv_f64m2(a_real_f64m2, b_real_f64m2, vector_length);
        ab_real_f64m2 = __riscv_vfnmsac_vv_f64m2(ab_real_f64m2, a_imag_f64m2, b_imag_f64m2, vector_length);
        vfloat64m2_t ab_imag_f64m2 = __riscv_vfmul_vv_f64m2(a_real_f64m2, b_imag_f64m2, vector_length);
        ab_imag_f64m2 = __riscv_vfmacc_vv_f64m2(ab_imag_f64m2, a_imag_f64m2, b_real_f64m2, vector_length);
        vfloat64m2_t y_real_f64m2 = __riscv_vfmul_vf_f64m2(ab_real_f64m2, alpha_real, vector_length);
        y_real_f64m2 = __riscv_vfnmsac_vf_f64m2(y_real_f64m2, alpha_imag, ab_imag_f64m2, vector_length);
        vfloat64m2_t y_imag_f64m2 = __riscv_vfmul_vf_f64m2(ab_imag_f64m2, alpha_real, vector_length);
        y_imag_f64m2 = __riscv_vfmacc_vf_f64m2(y_imag_f64m2, alpha_imag, ab_real_f64m2, vector_length);
        y_real_f64m2 = __riscv_vfmacc_vf_f64m2(y_real_f64m2, beta_real, c_real_f64m2, vector_length);
        y_real_f64m2 = __riscv_vfnmsac_vf_f64m2(y_real_f64m2, beta_imag, c_imag_f64m2, vector_length);
        y_imag_f64m2 = __riscv_vfmacc_vf_f64m2(y_imag_f64m2, beta_real, c_imag_f64m2, vector_length);
        y_imag_f64m2 = __riscv_vfmacc_vf_f64m2(y_imag_f64m2, beta_imag, c_real_f64m2, vector_length);
        vfloat64m2x2_t out_f64m2x2 = __riscv_vcreate_v_f64m2x2(y_real_f64m2, y_imag_f64m2);
        __riscv_vsseg2e64_v_f64m2x2(result_f64, out_f64m2x2, vector_length);
    }
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
#endif // NK_EACH_RVV_H
