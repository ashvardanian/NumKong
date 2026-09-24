/**
 *  @brief SIMD-accelerated Scalar Math Helpers for RISC-V.
 *  @file include/numkong/scalar/rvv.h
 *  @author Ash Vardanian
 *  @date March 1, 2026
 *
 *  @sa include/numkong/scalar.h
 *
 *  RVV scalar helpers use vector instructions with VL=1 for hardware sqrt/rsqrt.
 *  `vfrsqrt7` provides 7-bit mantissa precision; Newton-Raphson refines to full precision.
 */
#ifndef NK_SCALAR_RVV_H
#define NK_SCALAR_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC nk_f32_t nk_f32_rsqrt_rvv(nk_f32_t number) {
    vfloat32m1_t x_f32m1 = __riscv_vfmv_s_f_f32m1(number, 1);
    vfloat32m1_t estimate_f32m1 = __riscv_vfrsqrt7_v_f32m1(x_f32m1, 1);
    vfloat32m1_t half_f32m1 = __riscv_vfmv_s_f_f32m1(0.5f, 1);
    vfloat32m1_t three_half_f32m1 = __riscv_vfmv_s_f_f32m1(1.5f, 1);
    vfloat32m1_t half_x_f32m1 = __riscv_vfmul_vv_f32m1(half_f32m1, x_f32m1, 1);
    // Iteration 1
    vfloat32m1_t estimate_sq_f32m1 = __riscv_vfmul_vv_f32m1(estimate_f32m1, estimate_f32m1, 1);
    vfloat32m1_t correction_f32m1 = __riscv_vfmul_vv_f32m1(half_x_f32m1, estimate_sq_f32m1, 1);
    vfloat32m1_t factor_f32m1 = __riscv_vfsub_vv_f32m1(three_half_f32m1, correction_f32m1, 1);
    estimate_f32m1 = __riscv_vfmul_vv_f32m1(estimate_f32m1, factor_f32m1, 1);
    // Iteration 2
    estimate_sq_f32m1 = __riscv_vfmul_vv_f32m1(estimate_f32m1, estimate_f32m1, 1);
    correction_f32m1 = __riscv_vfmul_vv_f32m1(half_x_f32m1, estimate_sq_f32m1, 1);
    factor_f32m1 = __riscv_vfsub_vv_f32m1(three_half_f32m1, correction_f32m1, 1);
    estimate_f32m1 = __riscv_vfmul_vv_f32m1(estimate_f32m1, factor_f32m1, 1);
    return __riscv_vfmv_f_s_f32m1_f32(estimate_f32m1);
}

NK_PUBLIC nk_f32_t nk_f32_sqrt_rvv(nk_f32_t number) {
    vfloat32m1_t x_f32m1 = __riscv_vfmv_s_f_f32m1(number, 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfsqrt_v_f32m1(x_f32m1, 1));
}

NK_PUBLIC nk_f64_t nk_f64_rsqrt_rvv(nk_f64_t number) {
    vfloat64m1_t x_f64m1 = __riscv_vfmv_s_f_f64m1(number, 1);
    vfloat64m1_t estimate_f64m1 = __riscv_vfrsqrt7_v_f64m1(x_f64m1, 1);
    vfloat64m1_t half_f64m1 = __riscv_vfmv_s_f_f64m1(0.5, 1);
    vfloat64m1_t three_half_f64m1 = __riscv_vfmv_s_f_f64m1(1.5, 1);
    vfloat64m1_t half_x_f64m1 = __riscv_vfmul_vv_f64m1(half_f64m1, x_f64m1, 1);
    // Iteration 1
    vfloat64m1_t estimate_sq_f64m1 = __riscv_vfmul_vv_f64m1(estimate_f64m1, estimate_f64m1, 1);
    vfloat64m1_t correction_f64m1 = __riscv_vfmul_vv_f64m1(half_x_f64m1, estimate_sq_f64m1, 1);
    vfloat64m1_t factor_f64m1 = __riscv_vfsub_vv_f64m1(three_half_f64m1, correction_f64m1, 1);
    estimate_f64m1 = __riscv_vfmul_vv_f64m1(estimate_f64m1, factor_f64m1, 1);
    // Iteration 2
    estimate_sq_f64m1 = __riscv_vfmul_vv_f64m1(estimate_f64m1, estimate_f64m1, 1);
    correction_f64m1 = __riscv_vfmul_vv_f64m1(half_x_f64m1, estimate_sq_f64m1, 1);
    factor_f64m1 = __riscv_vfsub_vv_f64m1(three_half_f64m1, correction_f64m1, 1);
    estimate_f64m1 = __riscv_vfmul_vv_f64m1(estimate_f64m1, factor_f64m1, 1);
    // Iteration 3
    estimate_sq_f64m1 = __riscv_vfmul_vv_f64m1(estimate_f64m1, estimate_f64m1, 1);
    correction_f64m1 = __riscv_vfmul_vv_f64m1(half_x_f64m1, estimate_sq_f64m1, 1);
    factor_f64m1 = __riscv_vfsub_vv_f64m1(three_half_f64m1, correction_f64m1, 1);
    estimate_f64m1 = __riscv_vfmul_vv_f64m1(estimate_f64m1, factor_f64m1, 1);
    return __riscv_vfmv_f_s_f64m1_f64(estimate_f64m1);
}

NK_PUBLIC nk_f64_t nk_f64_sqrt_rvv(nk_f64_t number) {
    vfloat64m1_t x_f64m1 = __riscv_vfmv_s_f_f64m1(number, 1);
    return __riscv_vfmv_f_s_f64m1_f64(__riscv_vfsqrt_v_f64m1(x_f64m1, 1));
}

NK_PUBLIC nk_f32_t nk_f32_fma_rvv(nk_f32_t a, nk_f32_t b, nk_f32_t c) {
    vfloat32m1_t a_f32m1 = __riscv_vfmv_s_f_f32m1(a, 1);
    vfloat32m1_t c_f32m1 = __riscv_vfmv_s_f_f32m1(c, 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfmacc_vf_f32m1(c_f32m1, b, a_f32m1, 1));
}

NK_PUBLIC nk_f64_t nk_f64_fma_rvv(nk_f64_t a, nk_f64_t b, nk_f64_t c) {
    vfloat64m1_t a_f64m1 = __riscv_vfmv_s_f_f64m1(a, 1);
    vfloat64m1_t c_f64m1 = __riscv_vfmv_s_f_f64m1(c, 1);
    return __riscv_vfmv_f_s_f64m1_f64(__riscv_vfmacc_vf_f64m1(c_f64m1, b, a_f64m1, 1));
}

NK_PUBLIC nk_u8_t nk_u8_saturating_add_rvv(nk_u8_t a, nk_u8_t b) {
    vuint8m1_t a_u8m1 = __riscv_vmv_v_x_u8m1(a, 1);
    vuint8m1_t b_u8m1 = __riscv_vmv_v_x_u8m1(b, 1);
    return __riscv_vmv_x_s_u8m1_u8(__riscv_vsaddu_vv_u8m1(a_u8m1, b_u8m1, 1));
}

NK_PUBLIC nk_i8_t nk_i8_saturating_add_rvv(nk_i8_t a, nk_i8_t b) {
    vint8m1_t a_i8m1 = __riscv_vmv_v_x_i8m1(a, 1);
    vint8m1_t b_i8m1 = __riscv_vmv_v_x_i8m1(b, 1);
    return __riscv_vmv_x_s_i8m1_i8(__riscv_vsadd_vv_i8m1(a_i8m1, b_i8m1, 1));
}

NK_PUBLIC nk_u16_t nk_u16_saturating_add_rvv(nk_u16_t a, nk_u16_t b) {
    vuint16m1_t a_u16m1 = __riscv_vmv_v_x_u16m1(a, 1);
    vuint16m1_t b_u16m1 = __riscv_vmv_v_x_u16m1(b, 1);
    return __riscv_vmv_x_s_u16m1_u16(__riscv_vsaddu_vv_u16m1(a_u16m1, b_u16m1, 1));
}

NK_PUBLIC nk_i16_t nk_i16_saturating_add_rvv(nk_i16_t a, nk_i16_t b) {
    vint16m1_t a_i16m1 = __riscv_vmv_v_x_i16m1(a, 1);
    vint16m1_t b_i16m1 = __riscv_vmv_v_x_i16m1(b, 1);
    return __riscv_vmv_x_s_i16m1_i16(__riscv_vsadd_vv_i16m1(a_i16m1, b_i16m1, 1));
}

NK_PUBLIC nk_u32_t nk_u32_saturating_add_rvv(nk_u32_t a, nk_u32_t b) {
    vuint32m1_t a_u32m1 = __riscv_vmv_v_x_u32m1(a, 1);
    vuint32m1_t b_u32m1 = __riscv_vmv_v_x_u32m1(b, 1);
    return __riscv_vmv_x_s_u32m1_u32(__riscv_vsaddu_vv_u32m1(a_u32m1, b_u32m1, 1));
}

NK_PUBLIC nk_i32_t nk_i32_saturating_add_rvv(nk_i32_t a, nk_i32_t b) {
    vint32m1_t a_i32m1 = __riscv_vmv_v_x_i32m1(a, 1);
    vint32m1_t b_i32m1 = __riscv_vmv_v_x_i32m1(b, 1);
    return __riscv_vmv_x_s_i32m1_i32(__riscv_vsadd_vv_i32m1(a_i32m1, b_i32m1, 1));
}

NK_PUBLIC nk_u64_t nk_u64_saturating_add_rvv(nk_u64_t a, nk_u64_t b) {
    vuint64m1_t a_u64m1 = __riscv_vmv_v_x_u64m1(a, 1);
    vuint64m1_t b_u64m1 = __riscv_vmv_v_x_u64m1(b, 1);
    return __riscv_vmv_x_s_u64m1_u64(__riscv_vsaddu_vv_u64m1(a_u64m1, b_u64m1, 1));
}

NK_PUBLIC nk_i64_t nk_i64_saturating_add_rvv(nk_i64_t a, nk_i64_t b) {
    vint64m1_t a_i64m1 = __riscv_vmv_v_x_i64m1(a, 1);
    vint64m1_t b_i64m1 = __riscv_vmv_v_x_i64m1(b, 1);
    return __riscv_vmv_x_s_i64m1_i64(__riscv_vsadd_vv_i64m1(a_i64m1, b_i64m1, 1));
}

NK_PUBLIC nk_u8_t nk_u8_saturating_mul_rvv(nk_u8_t a, nk_u8_t b) {
    vuint8m1_t a_u8m1 = __riscv_vmv_v_x_u8m1(a, 1);
    vuint8m1_t b_u8m1 = __riscv_vmv_v_x_u8m1(b, 1);
    vuint16m2_t product_u16m2 = __riscv_vwmulu_vv_u16m2(a_u8m1, b_u8m1, 1);
    return __riscv_vmv_x_s_u8m1_u8(__riscv_vnclipu_wx_u8m1(product_u16m2, 0, __RISCV_VXRM_RDN, 1));
}
NK_PUBLIC nk_i8_t nk_i8_saturating_mul_rvv(nk_i8_t a, nk_i8_t b) {
    vint8m1_t a_i8m1 = __riscv_vmv_v_x_i8m1(a, 1);
    vint8m1_t b_i8m1 = __riscv_vmv_v_x_i8m1(b, 1);
    vint16m2_t product_i16m2 = __riscv_vwmul_vv_i16m2(a_i8m1, b_i8m1, 1);
    return __riscv_vmv_x_s_i8m1_i8(__riscv_vnclip_wx_i8m1(product_i16m2, 0, __RISCV_VXRM_RDN, 1));
}
NK_PUBLIC nk_u16_t nk_u16_saturating_mul_rvv(nk_u16_t a, nk_u16_t b) {
    vuint16m1_t a_u16m1 = __riscv_vmv_v_x_u16m1(a, 1);
    vuint16m1_t b_u16m1 = __riscv_vmv_v_x_u16m1(b, 1);
    vuint32m2_t product_u32m2 = __riscv_vwmulu_vv_u32m2(a_u16m1, b_u16m1, 1);
    return __riscv_vmv_x_s_u16m1_u16(__riscv_vnclipu_wx_u16m1(product_u32m2, 0, __RISCV_VXRM_RDN, 1));
}
NK_PUBLIC nk_i16_t nk_i16_saturating_mul_rvv(nk_i16_t a, nk_i16_t b) {
    vint16m1_t a_i16m1 = __riscv_vmv_v_x_i16m1(a, 1);
    vint16m1_t b_i16m1 = __riscv_vmv_v_x_i16m1(b, 1);
    vint32m2_t product_i32m2 = __riscv_vwmul_vv_i32m2(a_i16m1, b_i16m1, 1);
    return __riscv_vmv_x_s_i16m1_i16(__riscv_vnclip_wx_i16m1(product_i32m2, 0, __RISCV_VXRM_RDN, 1));
}
NK_PUBLIC nk_u32_t nk_u32_saturating_mul_rvv(nk_u32_t a, nk_u32_t b) {
    vuint32m1_t a_u32m1 = __riscv_vmv_v_x_u32m1(a, 1);
    vuint32m1_t b_u32m1 = __riscv_vmv_v_x_u32m1(b, 1);
    vuint64m2_t product_u64m2 = __riscv_vwmulu_vv_u64m2(a_u32m1, b_u32m1, 1);
    return __riscv_vmv_x_s_u32m1_u32(__riscv_vnclipu_wx_u32m1(product_u64m2, 0, __RISCV_VXRM_RDN, 1));
}
NK_PUBLIC nk_i32_t nk_i32_saturating_mul_rvv(nk_i32_t a, nk_i32_t b) {
    vint32m1_t a_i32m1 = __riscv_vmv_v_x_i32m1(a, 1);
    vint32m1_t b_i32m1 = __riscv_vmv_v_x_i32m1(b, 1);
    vint64m2_t product_i64m2 = __riscv_vwmul_vv_i64m2(a_i32m1, b_i32m1, 1);
    return __riscv_vmv_x_s_i32m1_i32(__riscv_vnclip_wx_i32m1(product_i64m2, 0, __RISCV_VXRM_RDN, 1));
}
NK_PUBLIC nk_u64_t nk_u64_saturating_mul_rvv(nk_u64_t a, nk_u64_t b) {
    vuint64m1_t a_u64m1 = __riscv_vmv_v_x_u64m1(a, 1);
    vuint64m1_t b_u64m1 = __riscv_vmv_v_x_u64m1(b, 1);
    nk_u64_t high = __riscv_vmv_x_s_u64m1_u64(__riscv_vmulhu_vv_u64m1(a_u64m1, b_u64m1, 1));
    return high ? 18446744073709551615ull : (a * b);
}
NK_PUBLIC nk_i64_t nk_i64_saturating_mul_rvv(nk_i64_t a, nk_i64_t b) {
    int sign = (a < 0) ^ (b < 0);
    nk_u64_t abs_a = a < 0 ? -(nk_u64_t)a : (nk_u64_t)a;
    nk_u64_t abs_b = b < 0 ? -(nk_u64_t)b : (nk_u64_t)b;
    vuint64m1_t a_u64m1 = __riscv_vmv_v_x_u64m1(abs_a, 1);
    vuint64m1_t b_u64m1 = __riscv_vmv_v_x_u64m1(abs_b, 1);
    nk_u64_t high = __riscv_vmv_x_s_u64m1_u64(__riscv_vmulhu_vv_u64m1(a_u64m1, b_u64m1, 1));
    nk_u64_t low = abs_a * abs_b;
    if (high || (sign && low > 9223372036854775808ull) || (!sign && low > 9223372036854775807ull))
        return sign ? (-9223372036854775807ll - 1ll) : 9223372036854775807ll;
    return sign ? -(nk_i64_t)low : (nk_i64_t)low;
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
#endif // NK_SCALAR_RVV_H
