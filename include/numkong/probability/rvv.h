/**
 *  @brief SIMD-accelerated Probability Distribution Similarity Measures for RISC-V.
 *  @file include/numkong/probability/rvv.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/probability.h
 *
 *  Implements KLD and JSD using RVV 1.0 vector intrinsics for f32, f64, f16, and bf16.
 *  The log2 approximation uses the same polynomial as the Haswell implementation,
 *  ported to RVV's vector fused-multiply-add instructions.
 *
 *  For f64, uses the s-series 14-term Horner log2 approximation (matching Skylake).
 *  For f16/bf16, converts to f32 using the cast helpers from cast/rvv.h,
 *  then uses the f32 algorithm.
 */
#ifndef NK_PROBABILITY_RVV_H
#define NK_PROBABILITY_RVV_H

#if NK_TARGET_RISCV64_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/probability/serial.h" // `nk_kld_f64_serial`, `nk_jsd_f64_serial`
#include "numkong/cast/rvv.h"           // `nk_f16m1_to_f32m2_rvv_`, `nk_bf16m1_to_f32m2_rvv_`
#include "numkong/spatial/rvv.h"        // `nk_f32_sqrt_rvv`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief  Computes `log2(x)` for a vector of f32 values using IEEE 754 bit manipulation
 *          and a 5-term Horner polynomial, matching the Haswell log2 approximation.
 *
 *  Decomposes each float into exponent and mantissa:
 *  - exponent = (bits >> 23) - 127
 *  - mantissa = (bits & 0x007FFFFF) | 0x3F800000, yielding m in [1, 2)
 *
 *  Then evaluates poly(m) via Horner's method:
 *    poly = -3.4436006e-2f
 *    poly = poly * m + 3.1821337e-1f
 *    poly = poly * m - 1.2315303f
 *    poly = poly * m + 2.5988452f
 *    poly = poly * m - 3.3241990f
 *    poly = poly * m + 3.1157899f
 *
 *  Final result: log2(x) = exponent + poly * (m - 1)
 */
NK_INTERNAL vfloat32m4_t nk_log2_f32m4_rvv_(vfloat32m4_t x, nk_size_t vector_length) {
    vuint32m4_t bits_u32m4 = __riscv_vreinterpret_v_f32m4_u32m4(x);

    // Extract exponent: (bits >> 23) - 127
    vuint32m4_t exponent_u32m4 = __riscv_vsrl_vx_u32m4(bits_u32m4, 23, vector_length);
    vint32m4_t exponent_i32m4 = __riscv_vsub_vx_i32m4(__riscv_vreinterpret_v_u32m4_i32m4(exponent_u32m4), 127,
                                                      vector_length);
    vfloat32m4_t exponent_f32m4 = __riscv_vfcvt_f_x_v_f32m4(exponent_i32m4, vector_length);

    // Extract mantissa: set exponent field to 0 (bias 127), so value is in [1, 2)
    vuint32m4_t mantissa_u32m4 = __riscv_vor_vx_u32m4(__riscv_vand_vx_u32m4(bits_u32m4, 0x007FFFFF, vector_length),
                                                      0x3F800000, vector_length);
    vfloat32m4_t mantissa_f32m4 = __riscv_vreinterpret_v_u32m4_f32m4(mantissa_u32m4);

    // Horner polynomial evaluation:
    //   vfmadd_vv(vd, vs1, vs2) computes vd = vd * vs1 + vs2
    //   So poly = vfmadd(poly, m, coeff) means poly = poly * m + coeff
    vfloat32m4_t poly_f32m4 = __riscv_vfmv_v_f_f32m4(-3.4436006e-2f, vector_length);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, mantissa_f32m4,
                                         __riscv_vfmv_v_f_f32m4(3.1821337e-1f, vector_length), vector_length);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, mantissa_f32m4, __riscv_vfmv_v_f_f32m4(-1.2315303f, vector_length),
                                         vector_length);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, mantissa_f32m4, __riscv_vfmv_v_f_f32m4(2.5988452f, vector_length),
                                         vector_length);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, mantissa_f32m4, __riscv_vfmv_v_f_f32m4(-3.3241990f, vector_length),
                                         vector_length);
    poly_f32m4 = __riscv_vfmadd_vv_f32m4(poly_f32m4, mantissa_f32m4, __riscv_vfmv_v_f_f32m4(3.1157899f, vector_length),
                                         vector_length);

    // result = exponent + poly * (m - 1)
    vfloat32m4_t m_minus_1_f32m4 = __riscv_vfsub_vf_f32m4(mantissa_f32m4, 1.0f, vector_length);
    return __riscv_vfmacc_vv_f32m4(exponent_f32m4, poly_f32m4, m_minus_1_f32m4, vector_length);
}

NK_INTERNAL vfloat32m2_t nk_log2_f32m2_rvv_(vfloat32m2_t x, nk_size_t vector_length) {
    vuint32m2_t bits_u32m2 = __riscv_vreinterpret_v_f32m2_u32m2(x);
    vuint32m2_t exponent_u32m2 = __riscv_vsrl_vx_u32m2(bits_u32m2, 23, vector_length);
    vint32m2_t exponent_i32m2 = __riscv_vsub_vx_i32m2(__riscv_vreinterpret_v_u32m2_i32m2(exponent_u32m2), 127,
                                                      vector_length);
    vfloat32m2_t exponent_f32m2 = __riscv_vfcvt_f_x_v_f32m2(exponent_i32m2, vector_length);
    vuint32m2_t mantissa_u32m2 = __riscv_vor_vx_u32m2(__riscv_vand_vx_u32m2(bits_u32m2, 0x007FFFFF, vector_length),
                                                      0x3F800000, vector_length);
    vfloat32m2_t mantissa_f32m2 = __riscv_vreinterpret_v_u32m2_f32m2(mantissa_u32m2);
    vfloat32m2_t poly_f32m2 = __riscv_vfmv_v_f_f32m2(-3.4436006e-2f, vector_length);
    poly_f32m2 = __riscv_vfmadd_vv_f32m2(poly_f32m2, mantissa_f32m2,
                                         __riscv_vfmv_v_f_f32m2(3.1821337e-1f, vector_length), vector_length);
    poly_f32m2 = __riscv_vfmadd_vv_f32m2(poly_f32m2, mantissa_f32m2, __riscv_vfmv_v_f_f32m2(-1.2315303f, vector_length),
                                         vector_length);
    poly_f32m2 = __riscv_vfmadd_vv_f32m2(poly_f32m2, mantissa_f32m2, __riscv_vfmv_v_f_f32m2(2.5988452f, vector_length),
                                         vector_length);
    poly_f32m2 = __riscv_vfmadd_vv_f32m2(poly_f32m2, mantissa_f32m2, __riscv_vfmv_v_f_f32m2(-3.3241990f, vector_length),
                                         vector_length);
    poly_f32m2 = __riscv_vfmadd_vv_f32m2(poly_f32m2, mantissa_f32m2, __riscv_vfmv_v_f_f32m2(3.1157899f, vector_length),
                                         vector_length);
    vfloat32m2_t m_minus_1_f32m2 = __riscv_vfsub_vf_f32m2(mantissa_f32m2, 1.0f, vector_length);
    return __riscv_vfmacc_vv_f32m2(exponent_f32m2, poly_f32m2, m_minus_1_f32m2, vector_length);
}

/**
 *  @brief  Computes `log2(x)` for a vector of f64 values using the s-series approach.
 *
 *  Uses s = (m-1)/(m+1), then evaluates ln(m) = 2 × s × P(s²) with 14-term Horner polynomial.
 *  Converts to log2 via multiplication by log2(e). Matches Skylake's f64 log2 algorithm.
 */
NK_INTERNAL vfloat64m4_t nk_log2_f64m4_rvv_(vfloat64m4_t x, nk_size_t vector_length) {
    // Extract exponent and mantissa via bit manipulation
    vuint64m4_t bits_u64m4 = __riscv_vreinterpret_v_f64m4_u64m4(x);
    vuint64m4_t exponent_u64m4 = __riscv_vsrl_vx_u64m4(bits_u64m4, 52, vector_length);
    vint64m4_t exponent_i64m4 = __riscv_vsub_vx_i64m4(__riscv_vreinterpret_v_u64m4_i64m4(exponent_u64m4), 1023,
                                                      vector_length);
    vfloat64m4_t exponent_f64m4 = __riscv_vfcvt_f_x_v_f64m4(exponent_i64m4, vector_length);
    vuint64m4_t mantissa_u64m4 = __riscv_vor_vx_u64m4(
        __riscv_vand_vx_u64m4(bits_u64m4, 0x000FFFFFFFFFFFFFULL, vector_length), 0x3FF0000000000000ULL, vector_length);
    vfloat64m4_t mantissa_f64m4 = __riscv_vreinterpret_v_u64m4_f64m4(mantissa_u64m4);

    // s = (m - 1) / (m + 1)
    vfloat64m4_t one_f64m4 = __riscv_vfmv_v_f_f64m4(1.0, vector_length);
    vfloat64m4_t s_f64m4 = __riscv_vfdiv_vv_f64m4(__riscv_vfsub_vv_f64m4(mantissa_f64m4, one_f64m4, vector_length),
                                                  __riscv_vfadd_vv_f64m4(mantissa_f64m4, one_f64m4, vector_length),
                                                  vector_length);
    vfloat64m4_t s2_f64m4 = __riscv_vfmul_vv_f64m4(s_f64m4, s_f64m4, vector_length);

    // P(s²) = 1 + s²/3 + s⁴/5 + ... (14 terms, Horner's method)
    vfloat64m4_t poly_f64m4 = __riscv_vfmv_v_f_f64m4(1.0 / 27.0, vector_length); // 1/(2*13+1)
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(s2_f64m4, poly_f64m4, __riscv_vfmv_v_f_f64m4(1.0 / 25.0, vector_length),
                                         vector_length);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(s2_f64m4, poly_f64m4, __riscv_vfmv_v_f_f64m4(1.0 / 23.0, vector_length),
                                         vector_length);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(s2_f64m4, poly_f64m4, __riscv_vfmv_v_f_f64m4(1.0 / 21.0, vector_length),
                                         vector_length);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(s2_f64m4, poly_f64m4, __riscv_vfmv_v_f_f64m4(1.0 / 19.0, vector_length),
                                         vector_length);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(s2_f64m4, poly_f64m4, __riscv_vfmv_v_f_f64m4(1.0 / 17.0, vector_length),
                                         vector_length);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(s2_f64m4, poly_f64m4, __riscv_vfmv_v_f_f64m4(1.0 / 15.0, vector_length),
                                         vector_length);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(s2_f64m4, poly_f64m4, __riscv_vfmv_v_f_f64m4(1.0 / 13.0, vector_length),
                                         vector_length);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(s2_f64m4, poly_f64m4, __riscv_vfmv_v_f_f64m4(1.0 / 11.0, vector_length),
                                         vector_length);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(s2_f64m4, poly_f64m4, __riscv_vfmv_v_f_f64m4(1.0 / 9.0, vector_length),
                                         vector_length);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(s2_f64m4, poly_f64m4, __riscv_vfmv_v_f_f64m4(1.0 / 7.0, vector_length),
                                         vector_length);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(s2_f64m4, poly_f64m4, __riscv_vfmv_v_f_f64m4(1.0 / 5.0, vector_length),
                                         vector_length);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(s2_f64m4, poly_f64m4, __riscv_vfmv_v_f_f64m4(1.0 / 3.0, vector_length),
                                         vector_length);
    poly_f64m4 = __riscv_vfmadd_vv_f64m4(s2_f64m4, poly_f64m4, one_f64m4, vector_length);

    // ln(m) = 2 × s × P(s²), log2(m) = ln(m) × log2(e), log2(x) = exp + log2(m)
    vfloat64m4_t two_s_f64m4 = __riscv_vfmul_vf_f64m4(s_f64m4, 2.0, vector_length);
    vfloat64m4_t ln_m_f64m4 = __riscv_vfmul_vv_f64m4(two_s_f64m4, poly_f64m4, vector_length);
    vfloat64m4_t log2_f64m4 = __riscv_vfmul_vf_f64m4(ln_m_f64m4, NK_F64_LOG2E_, vector_length);
    return __riscv_vfadd_vv_f64m4(exponent_f64m4, log2_f64m4, vector_length);
}

#pragma region Kullback Leibler Divergence

NK_PUBLIC void nk_kld_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vector_length_max);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, b += vector_length) {
        vector_length = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t a_f32m2 = __riscv_vle32_v_f32m2(a, vector_length);
        vfloat32m2_t b_f32m2 = __riscv_vle32_v_f32m2(b, vector_length);
        // ratio = (a + ε) / (b + ε)
        vfloat32m2_t a_eps_f32m2 = __riscv_vfadd_vf_f32m2(a_f32m2, NK_F32_DIVISION_EPSILON, vector_length);
        vfloat32m2_t b_eps_f32m2 = __riscv_vfadd_vf_f32m2(b_f32m2, NK_F32_DIVISION_EPSILON, vector_length);
        vfloat32m2_t ratio_f32m2 = __riscv_vfmul_vv_f32m2(
            a_eps_f32m2, nk_f32m2_reciprocal_rvv_(b_eps_f32m2, vector_length), vector_length);
        // log2(ratio)
        vfloat32m2_t log_ratio_f32m2 = nk_log2_f32m2_rvv_(ratio_f32m2, vector_length);
        // contribution = a * log2(a / b)
        vfloat32m2_t contribution_f32m2 = __riscv_vfmul_vv_f32m2(a_f32m2, log_ratio_f32m2, vector_length);
        vfloat64m4_t contribution_f64m4 = __riscv_vfwcvt_f_f_v_f64m4(contribution_f32m2, vector_length);
        sum_f64m4 = __riscv_vfadd_vv_f64m4_tu(sum_f64m4, sum_f64m4, contribution_f64m4, vector_length);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *result = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero_f64m1, vector_length_max)) *
              NK_F64_LN2_;
}

NK_PUBLIC void nk_kld_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_size_t max_vector_length = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, max_vector_length);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, b += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        vfloat64m4_t a_f64m4 = __riscv_vle64_v_f64m4(a, vector_length);
        vfloat64m4_t b_f64m4 = __riscv_vle64_v_f64m4(b, vector_length);
        // ratio = (a + ε) / (b + ε) — full precision division
        vfloat64m4_t a_eps_f64m4 = __riscv_vfadd_vf_f64m4(a_f64m4, NK_F64_DIVISION_EPSILON, vector_length);
        vfloat64m4_t b_eps_f64m4 = __riscv_vfadd_vf_f64m4(b_f64m4, NK_F64_DIVISION_EPSILON, vector_length);
        vfloat64m4_t ratio_f64m4 = __riscv_vfdiv_vv_f64m4(a_eps_f64m4, b_eps_f64m4, vector_length);
        // log2(ratio)
        vfloat64m4_t log_ratio_f64m4 = nk_log2_f64m4_rvv_(ratio_f64m4, vector_length);
        // contribution = a * log2(a / b)
        vfloat64m4_t contribution_f64m4 = __riscv_vfmul_vv_f64m4(a_f64m4, log_ratio_f64m4, vector_length);
        // Per-lane accumulation
        sum_f64m4 = __riscv_vfadd_vv_f64m4_tu(sum_f64m4, sum_f64m4, contribution_f64m4, vector_length);
    }
    // Single horizontal reduction after loop
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    // Convert from log2 to ln by multiplying by ln(2)
    *result = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero_f64m1, max_vector_length)) *
              NK_F64_LN2_;
}

NK_PUBLIC void nk_kld_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t max_vector_length = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, max_vector_length);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, b += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        // Load f16 as raw u16 bits
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b, vector_length);
        // Convert f16 to f32 (m1 -> m2)
        vfloat32m2_t a_f32m2 = nk_f16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_f16m1_to_f32m2_rvv_(b_u16m1, vector_length);
        // ratio = (a + ε) / (b + ε)
        vfloat32m2_t a_eps_f32m2 = __riscv_vfadd_vf_f32m2(a_f32m2, NK_F32_DIVISION_EPSILON, vector_length);
        vfloat32m2_t b_eps_f32m2 = __riscv_vfadd_vf_f32m2(b_f32m2, NK_F32_DIVISION_EPSILON, vector_length);
        vfloat32m2_t ratio_f32m2 = __riscv_vfmul_vv_f32m2(
            a_eps_f32m2, nk_f32m2_reciprocal_rvv_(b_eps_f32m2, vector_length), vector_length);
        vfloat32m2_t log_ratio_f32m2 = nk_log2_f32m2_rvv_(ratio_f32m2, vector_length);
        // contribution = a * log2(a / b)
        vfloat32m2_t contribution_f32m2 = __riscv_vfmul_vv_f32m2(a_f32m2, log_ratio_f32m2, vector_length);
        // Per-lane accumulation
        sum_f32m2 = __riscv_vfadd_vv_f32m2_tu(sum_f32m2, sum_f32m2, contribution_f32m2, vector_length);
    }
    // Single horizontal reduction after loop
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(sum_f32m2, zero_f32m1, max_vector_length)) *
              NK_F32_LN2_;
}

NK_PUBLIC void nk_kld_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t max_vector_length = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, max_vector_length);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, b += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        // Load bf16 as raw u16 bits
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b, vector_length);
        // Convert bf16 to f32 (m1 -> m2)
        vfloat32m2_t a_f32m2 = nk_bf16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t b_f32m2 = nk_bf16m1_to_f32m2_rvv_(b_u16m1, vector_length);
        // ratio = (a + ε) / (b + ε)
        vfloat32m2_t a_eps_f32m2 = __riscv_vfadd_vf_f32m2(a_f32m2, NK_F32_DIVISION_EPSILON, vector_length);
        vfloat32m2_t b_eps_f32m2 = __riscv_vfadd_vf_f32m2(b_f32m2, NK_F32_DIVISION_EPSILON, vector_length);
        vfloat32m2_t ratio_f32m2 = __riscv_vfmul_vv_f32m2(
            a_eps_f32m2, nk_f32m2_reciprocal_rvv_(b_eps_f32m2, vector_length), vector_length);
        vfloat32m2_t log_ratio_f32m2 = nk_log2_f32m2_rvv_(ratio_f32m2, vector_length);
        // contribution = a * log2(a / b)
        vfloat32m2_t contribution_f32m2 = __riscv_vfmul_vv_f32m2(a_f32m2, log_ratio_f32m2, vector_length);
        // Per-lane accumulation
        sum_f32m2 = __riscv_vfadd_vv_f32m2_tu(sum_f32m2, sum_f32m2, contribution_f32m2, vector_length);
    }
    // Single horizontal reduction after loop
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *result = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(sum_f32m2, zero_f32m1, max_vector_length)) *
              NK_F32_LN2_;
}

#pragma endregion Kullback Leibler Divergence

#pragma region Jensen Shannon Divergence

NK_PUBLIC void nk_jsd_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_size_t vector_length_max = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vector_length_max);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, b += vector_length) {
        vector_length = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t va_f32m2 = __riscv_vle32_v_f32m2(a, vector_length);
        vfloat32m2_t vb_f32m2 = __riscv_vle32_v_f32m2(b, vector_length);
        // M = (a + b) / 2
        vfloat32m2_t mean_f32m2 = __riscv_vfmul_vf_f32m2(__riscv_vfadd_vv_f32m2(va_f32m2, vb_f32m2, vector_length),
                                                         0.5f, vector_length);
        // ratio_a = (a + eps) / (M + eps)
        vfloat32m2_t va_eps_f32m2 = __riscv_vfadd_vf_f32m2(va_f32m2, NK_F32_DIVISION_EPSILON, vector_length);
        vfloat32m2_t vb_eps_f32m2 = __riscv_vfadd_vf_f32m2(vb_f32m2, NK_F32_DIVISION_EPSILON, vector_length);
        vfloat32m2_t mean_eps_f32m2 = __riscv_vfadd_vf_f32m2(mean_f32m2, NK_F32_DIVISION_EPSILON, vector_length);
        vfloat32m2_t mean_reciprocal_f32m2 = nk_f32m2_reciprocal_rvv_(mean_eps_f32m2, vector_length);
        vfloat32m2_t ratio_a_f32m2 = __riscv_vfmul_vv_f32m2(va_eps_f32m2, mean_reciprocal_f32m2, vector_length);
        vfloat32m2_t ratio_b_f32m2 = __riscv_vfmul_vv_f32m2(vb_eps_f32m2, mean_reciprocal_f32m2, vector_length);
        // log2(ratio_a), log2(ratio_b)
        vfloat32m2_t log_ratio_a_f32m2 = nk_log2_f32m2_rvv_(ratio_a_f32m2, vector_length);
        vfloat32m2_t log_ratio_b_f32m2 = nk_log2_f32m2_rvv_(ratio_b_f32m2, vector_length);
        // contribution_a = a * log2(a / M), contribution_b = b * log2(b / M)
        vfloat32m2_t contrib_a_f32m2 = __riscv_vfmul_vv_f32m2(va_f32m2, log_ratio_a_f32m2, vector_length);
        vfloat32m2_t contrib_b_f32m2 = __riscv_vfmul_vv_f32m2(vb_f32m2, log_ratio_b_f32m2, vector_length);
        vfloat32m2_t contrib_f32m2 = __riscv_vfadd_vv_f32m2(contrib_a_f32m2, contrib_b_f32m2, vector_length);
        vfloat64m4_t contrib_f64m4 = __riscv_vfwcvt_f_f_v_f64m4(contrib_f32m2, vector_length);
        sum_f64m4 = __riscv_vfadd_vv_f64m4_tu(sum_f64m4, sum_f64m4, contrib_f64m4, vector_length);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f64_t sum = __riscv_vfmv_f_s_f64m1_f64(
                       __riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero_f64m1, vector_length_max)) *
                   NK_F64_LN2_ / 2.0;
    *result = sum > 0 ? nk_f64_sqrt_rvv(sum) : 0;
}

NK_PUBLIC void nk_jsd_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_size_t max_vector_length = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_a_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, max_vector_length);
    vfloat64m4_t sum_b_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, max_vector_length);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, b += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(n);
        vfloat64m4_t va_f64m4 = __riscv_vle64_v_f64m4(a, vector_length);
        vfloat64m4_t vb_f64m4 = __riscv_vle64_v_f64m4(b, vector_length);
        // M = (a + b) / 2
        vfloat64m4_t mean_f64m4 = __riscv_vfmul_vf_f64m4(__riscv_vfadd_vv_f64m4(va_f64m4, vb_f64m4, vector_length), 0.5,
                                                         vector_length);
        // ratio_a = (a + eps) / (M + eps), ratio_b = (b + eps) / (M + eps)
        vfloat64m4_t va_eps_f64m4 = __riscv_vfadd_vf_f64m4(va_f64m4, NK_F64_DIVISION_EPSILON, vector_length);
        vfloat64m4_t vb_eps_f64m4 = __riscv_vfadd_vf_f64m4(vb_f64m4, NK_F64_DIVISION_EPSILON, vector_length);
        vfloat64m4_t mean_eps_f64m4 = __riscv_vfadd_vf_f64m4(mean_f64m4, NK_F64_DIVISION_EPSILON, vector_length);
        // Full precision division (not reciprocal approximation)
        vfloat64m4_t ratio_a_f64m4 = __riscv_vfdiv_vv_f64m4(va_eps_f64m4, mean_eps_f64m4, vector_length);
        vfloat64m4_t ratio_b_f64m4 = __riscv_vfdiv_vv_f64m4(vb_eps_f64m4, mean_eps_f64m4, vector_length);
        // log2(ratio_a), log2(ratio_b)
        vfloat64m4_t log_ratio_a_f64m4 = nk_log2_f64m4_rvv_(ratio_a_f64m4, vector_length);
        vfloat64m4_t log_ratio_b_f64m4 = nk_log2_f64m4_rvv_(ratio_b_f64m4, vector_length);
        // contribution_a = a * log2(a / M), contribution_b = b * log2(b / M)
        sum_a_f64m4 = __riscv_vfmacc_vv_f64m4_tu(sum_a_f64m4, va_f64m4, log_ratio_a_f64m4, vector_length);
        sum_b_f64m4 = __riscv_vfmacc_vv_f64m4_tu(sum_b_f64m4, vb_f64m4, log_ratio_b_f64m4, vector_length);
    }
    // Single horizontal reduction after loop
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    // JSD = sqrt((sum_a + sum_b) * ln(2) / 2)
    nk_f64_t sum = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(
                       __riscv_vfadd_vv_f64m4(sum_a_f64m4, sum_b_f64m4, max_vector_length), zero_f64m1,
                       max_vector_length)) *
                   NK_F64_LN2_ / 2;
    *result = sum > 0 ? nk_f64_sqrt_rvv(sum) : 0;
}

NK_PUBLIC void nk_jsd_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t max_vector_length = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, max_vector_length);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, b += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        // Load f16 as raw u16 bits
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b, vector_length);
        // Convert f16 to f32 (m1 -> m2)
        vfloat32m2_t va_f32m2 = nk_f16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t vb_f32m2 = nk_f16m1_to_f32m2_rvv_(b_u16m1, vector_length);
        // M = (a + b) / 2
        vfloat32m2_t mean_f32m2 = __riscv_vfmul_vf_f32m2(__riscv_vfadd_vv_f32m2(va_f32m2, vb_f32m2, vector_length),
                                                         0.5f, vector_length);
        // ratio_a = (a + eps) / (M + eps), ratio_b = (b + eps) / (M + eps)
        vfloat32m2_t va_eps_f32m2 = __riscv_vfadd_vf_f32m2(va_f32m2, NK_F32_DIVISION_EPSILON, vector_length);
        vfloat32m2_t vb_eps_f32m2 = __riscv_vfadd_vf_f32m2(vb_f32m2, NK_F32_DIVISION_EPSILON, vector_length);
        vfloat32m2_t mean_eps_f32m2 = __riscv_vfadd_vf_f32m2(mean_f32m2, NK_F32_DIVISION_EPSILON, vector_length);
        vfloat32m2_t mean_reciprocal_f32m2 = nk_f32m2_reciprocal_rvv_(mean_eps_f32m2, vector_length);
        vfloat32m2_t ratio_a_f32m2 = __riscv_vfmul_vv_f32m2(va_eps_f32m2, mean_reciprocal_f32m2, vector_length);
        vfloat32m2_t ratio_b_f32m2 = __riscv_vfmul_vv_f32m2(vb_eps_f32m2, mean_reciprocal_f32m2, vector_length);
        vfloat32m2_t log_ratio_a_f32m2 = nk_log2_f32m2_rvv_(ratio_a_f32m2, vector_length);
        vfloat32m2_t log_ratio_b_f32m2 = nk_log2_f32m2_rvv_(ratio_b_f32m2, vector_length);
        // contribution_a = a * log2(a / M), contribution_b = b * log2(b / M)
        vfloat32m2_t contrib_a_f32m2 = __riscv_vfmul_vv_f32m2(va_f32m2, log_ratio_a_f32m2, vector_length);
        vfloat32m2_t contrib_b_f32m2 = __riscv_vfmul_vv_f32m2(vb_f32m2, log_ratio_b_f32m2, vector_length);
        vfloat32m2_t contrib_f32m2 = __riscv_vfadd_vv_f32m2(contrib_a_f32m2, contrib_b_f32m2, vector_length);
        // Per-lane accumulation
        sum_f32m2 = __riscv_vfadd_vv_f32m2_tu(sum_f32m2, sum_f32m2, contrib_f32m2, vector_length);
    }
    // Single horizontal reduction after loop
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    nk_f32_t sum = __riscv_vfmv_f_s_f32m1_f32(
                       __riscv_vfredusum_vs_f32m2_f32m1(sum_f32m2, zero_f32m1, max_vector_length)) *
                   NK_F32_LN2_ / 2;
    *result = sum > 0 ? nk_f32_sqrt_rvv(sum) : 0;
}

NK_PUBLIC void nk_jsd_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t max_vector_length = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t sum_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, max_vector_length);
    for (nk_size_t vector_length; n > 0; n -= vector_length, a += vector_length, b += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(n);
        // Load bf16 as raw u16 bits
        vuint16m1_t a_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)a, vector_length);
        vuint16m1_t b_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)b, vector_length);
        // Convert bf16 to f32 (m1 -> m2)
        vfloat32m2_t va_f32m2 = nk_bf16m1_to_f32m2_rvv_(a_u16m1, vector_length);
        vfloat32m2_t vb_f32m2 = nk_bf16m1_to_f32m2_rvv_(b_u16m1, vector_length);
        // M = (a + b) / 2
        vfloat32m2_t mean_f32m2 = __riscv_vfmul_vf_f32m2(__riscv_vfadd_vv_f32m2(va_f32m2, vb_f32m2, vector_length),
                                                         0.5f, vector_length);
        // ratio_a = (a + eps) / (M + eps), ratio_b = (b + eps) / (M + eps)
        vfloat32m2_t va_eps_f32m2 = __riscv_vfadd_vf_f32m2(va_f32m2, NK_F32_DIVISION_EPSILON, vector_length);
        vfloat32m2_t vb_eps_f32m2 = __riscv_vfadd_vf_f32m2(vb_f32m2, NK_F32_DIVISION_EPSILON, vector_length);
        vfloat32m2_t mean_eps_f32m2 = __riscv_vfadd_vf_f32m2(mean_f32m2, NK_F32_DIVISION_EPSILON, vector_length);
        vfloat32m2_t mean_reciprocal_f32m2 = nk_f32m2_reciprocal_rvv_(mean_eps_f32m2, vector_length);
        vfloat32m2_t ratio_a_f32m2 = __riscv_vfmul_vv_f32m2(va_eps_f32m2, mean_reciprocal_f32m2, vector_length);
        vfloat32m2_t ratio_b_f32m2 = __riscv_vfmul_vv_f32m2(vb_eps_f32m2, mean_reciprocal_f32m2, vector_length);
        vfloat32m2_t log_ratio_a_f32m2 = nk_log2_f32m2_rvv_(ratio_a_f32m2, vector_length);
        vfloat32m2_t log_ratio_b_f32m2 = nk_log2_f32m2_rvv_(ratio_b_f32m2, vector_length);
        // contribution_a = a * log2(a / M), contribution_b = b * log2(b / M)
        vfloat32m2_t contrib_a_f32m2 = __riscv_vfmul_vv_f32m2(va_f32m2, log_ratio_a_f32m2, vector_length);
        vfloat32m2_t contrib_b_f32m2 = __riscv_vfmul_vv_f32m2(vb_f32m2, log_ratio_b_f32m2, vector_length);
        vfloat32m2_t contrib_f32m2 = __riscv_vfadd_vv_f32m2(contrib_a_f32m2, contrib_b_f32m2, vector_length);
        // Per-lane accumulation
        sum_f32m2 = __riscv_vfadd_vv_f32m2_tu(sum_f32m2, sum_f32m2, contrib_f32m2, vector_length);
    }
    // Single horizontal reduction after loop
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    nk_f32_t sum = __riscv_vfmv_f_s_f32m1_f32(
                       __riscv_vfredusum_vs_f32m2_f32m1(sum_f32m2, zero_f32m1, max_vector_length)) *
                   NK_F32_LN2_ / 2;
    *result = sum > 0 ? nk_f32_sqrt_rvv(sum) : 0;
}

#pragma endregion Jensen Shannon Divergence

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_RVV
#endif // NK_TARGET_RISCV64_
#endif // NK_PROBABILITY_RVV_H
