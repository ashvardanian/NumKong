/**
 *  @brief SIMD-accelerated Type Conversions for RISC-V.
 *  @file include/numkong/cast/rvv.h
 *  @author Ash Vardanian
 *  @date January 13, 2026
 *
 *  @sa include/numkong/cast.h
 *
 *  SpacemiT K1 and similar chips implement RVA22 profile with base RVV 1.0.
 *  This file provides vectorized type conversions for:
 *  - BF16 ↔ F32 (bit manipulation, no hardware support)
 *  - F16 ↔ F32 (bit manipulation, no hardware support)
 *  - E4M3 ↔ F32 (FP8 format for ML inference)
 *  - E5M2 ↔ F32 (FP8 format for ML training)
 *  - i4/u4 unpacking to i8/u8
 *
 *  Mini-float conversions use sign-symmetric magnitude LUTs: every mini-float
 *  format is sign|magnitude, so we store only the positive-half (magnitude)
 *  entries and extract the sign bit separately. This cuts LUT memory by 50-87%
 *  and fixes the E2M3FN NaN bug (E2M3FN has NO NaN; index 31 is +7.5, not NaN).
 *
 *  8-bit formats (e4m3, e5m2): sign = bit 7, magnitude = bits 6:0 (128 entries)
 *  6-bit formats (e2m3, e3m2): sign = bit 5, magnitude = bits 4:0 (32 entries)
 *
 *  @section rvv_cast_instructions Key RVV Cast Instructions
 *
 *      Intrinsic                       Purpose
 *      vzext_vf4_u32m4                 Zero-extend u8 → u32 (4x widening)
 *      vsext_vf4_i32m4                 Sign-extend i8 → i32 (4x widening)
 *      vsll_vx / vsrl_vx               Bit shifts for field extraction
 *      vand_vx                         Bit masking
 *      vor_vv                          Combining bit fields
 *      vfcvt_f_xu_v                    Unsigned int → float
 *      vmseq_vx                        Compare for conditional selection
 *      vmerge_vvm                      Conditional select (blend)
 */
#ifndef NK_CAST_RVV_H
#define NK_CAST_RVV_H

#if NK_TARGET_RISCV64_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_cast_serial`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Register to Register Helpers

/**
 *  @brief Convert bf16 (m1) to f32 (m2) register-to-register.
 *
 *  BF16 is the upper 16 bits of F32 (same sign + exponent + top 7 mantissa bits).
 *  Conversion is simply: f32_bits = bf16_bits << 16.
 */
NK_INTERNAL vfloat32m2_t nk_bf16m1_to_f32m2_rvv_(vuint16m1_t bf16_u16m1, nk_size_t vector_length) {
    vuint32m2_t bits_u32m2 = __riscv_vzext_vf2_u32m2(bf16_u16m1, vector_length);
    bits_u32m2 = __riscv_vsll_vx_u32m2(bits_u32m2, 16, vector_length);
    return __riscv_vreinterpret_v_u32m2_f32m2(bits_u32m2);
}

/**
 *  @brief Convert f32 (m2) to bf16 (m1) register-to-register.
 *
 *  Conversion with round-to-nearest-even (RNE): add (0x7FFF + lsb) to match hardware BF16 behavior.
 */
NK_INTERNAL vuint16m1_t nk_f32m2_to_bf16m1_rvv_(vfloat32m2_t f32_f32m2, nk_size_t vector_length) {
    vuint32m2_t bits_u32m2 = __riscv_vreinterpret_v_f32m2_u32m2(f32_f32m2);
    // Extract LSB of result (bit 16) for round-to-nearest-even
    vuint32m2_t lsb_u32m2 = __riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(bits_u32m2, 16, vector_length), 1,
                                                  vector_length);
    vuint32m2_t rounding_u32m2 = __riscv_vadd_vx_u32m2(lsb_u32m2, 0x7FFF, vector_length);
    vuint32m2_t rounded_u32m2 = __riscv_vadd_vv_u32m2(bits_u32m2, rounding_u32m2, vector_length);
    vuint32m2_t shifted_u32m2 = __riscv_vsrl_vx_u32m2(rounded_u32m2, 16, vector_length);
    return __riscv_vncvt_x_x_w_u16m1(shifted_u32m2, vector_length);
}

/**
 *  @brief Convert f16 (m1) to f32 (m2) register-to-register.
 *
 *  F16 format: S EEEEE MMMMMMMMMM (1 sign, 5 exponent bits with bias=15, 10 mantissa bits)
 *  F32 format: S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM (1 sign, 8 exponent bits with bias=127, 23 mantissa bits)
 *
 *  Uses the Giesen magic-multiply trick: treat the magnitude bits as a denormal f32 and
 *  multiply by 2^112 to rebias the exponent.  This correctly handles ±zero, denormals,
 *  and normals in a single FP multiply; only inf/NaN needs a fixup compare+merge.
 *
 *  https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
 */
NK_INTERNAL vfloat32m2_t nk_f16m1_to_f32m2_rvv_(vuint16m1_t f16_u16m1, nk_size_t vector_length) {
    // Widen to 32-bit for manipulation
    vuint32m2_t bits_u32m2 = __riscv_vzext_vf2_u32m2(f16_u16m1, vector_length);
    // Extract sign: (raw >> 15) << 31
    vuint32m2_t sign_u32m2 = __riscv_vsll_vx_u32m2(__riscv_vsrl_vx_u32m2(bits_u32m2, 15, vector_length), 31,
                                                   vector_length);
    // Strip sign, shift magnitude into f32 mantissa position.
    // For a normal f16 with exp E, this places E into the f32 exponent field,
    // creating a tiny f32 whose value is proportional to the f16 magnitude.
    vuint32m2_t nonsign_u32m2 = __riscv_vand_vx_u32m2(bits_u32m2, 0x7FFF, vector_length);
    vuint32m2_t shifted_u32m2 = __riscv_vsll_vx_u32m2(nonsign_u32m2, 13, vector_length);
    // Multiply by 2^112 (= magic 0x77800000 as f32) to rebias the exponent.
    // This single multiply correctly handles zero, denormals, and normals:
    //   zero:     0.0 × 2^112 = 0.0
    //   denormal: (M × 2^-136) × 2^112 = M × 2^-24  (correct f16 denormal value)
    //   normal:   (2^(E-127) × …) × 2^112 = 2^(E-15) × …  (correct rebiased value)
    vfloat32m2_t magic_f32m2 = __riscv_vreinterpret_v_u32m2_f32m2(
        __riscv_vmv_v_x_u32m2(((nk_u32_t)(254 - 15) << 23), vector_length));
    vfloat32m2_t result_f32m2 = __riscv_vfmul_vv_f32m2(__riscv_vreinterpret_v_u32m2_f32m2(shifted_u32m2), magic_f32m2,
                                                       vector_length);
    // Inf/NaN fixup: the multiply maps f16 exp=31 to a large finite f32.
    // Detect those lanes and force the f32 exponent to 255 (inf/NaN).
    // Threshold 0x47800000 = 2^16; any f16 with exp=31 exceeds it after scaling.
    vfloat32m2_t infnan_threshold_f32m2 = __riscv_vreinterpret_v_u32m2_f32m2(
        __riscv_vmv_v_x_u32m2(((nk_u32_t)(127 + 16) << 23), vector_length));
    vbool16_t is_infnan = __riscv_vmfge_vv_f32m2_b16(result_f32m2, infnan_threshold_f32m2, vector_length);
    vuint32m2_t result_u32m2 = __riscv_vreinterpret_v_f32m2_u32m2(result_f32m2);
    vuint32m2_t fixed_u32m2 = __riscv_vor_vx_u32m2(result_u32m2, 0x7F800000, vector_length);
    result_u32m2 = __riscv_vmerge_vvm_u32m2(result_u32m2, fixed_u32m2, is_infnan, vector_length);
    // Restore sign
    result_u32m2 = __riscv_vor_vv_u32m2(result_u32m2, sign_u32m2, vector_length);
    return __riscv_vreinterpret_v_u32m2_f32m2(result_u32m2);
}

/**
 *  @brief Convert f32 (m2) to f16 (m1) register-to-register.
 *
 *  Conversion: Rebias exponent from 127 to 15, truncate mantissa from 23 to 10 bits with rounding.
 */
NK_INTERNAL vuint16m1_t nk_f32m2_to_f16m1_rvv_(vfloat32m2_t f32_f32m2, nk_size_t vector_length) {
    // Full IEEE-754 f32→f16 with round-to-nearest-even, mirroring nk_f32_to_f16_serial. Each lane's
    // f32 exponent falls in exactly one bucket; per-bucket f16 magnitudes are computed unconditionally
    // and merged by mask. Sign is applied last; the result is narrowed u32m2 → u16m1.
    nk_size_t vl = vector_length;
    vuint32m2_t bits = __riscv_vreinterpret_v_f32m2_u32m2(f32_f32m2);
    vuint32m2_t sign = __riscv_vsll_vx_u32m2(__riscv_vsrl_vx_u32m2(bits, 31, vl), 15, vl);
    vuint32m2_t exp = __riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(bits, 23, vl), 0xFF, vl);
    vuint32m2_t mant = __riscv_vand_vx_u32m2(bits, 0x7FFFFF, vl);
    vuint32m2_t zeros = __riscv_vmv_v_x_u32m2(0, vl);

    // Exponent buckets: under [≤102], denormal [103..112], normal [113..142], overflow [143..255],
    // inf/nan [255]. exp==0 falls under "under" but yields 0 there, matching the serial zero case.
    vbool16_t m_under = __riscv_vmsleu_vx_u32m2_b16(exp, 102, vl);
    vbool16_t m_denorm = __riscv_vmand_mm_b16(__riscv_vmsgtu_vx_u32m2_b16(exp, 102, vl),
                                              __riscv_vmsleu_vx_u32m2_b16(exp, 112, vl), vl);
    vbool16_t m_normal = __riscv_vmand_mm_b16(__riscv_vmsgtu_vx_u32m2_b16(exp, 112, vl),
                                              __riscv_vmsleu_vx_u32m2_b16(exp, 142, vl), vl);
    vbool16_t m_over = __riscv_vmsgtu_vx_u32m2_b16(exp, 142, vl);
    vbool16_t m_infnan = __riscv_vmseq_vx_u32m2_b16(exp, 255, vl);

    // Inf/NaN: 0x7C00 | (mant>>13), bumping a truncated-to-zero NaN payload back to 1.
    vuint32m2_t payload = __riscv_vsrl_vx_u32m2(mant, 13, vl);
    vbool16_t need_qnan = __riscv_vmand_mm_b16(__riscv_vmsne_vx_u32m2_b16(mant, 0, vl),
                                               __riscv_vmseq_vx_u32m2_b16(payload, 0, vl), vl);
    payload = __riscv_vmerge_vxm_u32m2(payload, 1, need_qnan, vl);
    vuint32m2_t res_infnan = __riscv_vor_vx_u32m2(payload, 0x7C00, vl);

    // Underflow: smallest denormal (1) only at exp==102 with nonzero mantissa, else zero.
    vbool16_t under_one = __riscv_vmand_mm_b16(__riscv_vmseq_vx_u32m2_b16(exp, 102, vl),
                                               __riscv_vmsne_vx_u32m2_b16(mant, 0, vl), vl);
    vuint32m2_t res_under = __riscv_vmerge_vxm_u32m2(zeros, 1, under_one, vl);

    // f16 denormal range (exp 103..112): variable right-shift by sa = 126-exp (14..23) with RNE.
    vuint32m2_t sa = __riscv_vrsub_vx_u32m2(exp, 126, vl);
    vuint32m2_t sa1 = __riscv_vsub_vx_u32m2(sa, 1, vl);
    vuint32m2_t full = __riscv_vor_vx_u32m2(mant, 0x800000, vl);
    vuint32m2_t dmant = __riscv_vsrl_vv_u32m2(full, sa, vl);
    vuint32m2_t d_round = __riscv_vand_vx_u32m2(__riscv_vsrl_vv_u32m2(full, sa1, vl), 1, vl);
    vuint32m2_t d_stickymask = __riscv_vsub_vx_u32m2(__riscv_vsll_vv_u32m2(__riscv_vmv_v_x_u32m2(1, vl), sa1, vl), 1,
                                                     vl);
    vuint32m2_t d_sticky = __riscv_vand_vv_u32m2(full, d_stickymask, vl);
    vbool16_t d_roundup = __riscv_vmand_mm_b16(
        __riscv_vmsne_vx_u32m2_b16(d_round, 0, vl),
        __riscv_vmor_mm_b16(__riscv_vmsne_vx_u32m2_b16(d_sticky, 0, vl),
                            __riscv_vmsne_vx_u32m2_b16(__riscv_vand_vx_u32m2(dmant, 1, vl), 0, vl), vl),
        vl);
    vuint32m2_t res_denorm = __riscv_vadd_vv_u32m2(dmant, __riscv_vmerge_vxm_u32m2(zeros, 1, d_roundup, vl), vl);

    // Normal range (exp 113..142): rebias to exp-112, mantissa >>13 with RNE; carry can push to overflow.
    vuint32m2_t f16_exp = __riscv_vsub_vx_u32m2(exp, 112, vl);
    vuint32m2_t f16_mant = __riscv_vsrl_vx_u32m2(mant, 13, vl);
    vuint32m2_t n_round = __riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(mant, 12, vl), 1, vl);
    vuint32m2_t n_sticky = __riscv_vand_vx_u32m2(mant, 0xFFF, vl);
    vbool16_t n_roundup = __riscv_vmand_mm_b16(
        __riscv_vmsne_vx_u32m2_b16(n_round, 0, vl),
        __riscv_vmor_mm_b16(__riscv_vmsne_vx_u32m2_b16(n_sticky, 0, vl),
                            __riscv_vmsne_vx_u32m2_b16(__riscv_vand_vx_u32m2(f16_mant, 1, vl), 0, vl), vl),
        vl);
    f16_mant = __riscv_vadd_vv_u32m2(f16_mant, __riscv_vmerge_vxm_u32m2(zeros, 1, n_roundup, vl), vl);
    vbool16_t n_carry = __riscv_vmsgtu_vx_u32m2_b16(f16_mant, 0x3FF, vl);
    f16_mant = __riscv_vmerge_vxm_u32m2(f16_mant, 0, n_carry, vl);
    f16_exp = __riscv_vadd_vv_u32m2(f16_exp, __riscv_vmerge_vxm_u32m2(zeros, 1, n_carry, vl), vl);
    vuint32m2_t res_normal = __riscv_vor_vv_u32m2(__riscv_vsll_vx_u32m2(f16_exp, 10, vl), f16_mant, vl);
    res_normal = __riscv_vmerge_vxm_u32m2(res_normal, 0x7C00, __riscv_vmsgtu_vx_u32m2_b16(f16_exp, 30, vl), vl);

    // Merge buckets (default 0 = zero/underflow-to-zero); inf/nan merged last to win over overflow at exp==255.
    vuint32m2_t res = zeros;
    res = __riscv_vmerge_vvm_u32m2(res, res_under, m_under, vl);
    res = __riscv_vmerge_vvm_u32m2(res, res_denorm, m_denorm, vl);
    res = __riscv_vmerge_vvm_u32m2(res, res_normal, m_normal, vl);
    res = __riscv_vmerge_vxm_u32m2(res, 0x7C00, m_over, vl);
    res = __riscv_vmerge_vvm_u32m2(res, res_infnan, m_infnan, vl);
    res = __riscv_vor_vv_u32m2(res, sign, vl);
    return __riscv_vncvt_x_x_w_u16m1(res, vl);
}

/**
 *  @brief Convert e4m3 (m1) to f32 (m4) via Giesen magic-multiply.
 *  Reinterprets magnitude bits as a tiny f32, then multiplies by 2^(127-bias) to rebias.
 *  Handles zero, subnormals, and normals in a single vfmul. NaN fixup for magnitude 0x7F.
 *  https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
 */
NK_INTERNAL vfloat32m4_t nk_e4m3m1_to_f32m4_rvv_(vuint8m1_t e4m3_u8m1, nk_size_t vector_length) {
    // Extract sign: (raw & 0x80) → bit 7, shift to bit 31
    vuint32m4_t sign_u32m4 = __riscv_vsll_vx_u32m4(
        __riscv_vzext_vf4_u32m4(__riscv_vand_vx_u8m1(e4m3_u8m1, 0x80, vector_length), vector_length), 24,
        vector_length);
    // Strip sign to get 7-bit magnitude, widen to u32, shift left by 20
    vuint8m1_t nonsign_u8m1 = __riscv_vand_vx_u8m1(e4m3_u8m1, 0x7F, vector_length);
    vuint32m4_t nonsign_u32m4 = __riscv_vzext_vf4_u32m4(nonsign_u8m1, vector_length);
    vuint32m4_t shifted_u32m4 = __riscv_vsll_vx_u32m4(nonsign_u32m4, 20, vector_length);

    // Magic multiply: reinterpret as f32 × 2^120 rebiases from E4M3 (bias=7) to f32 (bias=127).
    vfloat32m4_t magic_f32m4 = __riscv_vreinterpret_v_u32m4_f32m4(
        __riscv_vmv_v_x_u32m4(0x7B800000, vector_length)); // 2^120 = (254-7)<<23
    vfloat32m4_t result_f32m4 = __riscv_vfmul_vv_f32m4(__riscv_vreinterpret_v_u32m4_f32m4(shifted_u32m4), magic_f32m4,
                                                       vector_length);

    // NaN fixup: masked OR writes sign|0x7FC00000 only into NaN lanes
    vbool8_t is_nan = __riscv_vmseq_vx_u8m1_b8(nonsign_u8m1, 0x7F, vector_length);
    vuint32m4_t result_u32m4 = __riscv_vor_vx_u32m4_mu(is_nan, __riscv_vreinterpret_v_f32m4_u32m4(result_f32m4),
                                                       sign_u32m4, 0x7FC00000, vector_length);

    // Restore sign
    result_u32m4 = __riscv_vor_vv_u32m4(result_u32m4, sign_u32m4, vector_length);
    return __riscv_vreinterpret_v_u32m4_f32m4(result_u32m4);
}

/**
 *  @brief Convert e5m2 (m1) to f32 (m4) via Giesen magic-multiply.
 *  Reinterprets magnitude bits as a tiny f32, then multiplies by 2^(127-bias) to rebias.
 *  Handles zero, subnormals, and normals in a single vfmul. Inf/NaN fixup for exp=31.
 *  https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
 */
NK_INTERNAL vfloat32m4_t nk_e5m2m1_to_f32m4_rvv_(vuint8m1_t e5m2_u8m1, nk_size_t vector_length) {
    // Extract sign: (raw & 0x80) → bit 7, shift to bit 31
    vuint32m4_t sign_u32m4 = __riscv_vsll_vx_u32m4(
        __riscv_vzext_vf4_u32m4(__riscv_vand_vx_u8m1(e5m2_u8m1, 0x80, vector_length), vector_length), 24,
        vector_length);
    // Strip sign to get 7-bit magnitude, widen to u32, shift left by 21
    vuint32m4_t nonsign_u32m4 = __riscv_vzext_vf4_u32m4(__riscv_vand_vx_u8m1(e5m2_u8m1, 0x7F, vector_length),
                                                        vector_length);
    vuint32m4_t shifted_u32m4 = __riscv_vsll_vx_u32m4(nonsign_u32m4, 21, vector_length);

    // Magic multiply: reinterpret as f32 × 2^112 rebiases from E5M2 (bias=15) to f32 (bias=127).
    vfloat32m4_t magic_f32m4 = __riscv_vreinterpret_v_u32m4_f32m4(
        __riscv_vmv_v_x_u32m4(0x77800000, vector_length)); // 2^112 = (254-15)<<23
    vfloat32m4_t result_f32m4 = __riscv_vfmul_vv_f32m4(__riscv_vreinterpret_v_u32m4_f32m4(shifted_u32m4), magic_f32m4,
                                                       vector_length);

    // Inf/NaN fixup: masked OR writes 0x7F800000 only into inf/NaN lanes (nonsign > 123)
    vbool8_t is_infnan = __riscv_vmsgtu_vx_u32m4_b8(nonsign_u32m4, 123, vector_length);
    vuint32m4_t result_u32m4 = __riscv_vor_vx_u32m4_mu(is_infnan, __riscv_vreinterpret_v_f32m4_u32m4(result_f32m4),
                                                       __riscv_vreinterpret_v_f32m4_u32m4(result_f32m4), 0x7F800000,
                                                       vector_length);

    // Restore sign
    result_u32m4 = __riscv_vor_vv_u32m4(result_u32m4, sign_u32m4, vector_length);
    return __riscv_vreinterpret_v_u32m4_f32m4(result_u32m4);
}

/**
 *  @brief Convert e2m3 (m1) to f32 (m4) via Giesen magic-multiply.
 *  Reinterprets magnitude bits as a tiny f32, then multiplies by 2^(127-bias) to rebias.
 *  Handles zero, subnormals, and normals in a single vfmul. No inf/NaN in E2M3FN.
 *  https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
 */
NK_INTERNAL vfloat32m4_t nk_e2m3m1_to_f32m4_rvv_(vuint8m1_t e2m3_u8m1, nk_size_t vector_length) {
    // Extract sign: bit 5 → bit 31
    vuint32m4_t sign_u32m4 = __riscv_vsll_vx_u32m4(
        __riscv_vzext_vf4_u32m4(__riscv_vand_vx_u8m1(e2m3_u8m1, 0x20, vector_length), vector_length), 26,
        vector_length);
    // Strip sign to get 5-bit magnitude, widen to u32, shift left by 20
    vuint32m4_t nonsign_u32m4 = __riscv_vzext_vf4_u32m4(__riscv_vand_vx_u8m1(e2m3_u8m1, 0x1F, vector_length),
                                                        vector_length);
    vuint32m4_t shifted_u32m4 = __riscv_vsll_vx_u32m4(nonsign_u32m4, 20, vector_length);

    // Magic multiply: reinterpret as f32 × 2^126 rebiases from E2M3 (bias=1) to f32 (bias=127).
    vfloat32m4_t magic_f32m4 = __riscv_vreinterpret_v_u32m4_f32m4(
        __riscv_vmv_v_x_u32m4(0x7E800000, vector_length)); // 2^126 = (254-1)<<23
    vfloat32m4_t result_f32m4 = __riscv_vfmul_vv_f32m4(__riscv_vreinterpret_v_u32m4_f32m4(shifted_u32m4), magic_f32m4,
                                                       vector_length);

    // Restore sign (no inf/NaN fixup needed for E2M3FN)
    vuint32m4_t result_u32m4 = __riscv_vor_vv_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(result_f32m4), sign_u32m4,
                                                    vector_length);
    return __riscv_vreinterpret_v_u32m4_f32m4(result_u32m4);
}

/**
 *  @brief Convert e3m2 (m1) to f32 (m4) via Giesen magic-multiply.
 *  Reinterprets magnitude bits as a tiny f32, then multiplies by 2^(127-bias) to rebias.
 *  Handles zero, subnormals, and normals in a single vfmul. No inf/NaN in E3M2FN.
 *  https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
 */
NK_INTERNAL vfloat32m4_t nk_e3m2m1_to_f32m4_rvv_(vuint8m1_t e3m2_u8m1, nk_size_t vector_length) {
    // Extract sign: bit 5 → bit 31
    vuint32m4_t sign_u32m4 = __riscv_vsll_vx_u32m4(
        __riscv_vzext_vf4_u32m4(__riscv_vand_vx_u8m1(e3m2_u8m1, 0x20, vector_length), vector_length), 26,
        vector_length);
    // Strip sign to get 5-bit magnitude, widen to u32, shift left by 21
    vuint32m4_t nonsign_u32m4 = __riscv_vzext_vf4_u32m4(__riscv_vand_vx_u8m1(e3m2_u8m1, 0x1F, vector_length),
                                                        vector_length);
    vuint32m4_t shifted_u32m4 = __riscv_vsll_vx_u32m4(nonsign_u32m4, 21, vector_length);

    // Magic multiply: reinterpret as f32 × 2^124 rebiases from E3M2 (bias=3) to f32 (bias=127).
    vfloat32m4_t magic_f32m4 = __riscv_vreinterpret_v_u32m4_f32m4(
        __riscv_vmv_v_x_u32m4(0x7D800000, vector_length)); // 2^124 = (254-3)<<23
    vfloat32m4_t result_f32m4 = __riscv_vfmul_vv_f32m4(__riscv_vreinterpret_v_u32m4_f32m4(shifted_u32m4), magic_f32m4,
                                                       vector_length);

    // Restore sign (no inf/NaN fixup needed for E3M2FN)
    vuint32m4_t result_u32m4 = __riscv_vor_vv_u32m4(__riscv_vreinterpret_v_f32m4_u32m4(result_f32m4), sign_u32m4,
                                                    vector_length);
    return __riscv_vreinterpret_v_u32m4_f32m4(result_u32m4);
}

/** @brief Convert e4m3 (m1) to bf16 (m2) via Giesen magic-multiply.
 *  Magic-multiply to f32, truncate upper 16 bits to bf16. NaN fixup for magnitude 0x7F. */
NK_INTERNAL vuint16m2_t nk_e4m3m1_to_bf16m2_rvv_(vuint8m1_t e4m3_u8m1, nk_size_t vector_length) {
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e4m3_u8m1, 0x80, vector_length);
    vuint8m1_t nonsign_u8m1 = __riscv_vand_vx_u8m1(e4m3_u8m1, 0x7F, vector_length);
    vuint32m4_t nonsign_u32m4 = __riscv_vzext_vf4_u32m4(nonsign_u8m1, vector_length);
    vuint32m4_t shifted_u32m4 = __riscv_vsll_vx_u32m4(nonsign_u32m4, 20, vector_length);
    // Magic multiply: reinterpret as f32 × 2^120
    vfloat32m4_t magic_f32m4 = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vmv_v_x_u32m4(0x7B800000, vector_length));
    vfloat32m4_t result_f32m4 = __riscv_vfmul_vv_f32m4(__riscv_vreinterpret_v_u32m4_f32m4(shifted_u32m4), magic_f32m4,
                                                       vector_length);
    // Truncate f32 → bf16 (right shift 16, exact for all e4m3 values)
    vuint16m2_t result_u16m2 = __riscv_vnsrl_wx_u16m2(__riscv_vreinterpret_v_f32m4_u32m4(result_f32m4), 16,
                                                      vector_length);
    // NaN fixup: magnitude 0x7F → bf16 quiet NaN 0x7FC0
    vbool8_t is_nan = __riscv_vmseq_vx_u8m1_b8(nonsign_u8m1, 0x7F, vector_length);
    result_u16m2 = __riscv_vmerge_vxm_u16m2(result_u16m2, 0x7FC0, is_nan, vector_length);
    // Restore sign: bit 7 → bf16 bit 15 (<<8)
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 8, vector_length);
    return __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
}

/** @brief Convert e5m2 (m1) to bf16 (m2) via Giesen magic-multiply.
 *  Magic-multiply to f32, inf/NaN fixup, truncate upper 16 bits to bf16. */
NK_INTERNAL vuint16m2_t nk_e5m2m1_to_bf16m2_rvv_(vuint8m1_t e5m2_u8m1, nk_size_t vector_length) {
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e5m2_u8m1, 0x80, vector_length);
    vuint8m1_t nonsign_u8m1 = __riscv_vand_vx_u8m1(e5m2_u8m1, 0x7F, vector_length);
    vuint32m4_t nonsign_u32m4 = __riscv_vzext_vf4_u32m4(nonsign_u8m1, vector_length);
    vuint32m4_t shifted_u32m4 = __riscv_vsll_vx_u32m4(nonsign_u32m4, 21, vector_length);
    // Magic multiply: reinterpret as f32 × 2^112
    vfloat32m4_t magic_f32m4 = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vmv_v_x_u32m4(0x77800000, vector_length));
    vfloat32m4_t result_f32m4 = __riscv_vfmul_vv_f32m4(__riscv_vreinterpret_v_u32m4_f32m4(shifted_u32m4), magic_f32m4,
                                                       vector_length);
    // Inf/NaN fixup: masked OR writes 0x7F800000 only into inf/NaN lanes (nonsign > 123)
    vbool8_t is_infnan = __riscv_vmsgtu_vx_u32m4_b8(nonsign_u32m4, 123, vector_length);
    vuint32m4_t f32_bits = __riscv_vor_vx_u32m4_mu(is_infnan, __riscv_vreinterpret_v_f32m4_u32m4(result_f32m4),
                                                   __riscv_vreinterpret_v_f32m4_u32m4(result_f32m4), 0x7F800000,
                                                   vector_length);
    // Truncate f32 → bf16 (right shift 16, exact for all e5m2 values)
    vuint16m2_t result_u16m2 = __riscv_vnsrl_wx_u16m2(f32_bits, 16, vector_length);
    // Restore sign: bit 7 → bf16 bit 15 (<<8)
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 8, vector_length);
    return __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
}

/** @brief Convert e2m3 (m1) to bf16 (m2) via Giesen magic-multiply.
 *  Magic-multiply to f32, truncate upper 16 bits to bf16. No inf/NaN in E2M3FN. */
NK_INTERNAL vuint16m2_t nk_e2m3m1_to_bf16m2_rvv_(vuint8m1_t e2m3_u8m1, nk_size_t vector_length) {
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e2m3_u8m1, 0x20, vector_length);
    vuint8m1_t nonsign_u8m1 = __riscv_vand_vx_u8m1(e2m3_u8m1, 0x1F, vector_length);
    vuint32m4_t nonsign_u32m4 = __riscv_vzext_vf4_u32m4(nonsign_u8m1, vector_length);
    vuint32m4_t shifted_u32m4 = __riscv_vsll_vx_u32m4(nonsign_u32m4, 20, vector_length);
    // Magic multiply: reinterpret as f32 × 2^126
    vfloat32m4_t magic_f32m4 = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vmv_v_x_u32m4(0x7E800000, vector_length));
    vfloat32m4_t result_f32m4 = __riscv_vfmul_vv_f32m4(__riscv_vreinterpret_v_u32m4_f32m4(shifted_u32m4), magic_f32m4,
                                                       vector_length);
    // Truncate f32 → bf16 (right shift 16, exact for all e2m3 values)
    vuint16m2_t result_u16m2 = __riscv_vnsrl_wx_u16m2(__riscv_vreinterpret_v_f32m4_u32m4(result_f32m4), 16,
                                                      vector_length);
    // Restore sign: bit 5 → bf16 bit 15 (<<10)
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 10,
                                                   vector_length);
    return __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
}

/** @brief Convert e3m2 (m1) to bf16 (m2) via Giesen magic-multiply.
 *  Magic-multiply to f32, truncate upper 16 bits to bf16. No inf/NaN in E3M2FN. */
NK_INTERNAL vuint16m2_t nk_e3m2m1_to_bf16m2_rvv_(vuint8m1_t e3m2_u8m1, nk_size_t vector_length) {
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e3m2_u8m1, 0x20, vector_length);
    vuint8m1_t nonsign_u8m1 = __riscv_vand_vx_u8m1(e3m2_u8m1, 0x1F, vector_length);
    vuint32m4_t nonsign_u32m4 = __riscv_vzext_vf4_u32m4(nonsign_u8m1, vector_length);
    vuint32m4_t shifted_u32m4 = __riscv_vsll_vx_u32m4(nonsign_u32m4, 21, vector_length);
    // Magic multiply: reinterpret as f32 × 2^124
    vfloat32m4_t magic_f32m4 = __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vmv_v_x_u32m4(0x7D800000, vector_length));
    vfloat32m4_t result_f32m4 = __riscv_vfmul_vv_f32m4(__riscv_vreinterpret_v_u32m4_f32m4(shifted_u32m4), magic_f32m4,
                                                       vector_length);
    // Truncate f32 → bf16 (right shift 16, exact for all e3m2 values)
    vuint16m2_t result_u16m2 = __riscv_vnsrl_wx_u16m2(__riscv_vreinterpret_v_f32m4_u32m4(result_f32m4), 16,
                                                      vector_length);
    // Restore sign: bit 5 → bf16 bit 15 (<<10)
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 10,
                                                   vector_length);
    return __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
}

/** @brief Convert e4m3 (m1) to f16 (m2) via sign-symmetric magnitude LUT. Sign bit 7 → f16 bit 15 (<<8). */
NK_INTERNAL vuint16m2_t nk_e4m3m1_to_f16m2_rvv_(vuint8m1_t e4m3_u8m1, nk_size_t vector_length) {
    static nk_u16_t const nk_e4m3_mag_to_f16_lut_[128] = {
        0x0000u, 0x1800u, 0x1C00u, 0x1E00u, 0x2000u, 0x2100u, 0x2200u, 0x2300u, /* [  0..  7] */
        0x2400u, 0x2480u, 0x2500u, 0x2580u, 0x2600u, 0x2680u, 0x2700u, 0x2780u, /* [  8.. 15] */
        0x2800u, 0x2880u, 0x2900u, 0x2980u, 0x2A00u, 0x2A80u, 0x2B00u, 0x2B80u, /* [ 16.. 23] */
        0x2C00u, 0x2C80u, 0x2D00u, 0x2D80u, 0x2E00u, 0x2E80u, 0x2F00u, 0x2F80u, /* [ 24.. 31] */
        0x3000u, 0x3080u, 0x3100u, 0x3180u, 0x3200u, 0x3280u, 0x3300u, 0x3380u, /* [ 32.. 39] */
        0x3400u, 0x3480u, 0x3500u, 0x3580u, 0x3600u, 0x3680u, 0x3700u, 0x3780u, /* [ 40.. 47] */
        0x3800u, 0x3880u, 0x3900u, 0x3980u, 0x3A00u, 0x3A80u, 0x3B00u, 0x3B80u, /* [ 48.. 55] */
        0x3C00u, 0x3C80u, 0x3D00u, 0x3D80u, 0x3E00u, 0x3E80u, 0x3F00u, 0x3F80u, /* [ 56.. 63] */
        0x4000u, 0x4080u, 0x4100u, 0x4180u, 0x4200u, 0x4280u, 0x4300u, 0x4380u, /* [ 64.. 71] */
        0x4400u, 0x4480u, 0x4500u, 0x4580u, 0x4600u, 0x4680u, 0x4700u, 0x4780u, /* [ 72.. 79] */
        0x4800u, 0x4880u, 0x4900u, 0x4980u, 0x4A00u, 0x4A80u, 0x4B00u, 0x4B80u, /* [ 80.. 87] */
        0x4C00u, 0x4C80u, 0x4D00u, 0x4D80u, 0x4E00u, 0x4E80u, 0x4F00u, 0x4F80u, /* [ 88.. 95] */
        0x5000u, 0x5080u, 0x5100u, 0x5180u, 0x5200u, 0x5280u, 0x5300u, 0x5380u, /* [ 96..103] */
        0x5400u, 0x5480u, 0x5500u, 0x5580u, 0x5600u, 0x5680u, 0x5700u, 0x5780u, /* [104..111] */
        0x5800u, 0x5880u, 0x5900u, 0x5980u, 0x5A00u, 0x5A80u, 0x5B00u, 0x5B80u, /* [112..119] */
        0x5C00u, 0x5C80u, 0x5D00u, 0x5D80u, 0x5E00u, 0x5E80u, 0x5F00u, 0x7E00u  /* [120..127] */
    };
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e4m3_u8m1, 0x80, vector_length);
    vuint8m1_t nonsign_u8m1 = __riscv_vand_vx_u8m1(e4m3_u8m1, 0x7F, vector_length);
    vuint16m2_t offsets_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(nonsign_u8m1, vector_length), 1,
                                                      vector_length);
    vuint16m2_t result_u16m2 = __riscv_vluxei16_v_u16m2(nk_e4m3_mag_to_f16_lut_, offsets_u16m2, vector_length);
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 8, vector_length);
    return __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
}

/** @brief Convert e2m3 (m1) to f16 (m2) via sign-symmetric magnitude LUT. Sign bit 5 → f16 bit 15 (<<10). */
NK_INTERNAL vuint16m2_t nk_e2m3m1_to_f16m2_rvv_(vuint8m1_t e2m3_u8m1, nk_size_t vector_length) {
    static nk_u16_t const nk_e2m3_mag_to_f16_lut_[32] = {
        0x0000u, 0x3000u, 0x3400u, 0x3600u, 0x3800u, 0x3900u, 0x3A00u, 0x3B00u, /* [  0..  7] */
        0x3C00u, 0x3C80u, 0x3D00u, 0x3D80u, 0x3E00u, 0x3E80u, 0x3F00u, 0x3F80u, /* [  8.. 15] */
        0x4000u, 0x4080u, 0x4100u, 0x4180u, 0x4200u, 0x4280u, 0x4300u, 0x4380u, /* [ 16.. 23] */
        0x4400u, 0x4480u, 0x4500u, 0x4580u, 0x4600u, 0x4680u, 0x4700u, 0x4780u  /* [ 24.. 31] */
    };
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e2m3_u8m1, 0x20, vector_length);
    vuint8m1_t nonsign_u8m1 = __riscv_vand_vx_u8m1(e2m3_u8m1, 0x1F, vector_length);
    vuint16m2_t offsets_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(nonsign_u8m1, vector_length), 1,
                                                      vector_length);
    vuint16m2_t result_u16m2 = __riscv_vluxei16_v_u16m2(nk_e2m3_mag_to_f16_lut_, offsets_u16m2, vector_length);
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 10,
                                                   vector_length);
    return __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
}

/** @brief Convert e3m2 (m1) to f16 (m2) via sign-symmetric magnitude LUT. Sign bit 5 → f16 bit 15 (<<10). */
NK_INTERNAL vuint16m2_t nk_e3m2m1_to_f16m2_rvv_(vuint8m1_t e3m2_u8m1, nk_size_t vector_length) {
    static nk_u16_t const nk_e3m2_mag_to_f16_lut_[32] = {
        0x0000u, 0x2C00u, 0x3000u, 0x3200u, 0x3400u, 0x3500u, 0x3600u, 0x3700u, /* [  0..  7] */
        0x3800u, 0x3900u, 0x3A00u, 0x3B00u, 0x3C00u, 0x3D00u, 0x3E00u, 0x3F00u, /* [  8.. 15] */
        0x4000u, 0x4100u, 0x4200u, 0x4300u, 0x4400u, 0x4500u, 0x4600u, 0x4700u, /* [ 16.. 23] */
        0x4800u, 0x4900u, 0x4A00u, 0x4B00u, 0x4C00u, 0x4D00u, 0x4E00u, 0x4F00u  /* [ 24.. 31] */
    };
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e3m2_u8m1, 0x20, vector_length);
    vuint8m1_t nonsign_u8m1 = __riscv_vand_vx_u8m1(e3m2_u8m1, 0x1F, vector_length);
    vuint16m2_t offsets_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(nonsign_u8m1, vector_length), 1,
                                                      vector_length);
    vuint16m2_t result_u16m2 = __riscv_vluxei16_v_u16m2(nk_e3m2_mag_to_f16_lut_, offsets_u16m2, vector_length);
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 10,
                                                   vector_length);
    return __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
}

/**
 *  @brief Unpack i4 (m1) nibbles to i8 (m2) register-to-register.
 *
 *  Packed format: byte[i] contains two nibbles:
 *  - High nibble (bits [7:4]) → output[i*2]
 *  - Low nibble (bits [3:0]) → output[i*2+1]
 *
 *  Sign extension: 4-bit signed value [-8,7] extended to 8-bit.
 *  Trick: (x ^ 8) - 8 sign-extends a 4-bit value to larger type.
 *
 *  Returns a tuple of two m1 vectors (high nibbles, low nibbles) for segment store.
 */
NK_INTERNAL vint8m1x2_t nk_i4m1_to_i8m2_rvv_(vuint8m1_t packed_u8m1, nk_size_t vector_length) {
    // Extract high nibble (even indices in output)
    vuint8m1_t high_u8m1 = __riscv_vsrl_vx_u8m1(packed_u8m1, 4, vector_length);
    // Sign extend: (x ^ 8) - 8
    vint8m1_t high_i8m1 = __riscv_vsub_vx_i8m1(
        __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(high_u8m1), 8, vector_length), 8, vector_length);

    // Extract low nibble (odd indices in output)
    vuint8m1_t low_u8m1 = __riscv_vand_vx_u8m1(packed_u8m1, 0x0F, vector_length);
    // Sign extend: (x ^ 8) - 8
    vint8m1_t low_i8m1 = __riscv_vsub_vx_i8m1(
        __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(low_u8m1), 8, vector_length), 8, vector_length);

    return __riscv_vcreate_v_i8m1x2(high_i8m1, low_i8m1);
}

/**
 *  @brief Unpack u4 (m1) nibbles to u8 (m2) register-to-register.
 *
 *  Returns a tuple of two m1 vectors (high nibbles, low nibbles) for segment store.
 */
NK_INTERNAL vuint8m1x2_t nk_u4m1_to_u8m2_rvv_(vuint8m1_t packed_u8m1, nk_size_t vector_length) {
    // Extract high nibble (even indices in output)
    vuint8m1_t high_u8m1 = __riscv_vsrl_vx_u8m1(packed_u8m1, 4, vector_length);

    // Extract low nibble (odd indices in output)
    vuint8m1_t low_u8m1 = __riscv_vand_vx_u8m1(packed_u8m1, 0x0F, vector_length);

    return __riscv_vcreate_v_u8m1x2(high_u8m1, low_u8m1);
}

/**
 *  @brief Pack i8 (m2) to i4 (m1) nibbles register-to-register.
 *
 *  Takes a tuple of two m1 vectors (high nibbles, low nibbles from segment load).
 *  Values are clamped to [-8, 7] before packing.
 */
NK_INTERNAL vuint8m1_t nk_i8m2_to_i4m1_rvv_(vint8m1_t high_i8m1, vint8m1_t low_i8m1, nk_size_t vector_length) {
    // Clamp to [-8, 7]
    high_i8m1 = __riscv_vmax_vx_i8m1(__riscv_vmin_vx_i8m1(high_i8m1, 7, vector_length), -8, vector_length);
    low_i8m1 = __riscv_vmax_vx_i8m1(__riscv_vmin_vx_i8m1(low_i8m1, 7, vector_length), -8, vector_length);

    // Convert to unsigned nibbles: value & 0x0F
    vuint8m1_t high_u4m1 = __riscv_vand_vx_u8m1(__riscv_vreinterpret_v_i8m1_u8m1(high_i8m1), 0x0F, vector_length);
    vuint8m1_t low_u4m1 = __riscv_vand_vx_u8m1(__riscv_vreinterpret_v_i8m1_u8m1(low_i8m1), 0x0F, vector_length);

    // Pack: (hi << 4) | lo
    return __riscv_vor_vv_u8m1(__riscv_vsll_vx_u8m1(high_u4m1, 4, vector_length), low_u4m1, vector_length);
}

/**
 *  @brief Pack u8 (m2) to u4 (m1) nibbles register-to-register.
 *
 *  Takes a tuple of two m1 vectors (high nibbles, low nibbles from segment load).
 *  Values are clamped to [0, 15] before packing.
 */
NK_INTERNAL vuint8m1_t nk_u8m2_to_u4m1_rvv_(vuint8m1_t high_u8m1, vuint8m1_t low_u8m1, nk_size_t vector_length) {
    // Clamp to [0, 15]
    high_u8m1 = __riscv_vminu_vx_u8m1(high_u8m1, 15, vector_length);
    low_u8m1 = __riscv_vminu_vx_u8m1(low_u8m1, 15, vector_length);

    // Pack: (hi << 4) | lo
    return __riscv_vor_vv_u8m1(__riscv_vsll_vx_u8m1(high_u8m1, 4, vector_length), low_u8m1, vector_length);
}

/**
 *  @brief Convert f32 (m4) to e4m3 (m1) register-to-register.
 *
 *  E4M3FN format: S EEEE MMM (1 sign, 4 exponent bits with bias=7, 3 mantissa bits)
 *  Handles normal, subnormal, overflow, and NaN. Uses RNE mantissa rounding.
 *  E4M3FN quirk: exp=15 with mant=7 is NaN (0x7F), so max finite is 0x7E (exp=15, mant=6).
 */
NK_INTERNAL vuint8m1_t nk_f32m4_to_e4m3m1_rvv_(vfloat32m4_t f32_f32m4, nk_size_t vector_length) {
    vuint32m4_t bits_u32m4 = __riscv_vreinterpret_v_f32m4_u32m4(f32_f32m4);
    vuint32m4_t sign_u32m4 = __riscv_vsrl_vx_u32m4(bits_u32m4, 31, vector_length);
    vuint32m4_t abs_bits_u32m4 = __riscv_vand_vx_u32m4(bits_u32m4, 0x7FFFFFFF, vector_length);
    vuint32m4_t f32_exp_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(bits_u32m4, 23, vector_length), 0xFF,
                                                      vector_length);

    // Round mantissa from 23 to 3 bits using RNE (round to nearest, ties to even)
    vuint32m4_t significand_u32m4 = __riscv_vor_vx_u32m4(__riscv_vand_vx_u32m4(bits_u32m4, 0x007FFFFF, vector_length),
                                                         0x00800000, vector_length);
    vuint32m4_t lsb_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(significand_u32m4, 20, vector_length), 1,
                                                  vector_length);
    vuint32m4_t rounding_bias_u32m4 = __riscv_vadd_vx_u32m4(lsb_u32m4, 0x0007FFFF, vector_length);
    vuint32m4_t rounded_sig_u32m4 = __riscv_vadd_vv_u32m4(significand_u32m4, rounding_bias_u32m4, vector_length);
    vuint32m4_t carry_u32m4 = __riscv_vsrl_vx_u32m4(rounded_sig_u32m4, 24, vector_length);
    vuint32m4_t f32_mantissa_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(rounded_sig_u32m4, 20, vector_length),
                                                           0x07, vector_length);
    // If carry, mantissa becomes 0 (rounded up to next power of 2)
    vbool8_t has_carry_b8 = __riscv_vmsne_vx_u32m4_b8(carry_u32m4, 0, vector_length);
    f32_mantissa_u32m4 = __riscv_vmerge_vxm_u32m4(f32_mantissa_u32m4, 0, has_carry_b8, vector_length);

    // e4m3_exp = f32_exp + carry - 120
    vint32m4_t e4m3_exp_i32m4 = __riscv_vsub_vx_i32m4(
        __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vadd_vv_u32m4(f32_exp_u32m4, carry_u32m4, vector_length)), 120,
        vector_length);

    // Detect subnormal (exp <= 0) and overflow (exp > 15)
    vbool8_t is_subnormal_b8 = __riscv_vmsle_vx_i32m4_b8(e4m3_exp_i32m4, 0, vector_length);
    vbool8_t is_overflow_b8 = __riscv_vmsgt_vx_i32m4_b8(e4m3_exp_i32m4, 15, vector_length);

    // Normal path: clamp exp to [1,15]
    vint32m4_t clamped_exp_i32m4 = __riscv_vmax_vx_i32m4(e4m3_exp_i32m4, 1, vector_length);
    clamped_exp_i32m4 = __riscv_vmin_vx_i32m4(clamped_exp_i32m4, 15, vector_length);
    // E4M3FN quirk: exp=15 with mant=7 is NaN, so cap mantissa to 6 when exp=15
    vbool8_t is_max_exp_b8 = __riscv_vmseq_vx_i32m4_b8(clamped_exp_i32m4, 15, vector_length);
    vuint32m4_t max_mant_u32m4 = __riscv_vmerge_vxm_u32m4(__riscv_vmv_v_x_u32m4(7, vector_length), 6, is_max_exp_b8,
                                                          vector_length);
    vuint32m4_t normal_mant_u32m4 = __riscv_vminu_vv_u32m4(f32_mantissa_u32m4, max_mant_u32m4, vector_length);
    // On overflow, saturate to max finite (exp=15, mant=6 = 0x7E with sign)
    normal_mant_u32m4 = __riscv_vmerge_vxm_u32m4(normal_mant_u32m4, 0x06, is_overflow_b8, vector_length);
    vuint32m4_t normal_u32m4 = __riscv_vor_vv_u32m4(
        __riscv_vsll_vx_u32m4(sign_u32m4, 7, vector_length),
        __riscv_vor_vv_u32m4(
            __riscv_vsll_vx_u32m4(__riscv_vreinterpret_v_i32m4_u32m4(clamped_exp_i32m4), 3, vector_length),
            normal_mant_u32m4, vector_length),
        vector_length);

    // Subnormal path: mantissa = round(|f32| * 512)
    vfloat32m4_t abs_f32m4 = __riscv_vreinterpret_v_u32m4_f32m4(abs_bits_u32m4);
    vfloat32m4_t scaled_f32m4 = __riscv_vfmul_vf_f32m4(abs_f32m4, 512.0f, vector_length);
    vint32m4_t subnorm_mant_i32m4 = __riscv_vfcvt_x_f_v_i32m4(scaled_f32m4, vector_length); // RNE rounding
    // If rounds to 8+, promote to first normal (exp=1, mant=0 = 0x08)
    vbool8_t promotes_b8 = __riscv_vmsgt_vx_i32m4_b8(subnorm_mant_i32m4, 7, vector_length);
    subnorm_mant_i32m4 = __riscv_vmin_vx_i32m4(subnorm_mant_i32m4, 7, vector_length);
    subnorm_mant_i32m4 = __riscv_vmax_vx_i32m4(subnorm_mant_i32m4, 0, vector_length);
    vuint32m4_t subnorm_u32m4 = __riscv_vor_vv_u32m4(__riscv_vsll_vx_u32m4(sign_u32m4, 7, vector_length),
                                                     __riscv_vreinterpret_v_i32m4_u32m4(subnorm_mant_i32m4),
                                                     vector_length);
    vuint32m4_t first_normal_u32m4 = __riscv_vor_vx_u32m4(__riscv_vsll_vx_u32m4(sign_u32m4, 7, vector_length), 0x08,
                                                          vector_length);
    subnorm_u32m4 = __riscv_vmerge_vvm_u32m4(subnorm_u32m4, first_normal_u32m4, promotes_b8, vector_length);

    // Select: subnormal when exp <= 0, else normal
    vuint32m4_t result_u32m4 = __riscv_vmerge_vvm_u32m4(normal_u32m4, subnorm_u32m4, is_subnormal_b8, vector_length);

    // Handle NaN: f32 NaN (abs_bits > 0x7F800000) → e4m3 NaN (sign | 0x7F)
    vbool8_t is_nan_b8 = __riscv_vmsgtu_vx_u32m4_b8(abs_bits_u32m4, 0x7F800000, vector_length);
    vuint32m4_t nan_u32m4 = __riscv_vor_vx_u32m4(__riscv_vsll_vx_u32m4(sign_u32m4, 7, vector_length), 0x7F,
                                                 vector_length);
    result_u32m4 = __riscv_vmerge_vvm_u32m4(result_u32m4, nan_u32m4, is_nan_b8, vector_length);

    // Narrow u32m4 → u16m2 → u8m1
    vuint16m2_t result_u16m2 = __riscv_vncvt_x_x_w_u16m2(result_u32m4, vector_length);
    return __riscv_vncvt_x_x_w_u8m1(result_u16m2, vector_length);
}

/**
 *  @brief Convert f32 (m4) to e5m2 (m1) register-to-register.
 *
 *  E5M2 format: S EEEEE MM (1 sign, 5 exponent bits with bias=15, 2 mantissa bits)
 *  Handles normal, subnormal, overflow (→ infinity), and NaN. Uses RNE mantissa rounding.
 */
NK_INTERNAL vuint8m1_t nk_f32m4_to_e5m2m1_rvv_(vfloat32m4_t f32_f32m4, nk_size_t vector_length) {
    vuint32m4_t bits_u32m4 = __riscv_vreinterpret_v_f32m4_u32m4(f32_f32m4);
    vuint32m4_t sign_u32m4 = __riscv_vsrl_vx_u32m4(bits_u32m4, 31, vector_length);
    vuint32m4_t abs_bits_u32m4 = __riscv_vand_vx_u32m4(bits_u32m4, 0x7FFFFFFF, vector_length);
    vuint32m4_t f32_exp_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(bits_u32m4, 23, vector_length), 0xFF,
                                                      vector_length);

    // Round mantissa from 23 to 2 bits using RNE
    vuint32m4_t significand_u32m4 = __riscv_vor_vx_u32m4(__riscv_vand_vx_u32m4(bits_u32m4, 0x007FFFFF, vector_length),
                                                         0x00800000, vector_length);
    vuint32m4_t lsb_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(significand_u32m4, 21, vector_length), 1,
                                                  vector_length);
    vuint32m4_t rounding_bias_u32m4 = __riscv_vadd_vx_u32m4(lsb_u32m4, 0x000FFFFF, vector_length);
    vuint32m4_t rounded_sig_u32m4 = __riscv_vadd_vv_u32m4(significand_u32m4, rounding_bias_u32m4, vector_length);
    vuint32m4_t carry_u32m4 = __riscv_vsrl_vx_u32m4(rounded_sig_u32m4, 24, vector_length);
    vuint32m4_t f32_mantissa_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(rounded_sig_u32m4, 21, vector_length),
                                                           0x03, vector_length);
    vbool8_t has_carry_b8 = __riscv_vmsne_vx_u32m4_b8(carry_u32m4, 0, vector_length);
    f32_mantissa_u32m4 = __riscv_vmerge_vxm_u32m4(f32_mantissa_u32m4, 0, has_carry_b8, vector_length);

    // e5m2_exp = f32_exp + carry - 112
    vint32m4_t e5m2_exp_i32m4 = __riscv_vsub_vx_i32m4(
        __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vadd_vv_u32m4(f32_exp_u32m4, carry_u32m4, vector_length)), 112,
        vector_length);

    // Detect subnormal (exp <= 0) and overflow (exp > 31)
    vbool8_t is_subnormal_b8 = __riscv_vmsle_vx_i32m4_b8(e5m2_exp_i32m4, 0, vector_length);
    vbool8_t is_overflow_b8 = __riscv_vmsgt_vx_i32m4_b8(e5m2_exp_i32m4, 31, vector_length);

    // Normal path: clamp exp to [1,31], on overflow return infinity (exp=31, mant=0)
    vint32m4_t clamped_exp_i32m4 = __riscv_vmax_vx_i32m4(e5m2_exp_i32m4, 1, vector_length);
    clamped_exp_i32m4 = __riscv_vmin_vx_i32m4(clamped_exp_i32m4, 31, vector_length);
    vuint32m4_t normal_mant_u32m4 = __riscv_vmerge_vxm_u32m4(f32_mantissa_u32m4, 0, is_overflow_b8, vector_length);
    vuint32m4_t normal_u32m4 = __riscv_vor_vv_u32m4(
        __riscv_vsll_vx_u32m4(sign_u32m4, 7, vector_length),
        __riscv_vor_vv_u32m4(
            __riscv_vsll_vx_u32m4(__riscv_vreinterpret_v_i32m4_u32m4(clamped_exp_i32m4), 2, vector_length),
            normal_mant_u32m4, vector_length),
        vector_length);

    // Subnormal path: mantissa = round(|f32| * 65536)
    vfloat32m4_t abs_f32m4 = __riscv_vreinterpret_v_u32m4_f32m4(abs_bits_u32m4);
    vfloat32m4_t scaled_f32m4 = __riscv_vfmul_vf_f32m4(abs_f32m4, 65536.0f, vector_length);
    vint32m4_t subnorm_mant_i32m4 = __riscv_vfcvt_x_f_v_i32m4(scaled_f32m4, vector_length);
    vbool8_t promotes_b8 = __riscv_vmsgt_vx_i32m4_b8(subnorm_mant_i32m4, 3, vector_length);
    subnorm_mant_i32m4 = __riscv_vmin_vx_i32m4(subnorm_mant_i32m4, 3, vector_length);
    subnorm_mant_i32m4 = __riscv_vmax_vx_i32m4(subnorm_mant_i32m4, 0, vector_length);
    vuint32m4_t subnorm_u32m4 = __riscv_vor_vv_u32m4(__riscv_vsll_vx_u32m4(sign_u32m4, 7, vector_length),
                                                     __riscv_vreinterpret_v_i32m4_u32m4(subnorm_mant_i32m4),
                                                     vector_length);
    vuint32m4_t first_normal_u32m4 = __riscv_vor_vx_u32m4(__riscv_vsll_vx_u32m4(sign_u32m4, 7, vector_length), 0x04,
                                                          vector_length);
    subnorm_u32m4 = __riscv_vmerge_vvm_u32m4(subnorm_u32m4, first_normal_u32m4, promotes_b8, vector_length);

    // Select: subnormal when exp <= 0, else normal
    vuint32m4_t result_u32m4 = __riscv_vmerge_vvm_u32m4(normal_u32m4, subnorm_u32m4, is_subnormal_b8, vector_length);

    // Handle NaN: f32 NaN (abs_bits > 0x7F800000) → e5m2 NaN (sign | 0x7D)
    vbool8_t is_nan_b8 = __riscv_vmsgtu_vx_u32m4_b8(abs_bits_u32m4, 0x7F800000, vector_length);
    vuint32m4_t nan_u32m4 = __riscv_vor_vx_u32m4(__riscv_vsll_vx_u32m4(sign_u32m4, 7, vector_length), 0x7D,
                                                 vector_length);
    result_u32m4 = __riscv_vmerge_vvm_u32m4(result_u32m4, nan_u32m4, is_nan_b8, vector_length);

    // Narrow u32m4 → u16m2 → u8m1
    vuint16m2_t result_u16m2 = __riscv_vncvt_x_x_w_u16m2(result_u32m4, vector_length);
    return __riscv_vncvt_x_x_w_u8m1(result_u16m2, vector_length);
}

#pragma endregion Register - to - Register Helpers

#pragma region Unified Cast Dispatcher

NK_PUBLIC void nk_cast_rvv(void const *from, nk_dtype_t from_type, nk_size_t count, void *to, nk_dtype_t to_type) {
    // bf16 → f32
    if (from_type == nk_bf16_k && to_type == nk_f32_k) {
        nk_bf16_t const *source = (nk_bf16_t const *)from;
        nk_f32_t *destination = (nk_f32_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e16m1(count);
            vuint16m1_t bf16_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)source, vector_length);
            vfloat32m2_t f32_f32m2 = nk_bf16m1_to_f32m2_rvv_(bf16_u16m1, vector_length);
            __riscv_vse32_v_f32m2(destination, f32_f32m2, vector_length);
        }
        return;
    }

    // f32 → bf16
    if (from_type == nk_f32_k && to_type == nk_bf16_k) {
        nk_f32_t const *source = (nk_f32_t const *)from;
        nk_bf16_t *destination = (nk_bf16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e32m2(count);
            vfloat32m2_t f32_f32m2 = __riscv_vle32_v_f32m2(source, vector_length);
            vuint16m1_t bf16_u16m1 = nk_f32m2_to_bf16m1_rvv_(f32_f32m2, vector_length);
            __riscv_vse16_v_u16m1((nk_u16_t *)destination, bf16_u16m1, vector_length);
        }
        return;
    }

    // f16 → f32
    if (from_type == nk_f16_k && to_type == nk_f32_k) {
        nk_f16_t const *source = (nk_f16_t const *)from;
        nk_f32_t *destination = (nk_f32_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e16m1(count);
            vuint16m1_t f16_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)source, vector_length);
            vfloat32m2_t f32_f32m2 = nk_f16m1_to_f32m2_rvv_(f16_u16m1, vector_length);
            __riscv_vse32_v_f32m2(destination, f32_f32m2, vector_length);
        }
        return;
    }

    // f32 → f16
    if (from_type == nk_f32_k && to_type == nk_f16_k) {
        nk_f32_t const *source = (nk_f32_t const *)from;
        nk_f16_t *destination = (nk_f16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e32m2(count);
            vfloat32m2_t f32_f32m2 = __riscv_vle32_v_f32m2(source, vector_length);
            vuint16m1_t f16_u16m1 = nk_f32m2_to_f16m1_rvv_(f32_f32m2, vector_length);
            __riscv_vse16_v_u16m1((nk_u16_t *)destination, f16_u16m1, vector_length);
        }
        return;
    }

    // e4m3 → f32
    if (from_type == nk_e4m3_k && to_type == nk_f32_k) {
        nk_e4m3_t const *source = (nk_e4m3_t const *)from;
        nk_f32_t *destination = (nk_f32_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e4m3_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vfloat32m4_t f32_f32m4 = nk_e4m3m1_to_f32m4_rvv_(e4m3_u8m1, vector_length);
            __riscv_vse32_v_f32m4(destination, f32_f32m4, vector_length);
        }
        return;
    }

    // e5m2 → f32
    if (from_type == nk_e5m2_k && to_type == nk_f32_k) {
        nk_e5m2_t const *source = (nk_e5m2_t const *)from;
        nk_f32_t *destination = (nk_f32_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e5m2_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vfloat32m4_t f32_f32m4 = nk_e5m2m1_to_f32m4_rvv_(e5m2_u8m1, vector_length);
            __riscv_vse32_v_f32m4(destination, f32_f32m4, vector_length);
        }
        return;
    }

    // e2m3 → f32
    if (from_type == nk_e2m3_k && to_type == nk_f32_k) {
        nk_e2m3_t const *source = (nk_e2m3_t const *)from;
        nk_f32_t *destination = (nk_f32_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e2m3_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vfloat32m4_t f32_f32m4 = nk_e2m3m1_to_f32m4_rvv_(e2m3_u8m1, vector_length);
            __riscv_vse32_v_f32m4(destination, f32_f32m4, vector_length);
        }
        return;
    }

    // e3m2 → f32
    if (from_type == nk_e3m2_k && to_type == nk_f32_k) {
        nk_e3m2_t const *source = (nk_e3m2_t const *)from;
        nk_f32_t *destination = (nk_f32_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e3m2_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vfloat32m4_t f32_f32m4 = nk_e3m2m1_to_f32m4_rvv_(e3m2_u8m1, vector_length);
            __riscv_vse32_v_f32m4(destination, f32_f32m4, vector_length);
        }
        return;
    }

    // e4m3 → bf16
    if (from_type == nk_e4m3_k && to_type == nk_bf16_k) {
        nk_e4m3_t const *source = (nk_e4m3_t const *)from;
        nk_bf16_t *destination = (nk_bf16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e4m3_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint16m2_t bf16_u16m2 = nk_e4m3m1_to_bf16m2_rvv_(e4m3_u8m1, vector_length);
            __riscv_vse16_v_u16m2((nk_u16_t *)destination, bf16_u16m2, vector_length);
        }
        return;
    }

    // e5m2 → bf16
    if (from_type == nk_e5m2_k && to_type == nk_bf16_k) {
        nk_e5m2_t const *source = (nk_e5m2_t const *)from;
        nk_bf16_t *destination = (nk_bf16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e5m2_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint16m2_t bf16_u16m2 = nk_e5m2m1_to_bf16m2_rvv_(e5m2_u8m1, vector_length);
            __riscv_vse16_v_u16m2((nk_u16_t *)destination, bf16_u16m2, vector_length);
        }
        return;
    }

    // e2m3 → bf16
    if (from_type == nk_e2m3_k && to_type == nk_bf16_k) {
        nk_e2m3_t const *source = (nk_e2m3_t const *)from;
        nk_bf16_t *destination = (nk_bf16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e2m3_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint16m2_t bf16_u16m2 = nk_e2m3m1_to_bf16m2_rvv_(e2m3_u8m1, vector_length);
            __riscv_vse16_v_u16m2((nk_u16_t *)destination, bf16_u16m2, vector_length);
        }
        return;
    }

    // e3m2 → bf16
    if (from_type == nk_e3m2_k && to_type == nk_bf16_k) {
        nk_e3m2_t const *source = (nk_e3m2_t const *)from;
        nk_bf16_t *destination = (nk_bf16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e3m2_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint16m2_t bf16_u16m2 = nk_e3m2m1_to_bf16m2_rvv_(e3m2_u8m1, vector_length);
            __riscv_vse16_v_u16m2((nk_u16_t *)destination, bf16_u16m2, vector_length);
        }
        return;
    }

    // e4m3 → f16
    if (from_type == nk_e4m3_k && to_type == nk_f16_k) {
        nk_e4m3_t const *source = (nk_e4m3_t const *)from;
        nk_f16_t *destination = (nk_f16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e4m3_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint16m2_t f16_u16m2 = nk_e4m3m1_to_f16m2_rvv_(e4m3_u8m1, vector_length);
            __riscv_vse16_v_u16m2((nk_u16_t *)destination, f16_u16m2, vector_length);
        }
        return;
    }

    // e2m3 → f16
    if (from_type == nk_e2m3_k && to_type == nk_f16_k) {
        nk_e2m3_t const *source = (nk_e2m3_t const *)from;
        nk_f16_t *destination = (nk_f16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e2m3_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint16m2_t f16_u16m2 = nk_e2m3m1_to_f16m2_rvv_(e2m3_u8m1, vector_length);
            __riscv_vse16_v_u16m2((nk_u16_t *)destination, f16_u16m2, vector_length);
        }
        return;
    }

    // e3m2 → f16
    if (from_type == nk_e3m2_k && to_type == nk_f16_k) {
        nk_e3m2_t const *source = (nk_e3m2_t const *)from;
        nk_f16_t *destination = (nk_f16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e3m2_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint16m2_t f16_u16m2 = nk_e3m2m1_to_f16m2_rvv_(e3m2_u8m1, vector_length);
            __riscv_vse16_v_u16m2((nk_u16_t *)destination, f16_u16m2, vector_length);
        }
        return;
    }

    // i4 → i8
    if (from_type == nk_i4_k && to_type == nk_i8_k) {
        nk_i4x2_t const *source = (nk_i4x2_t const *)from;
        nk_i8_t *destination = (nk_i8_t *)to;
        nk_size_t n_bytes = count / 2;
        for (nk_size_t vector_length; n_bytes > 0;
             n_bytes -= vector_length, source += vector_length, destination += vector_length * 2) {
            vector_length = __riscv_vsetvl_e8m1(n_bytes);
            vuint8m1_t packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vint8m1x2_t unpacked_i8m1x2 = nk_i4m1_to_i8m2_rvv_(packed_u8m1, vector_length);
            __riscv_vsseg2e8_v_i8m1x2(destination, unpacked_i8m1x2, vector_length);
        }
        // Odd count: the final element (the high nibble of byte count/2) is not covered by the
        // count/2 full-byte loop. Unpack it to match the serial oracle (high nibble first).
        if (count & 1) {
            int nibble = (*(nk_u8_t const *)source >> 4) & 0x0F;
            *(nk_i8_t *)destination = (nk_i8_t)((nibble & 0x08) ? (nibble | 0xF0) : nibble);
        }
        return;
    }

    // u4 → u8
    if (from_type == nk_u4_k && to_type == nk_u8_k) {
        nk_u4x2_t const *source = (nk_u4x2_t const *)from;
        nk_u8_t *destination = (nk_u8_t *)to;
        nk_size_t n_bytes = count / 2;
        for (nk_size_t vector_length; n_bytes > 0;
             n_bytes -= vector_length, source += vector_length, destination += vector_length * 2) {
            vector_length = __riscv_vsetvl_e8m1(n_bytes);
            vuint8m1_t packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint8m1x2_t unpacked_u8m1x2 = nk_u4m1_to_u8m2_rvv_(packed_u8m1, vector_length);
            __riscv_vsseg2e8_v_u8m1x2(destination, unpacked_u8m1x2, vector_length);
        }
        // Odd count: unpack the trailing high nibble of byte count/2 (high nibble first).
        if (count & 1) *(nk_u8_t *)destination = (nk_u8_t)((*(nk_u8_t const *)source >> 4) & 0x0F);
        return;
    }

    // i8 → i4
    if (from_type == nk_i8_k && to_type == nk_i4_k) {
        nk_i8_t const *source = (nk_i8_t const *)from;
        nk_i4x2_t *destination = (nk_i4x2_t *)to;
        nk_size_t n_bytes = count / 2;
        for (nk_size_t vector_length; n_bytes > 0;
             n_bytes -= vector_length, source += vector_length * 2, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(n_bytes);
            vint8m1x2_t loaded_i8m1x2 = __riscv_vlseg2e8_v_i8m1x2(source, vector_length);
            vint8m1_t high_i8m1 = __riscv_vget_v_i8m1x2_i8m1(loaded_i8m1x2, 0);
            vint8m1_t low_i8m1 = __riscv_vget_v_i8m1x2_i8m1(loaded_i8m1x2, 1);
            vuint8m1_t packed_u8m1 = nk_i8m2_to_i4m1_rvv_(high_i8m1, low_i8m1, vector_length);
            __riscv_vse8_v_u8m1((nk_u8_t *)destination, packed_u8m1, vector_length);
        }
        // Odd count: pack the trailing element into the high nibble of byte count/2 (low nibble 0),
        // clamping to [-8, 7] to match nk_i8m2_to_i4m1_rvv_.
        if (count & 1) {
            int v = *(nk_i8_t const *)source;
            v = v > 7 ? 7 : (v < -8 ? -8 : v);
            *(nk_u8_t *)destination = (nk_u8_t)((v & 0x0F) << 4);
        }
        return;
    }

    // u8 → u4
    if (from_type == nk_u8_k && to_type == nk_u4_k) {
        nk_u8_t const *source = (nk_u8_t const *)from;
        nk_u4x2_t *destination = (nk_u4x2_t *)to;
        nk_size_t n_bytes = count / 2;
        for (nk_size_t vector_length; n_bytes > 0;
             n_bytes -= vector_length, source += vector_length * 2, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(n_bytes);
            vuint8m1x2_t loaded_u8m1x2 = __riscv_vlseg2e8_v_u8m1x2(source, vector_length);
            vuint8m1_t high_u8m1 = __riscv_vget_v_u8m1x2_u8m1(loaded_u8m1x2, 0);
            vuint8m1_t low_u8m1 = __riscv_vget_v_u8m1x2_u8m1(loaded_u8m1x2, 1);
            vuint8m1_t packed_u8m1 = nk_u8m2_to_u4m1_rvv_(high_u8m1, low_u8m1, vector_length);
            __riscv_vse8_v_u8m1((nk_u8_t *)destination, packed_u8m1, vector_length);
        }
        // Odd count: pack the trailing element into the high nibble of byte count/2 (low nibble 0),
        // clamping to [0, 15] to match nk_u8m2_to_u4m1_rvv_.
        if (count & 1) {
            unsigned v = *(nk_u8_t const *)source;
            *(nk_u8_t *)destination = (nk_u8_t)((v > 15 ? 15 : v) << 4);
        }
        return;
    }

    // Fallback to serial for unimplemented conversions
    nk_cast_serial(from, from_type, count, to, to_type);
}

#pragma endregion Unified Cast Dispatcher

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
#endif // NK_CAST_RVV_H
